# ============================================================
# engine/learn_driver.jl (FINAL: SINTAXE E ESCOPO CORRIGIDOS)
# ============================================================
using Flux
using LinearAlgebra
using Printf
using Zygote

# Importação condicional de CUDA
let
    try
        @eval using CUDA
            @eval const _cuda_available = CUDA.has_cuda()
    catch e
        @warn "CUDA não disponível em learn_driver.jl, usando apenas CPU" error=e
        @eval const _cuda_available = false
        @eval module CUDA
            has_cuda() = false
            CuArray = Array
            set_runtime_version!(v) = nothing
            cu(x) = x  # Função identidade quando CUDA não disponível
        end
    end
end

function sparsity_loss(coeffs_detail)
    loss = 0.0f0
    count = 0.0f0
    for coeffs in coeffs_detail
        loss += sum(abs, coeffs)
        count += Float32(length(coeffs))
    end
    return count > 0 ? loss / count : loss
end

# CORREÇÃO: Assinatura de função com sintaxe Julia válida (sem \ no final da linha)
function optimize_wavelet_sparsity(
    data::AbstractVector{<:Number};
    L::Int,
    chi::Int,
    chimid::Int,
    opts::Dict,
    initial::Union{Nothing,NamedTuple}=nothing
)
    # As funções auxiliares (init_random_tensors, TensorUpdateSVD) são
    # acessíveis diretamente pois foram incluídas no módulo Wave6G.
    
    numiter = get(opts, "numiter", 200)
    lr      = get(opts, "lr", 0.001)

    # 1. Inicialização das Matrizes 
    if initial !== nothing
        @assert haskey(initial, :wC) && haskey(initial, :vC) "Initial tensors must include wC and vC"
        w_base = initial.wC
        v_base = initial.vC
        u_base = haskey(initial, :uC) ? initial.uC : nothing
    else
        w_init, v_init, u_init = Base.invokelatest(init_random_tensors, L, chi, chimid)
        w_base = w_init
        v_base = v_init
        u_base = u_init
    end

    if u_base === nothing
        _, _, u_tmp = Base.invokelatest(init_random_tensors, L, chi, chimid)
        u_base = u_tmp
    end

    @assert length(w_base) == L "Inconsistent length for wC"
    @assert length(v_base) == L "Inconsistent length for vC"
    @assert length(u_base) == L "Inconsistent length for uC"

    wC = [ComplexF32.(_cuda_available ? cu(w) : w) for w in w_base]
    vC = [ComplexF32.(_cuda_available ? cu(v) : v) for v in v_base]
    uC = [ComplexF32.(_cuda_available ? cu(u) : u) for u in u_base]
    data_complex = _cuda_available ? cu(ComplexF32.(data)) : ComplexF32.(data)
        @info "Tipo de wC[1]: $(typeof(wC[1]))"
        @info "Tipo de vC[1]: $(typeof(vC[1]))"
        @info "Tipo de uC[1]: $(typeof(uC[1]))"
        @info "Tipo de data_complex: $(typeof(data_complex))"

    # 2. Setup do Otimizador (Adam)
    opt = Flux.Adam(lr)

    # 3. Empacota os tensores (Parâmetros) no formato aceito pela API moderna do Flux
    params = (wC = wC, vC = vC)

    # Inicializa o estado do otimizador (padrão Flux moderno)
    opt_state = Flux.setup(opt, params)

    zero_like_array(A) = fill!(similar(A), zero(eltype(A)))

    function sanitize_grads(params_named, grads_named)
        w_grads = grads_named.wC
        if w_grads === nothing
            w_grads = [zero_like_array(p) for p in params_named.wC]
        else
            w_grads = [g === nothing ? zero_like_array(p) : g for (g, p) in zip(w_grads, params_named.wC)]
        end

        v_grads = grads_named.vC
        if v_grads === nothing
            v_grads = [zero_like_array(p) for p in params_named.vC]
        else
            v_grads = [g === nothing ? zero_like_array(p) : g for (g, p) in zip(v_grads, params_named.vC)]
        end

        return (wC = w_grads, vC = v_grads)
    end

    sparsity_weight = Float32(get(opts, "sparsity_weight", 1.0))
    mse_weight = Float32(get(opts, "mse_weight", 0.0))

    @info "Iniciando otimização AD com Adam por $numiter iterações (LR=$lr)." mse_weight=mse_weight sparsity_weight=sparsity_weight

    final_loss = 0.0f0

    # Verificar se podemos usar múltiplas streams
    streams = _cuda_available ? Wave6G._get_cuda_streams() : nothing
    use_streams = streams !== nothing && length(streams) >= 3

    # 4. Loop de Otimização com paralelização CUDA
    for k in 1:numiter

        # === Forward Pass e Gradiente Automático (AD) ===
        loss, raw_grads = Flux.withgradient(params) do p
            coeffs_approx, coeffs_detail = mera_analyze(data_complex, p.wC, p.vC)
            sparsity_term = sparsity_loss(coeffs_detail)
            mse_term = 0.0f0
            if mse_weight > 0
                reconstructed = mera_synthesize(coeffs_approx, coeffs_detail, p.wC, p.vC)
                diff = reconstructed .- data_complex
                mse_term = mean(abs2, diff)
            end
            return sparsity_weight * sparsity_term + mse_weight * mse_term
        end
        grads = sanitize_grads(params, raw_grads[1])

        # === Aplicação do Gradiente ===
        opt_state, params = Flux.update!(opt_state, params, grads)
        wC = params.wC
        vC = params.vC

        # === Projeção de Isometria (Restrição MERA) com paralelização ===
        if use_streams && L > 1
            # Paralelizar projeções SVD usando múltiplas streams
            tasks = []
            for p in 1:L
                stream_idx = mod1(p, length(streams))
                push!(tasks, @async CUDA.stream!(streams[stream_idx]) do
                    wC[p] = TensorUpdateSVD(wC[p], 1)
                    vC[p] = TensorUpdateSVD(vC[p], 1)
                    (p, wC[p], vC[p])
                end)
            end

            # Aguardar todas as projeções terminarem
            for task in tasks
                p, w_updated, v_updated = fetch(task)
                wC[p] = w_updated
                vC[p] = v_updated
            end
        else
            # Projeção com Threads na CPU (fallback paralelizado)
            # Permite desabilitar paralelismo intra-janela quando usando paralelismo entre janelas
            # via variável de ambiente: W6G_INTRA_THREADS=0
            intra_threads_enabled = get(ENV, "W6G_INTRA_THREADS", "1") != "0"
            if !_cuda_available && Threads.nthreads() > 1 && L > 1 && intra_threads_enabled
                Threads.@threads for p in 1:L
                    wC[p] = TensorUpdateSVD(wC[p], 1)
                    vC[p] = TensorUpdateSVD(vC[p], 1)
                end
            else
                for p in 1:L
                    wC[p] = TensorUpdateSVD(wC[p], 1)
                    vC[p] = TensorUpdateSVD(vC[p], 1)
                end
            end
        end

        params = (wC = wC, vC = vC)

        final_loss = Float32(loss)

        # Progress logging with percentage
        local step = max(1, cld(numiter, 10)) # ~10 updates per stage
        if k == 1 || k % step == 0 || k == numiter
            pct = 100.0 * k / numiter
            @printf "Iteração: %d/%d (%.1f%%), Loss: %.6f\n" k numiter pct final_loss
        end
    end

    return wC, vC, uC, final_loss
end
