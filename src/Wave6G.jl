# ============================================================
# src/Wave6G.jl  —  Módulo principal (Versão FINAL E CONSOLIDADA)
# ============================================================

__precompile__(true)

module Wave6G

# ----------------------------
# Deps padrão e externos
# ----------------------------
using LinearAlgebra
using Statistics
using Printf
using Random
using FilePathsBase # Para inclusão de engine
using Zygote

using TensorOperations: ncon
using JLD2
using Flux

# Importação condicional de CUDA
let
    try
        @eval using CUDA
        @eval const _cuda_available = CUDA.has_cuda()
        @eval const ComplexStorage = Union{Array{ComplexF32,N} where N, CUDA.CuArray{ComplexF32,N} where N}
    catch e
        @warn "CUDA não disponível, usando apenas CPU" error=e
        @eval const _cuda_available = false
        @eval const ComplexStorage = Array{ComplexF32,N} where N
        @eval module CUDA
            has_cuda() = false
            CuArray = Array
            set_runtime_version!(v) = nothing
            cu(x) = x  # Função identidade quando CUDA não disponível
        end
    end
end

using ChainRulesCore

function _configure_cuda_compiler!()
    _cuda_available || return
    try
        dev = CUDA.device()
        default_cfg = CUDA.compiler_config(dev)
        custom_cfg = CUDA.compiler_config(dev; debuginfo=false)
        for (key, cfg) in CUDA._compiler_configs
            if cfg === default_cfg
                CUDA._compiler_configs[key] = custom_cfg
            end
        end
    catch err
        @warn "Nao foi possivel ajustar configuracao do compilador CUDA para remover debug info." error=err
    end
end

# ----------------------------
# CUDA Streams e Otimizações
# ----------------------------
const _cuda_streams = Ref{Union{Nothing,Vector{CUDA.CuStream}}}(nothing)

function _get_cuda_streams()
    if _cuda_available && _cuda_streams[] === nothing
        # Criar múltiplos streams para operações assíncronas
        num_streams = min(4, CUDA.ndevices() * 2)  # Até 4 streams por GPU
        _cuda_streams[] = [CUDA.CuStream() for _ in 1:num_streams]
        @info "CUDA Streams inicializados: $num_streams streams criados"
    end
    return _cuda_streams[]
end

function _get_stream(idx::Int=1)
    streams = _get_cuda_streams()
    streams === nothing && return nothing
    return streams[mod1(idx, length(streams))]
end

_configure_cuda_compiler!()

# ----------------------------
# Cache Inteligente para Experimentos
# ----------------------------
const _computation_cache = Dict{String, Any}()
const _cache_hits = Ref(0)
const _cache_misses = Ref(0)

function _cache_key(func::Symbol, args...)
    # Criar chave única baseada na função e argumentos principais
    key_parts = [string(func)]
    for arg in args
        if arg isa Number
            push!(key_parts, string(round(arg, digits=6)))
        elseif arg isa AbstractArray
            # Hash do array para arrays grandes
            push!(key_parts, string(hash(vec(arg))))
        else
            push!(key_parts, string(arg))
        end
    end
    return join(key_parts, "|")
end

function _get_cached_result(key::String)
    result = get(_computation_cache, key, nothing)
    if result !== nothing
        _cache_hits[] += 1
        @debug "Cache HIT: $key"
    else
        _cache_misses[] += 1
        @debug "Cache MISS: $key"
    end
    return result
end

function _set_cached_result(key::String, result)
    _computation_cache[key] = result
    # Limitar tamanho do cache para evitar uso excessivo de memória
    if length(_computation_cache) > 1000
        # Remover entradas antigas (simples LRU approximation)
        keys_to_remove = collect(keys(_computation_cache))[1:100]
        for k in keys_to_remove
            delete!(_computation_cache, k)
        end
    end
end

function clear_computation_cache!()
    empty!(_computation_cache)
    _cache_hits[] = 0
    _cache_misses[] = 0
    @info "Cache de computação limpo"
end

function get_cache_stats()
    total = _cache_hits[] + _cache_misses[]
    hit_rate = total > 0 ? round(_cache_hits[] / total * 100, digits=1) : 0.0
    return (hits=_cache_hits[], misses=_cache_misses[], hit_rate=hit_rate)
end

include("MAWI.jl")
using .MAWI

# ----------------------------
# Constantes e Modos
# ----------------------------
const _engine_mode = Ref(:STUB)

# ----------------------------
# Otimizações Numéricas e Precision Mista
# ----------------------------
const _use_mixed_precision = Ref(true)  # Usar Float16/Float32 quando apropriado
const _use_blas_optimization = Ref(true) # Otimizar BLAS

function enable_mixed_precision!(enable::Bool=true)
    _use_mixed_precision[] = enable
    if enable
        @info "Precision mista habilitada (Float16/Float32)"
    else
        @info "Precision mista desabilitada (Float32 apenas)"
    end
end

function enable_blas_optimization!(enable::Bool=true)
    _use_blas_optimization[] = enable
    if enable && _cuda_available
        # Configurar CUDA para usar Tensor Cores quando disponível
        try
            CUDA.math_mode!(CUDA.MATH_MODE_DEFAULT)
            @info "BLAS otimizado habilitado com Tensor Cores"
        catch e
            @warn "Não foi possível configurar Tensor Cores" error=e
        end
    elseif enable
        @info "BLAS otimizado habilitado (CPU)"
    else
        @info "BLAS otimizado desabilitado"
    end
end

# Inicializar otimizações
enable_mixed_precision!()
enable_blas_optimization!()

# ----------------------------
# Exports
# ----------------------------
export zscore, overlap_windows, mera_analyze, mera_synthesize,
       psnr, parseval_test, learn_wavelet_filters,
       run_learning_validation,
       variational_default_schedule, prepare_variational_state,
       optimize_variational_schedule!, VariationalState,
       load_mawi_series, truncate_dyadic, window_series,
       normalize_windows, prepare_window_dataset,
       wavelet_hurst_estimate, wavelet_hurst_estimate_with_ci, wavelet_multifractal_metrics,
       optimize_wavelet_sparsity

# ----------------------------
# Engine loaders (lazy include)
# ----------------------------
const _learning_engine_loaded = Ref(false)
const _learning_engine_lock = ReentrantLock()

function _engine_root()
    return Path(@__DIR__) |> parent
end

function _ensure_learning_engine!()
    # Guard includes with a lock to avoid races under multi-threaded callers
    # (parallel window runs may trigger this concurrently)
    lock(_learning_engine_lock)
    try
        if _learning_engine_loaded[]
            return
        end
        engine_path = _engine_root()
        Base.include(@__MODULE__, engine_path * "/engine/mera_utils.jl")
        Base.include(@__MODULE__, engine_path * "/engine/learn_driver.jl")
        _learning_engine_loaded[] = true
    finally
        unlock(_learning_engine_lock)
    end
end

# ============================================================
# Utilitários gerais
# ============================================================

function zscore(x::AbstractVector)
    arr = (_cuda_available && try; CUDA.functional(); catch; false; end) ? cu(x) : x
    μ = mean(arr)
    σ = std(arr)
    σ < eps() && (σ = 1.0)
    return (arr .- μ) ./ σ, μ, σ
end

function overlap_windows(x::AbstractVector, winlen::Int; overlap::Float64=0.0)
    @assert 0.0 <= overlap < 1.0
    step = max(1, floor(Int, winlen * (1.0 - overlap)))
    N = length(x)
    windows = []
    
    for i in 1:step:(N - winlen + 1)
        push!(windows, x[i:(i + winlen - 1)])
    end
    
    return windows
end

# ============================================================
# MERA-Wavelet Core Functions (AD-Compatible, Funcional)
# ============================================================

# Funções auxiliares (wavelet MERA)

_eps_for(::Type{T}) where {T} = eps(typeof(real(T(1))))

function _normalize_filter(vec)
    norm_val = sqrt(sum(abs2, vec))
    eps_val = _eps_for(eltype(vec))
    norm_val = norm_val > eps_val ? norm_val : eps_val
    return vec ./ norm_val
end

function _quadrature_mirror(h::AbstractVector)
    L = length(h)
    phase(k) = isodd(k) ? one(eltype(h)) : -one(eltype(h))
    return [phase(k) * conj(h[L - k + 1]) for k in 1:L]
end

function _extract_filters(w::AbstractArray, v::AbstractArray)
    # Convention: h ≡ G (low-pass) and g ≡ H (high-pass) throughout the code.
    # We keep the existing names to avoid breaking callers; this matches the paper notation G/H.
    on_gpu = _cuda_available && (w isa CUDA.CuArray || v isa CUDA.CuArray)
    w_slice = Array(@view w[1:2, 1:2, 1, 1])

    h_raw = vec(w_slice[1, :])
    g_raw = vec(w_slice[2, :])

    eps_val = _eps_for(eltype(h_raw))
    if sqrt(sum(abs2, h_raw)) <= eps_val
        h_raw = fill(one(eltype(h_raw)), length(h_raw))
    end
    h = _normalize_filter(h_raw)

    g_proj = g_raw .- h * dot(h, g_raw)
    g_norm = sqrt(sum(abs2, g_proj))
    g = g_norm <= eps_val ? _quadrature_mirror(h) : g_proj ./ g_norm

    if on_gpu
        return cu(h), cu(g)
    else
        return h, g
    end
end

function _wavelet_analysis(signal::AbstractVector, h::AbstractVector, g::AbstractVector)
    N = length(signal)
    @assert iseven(N) "Comprimento do sinal deve ser par."

    half = div(N, 2)
    L = length(h)

    approx = map(1:half) do k
        idx = 2k - 1
        sum((conj(h[m]) * signal[mod(idx + m - 2, N) + 1] for m in 1:L); init=zero(eltype(signal)))
    end

    detail = map(1:half) do k
        idx = 2k - 1
        sum((conj(g[m]) * signal[mod(idx + m - 2, N) + 1] for m in 1:L); init=zero(eltype(signal)))
    end

    return approx, detail
end


function _wavelet_synthesis(approx::AbstractVector, detail::AbstractVector, h::AbstractVector, g::AbstractVector)
    @assert length(approx) == length(detail) "Approx e detail devem ter o mesmo comprimento."
    N = length(approx) * 2
    buf = Zygote.Buffer(Vector{eltype(approx)}(undef, N))
    for i in 1:N
        buf[i] = zero(eltype(approx))
    end
    L = length(h)

    for k in 1:length(approx)
        idx = 2k - 1
        for m in 0:(L - 1)
            pos = mod(idx + m - 1, N) + 1
            buf[pos] += h[m + 1] * approx[k] + g[m + 1] * detail[k]
        end
    end

    return copy(buf)
end

function _load_gpu_wavelet_kernels!()
    _cuda_available || return
    gpu_code = raw"""
function _wavelet_analysis(signal::CUDA.CuArray{T,1}, h::CUDA.CuArray{T,1}, g::CUDA.CuArray{T,1}) where {T}
    N = length(signal)
    @assert iseven(N) "Comprimento do sinal deve ser par."

    half = div(N, 2)
    L = length(h)
    if half == 0
        empty_vec = similar(signal, 0)
        return empty_vec, empty_vec
    end

    idxs = CUDA.CuArray(collect(0:(half - 1)))

    approx = CUDA.map(idxs) do k0
        idx0 = 2 * k0
        acc = zero(T)
        @inbounds for m in 0:(L - 1)
            pos = mod(idx0 + m, N)
            acc += conj(h[m + 1]) * signal[pos + 1]
        end
        acc
    end

    detail = CUDA.map(idxs) do k0
        idx0 = 2 * k0
        acc = zero(T)
        @inbounds for m in 0:(L - 1)
            pos = mod(idx0 + m, N)
            acc += conj(g[m + 1]) * signal[pos + 1]
        end
        acc
    end

    return approx, detail
end

function _wavelet_synthesis(
    approx::CUDA.CuArray{T,1},
    detail::CUDA.CuArray{T,1},
    h::CUDA.CuArray{T,1},
    g::CUDA.CuArray{T,1}
) where {T}
    @assert length(approx) == length(detail) "Approx e detail devem ter o mesmo comprimento."
    N = length(approx) * 2
    if N == 0
        return similar(approx, 0)
    end

    L = length(h)
    idxs_up = CUDA.CuArray(collect(0:(N - 1)))

    approx_up = CUDA.map(idxs_up) do pos0
        iseven(pos0) ? approx[div(pos0, 2) + 1] : zero(T)
    end

    detail_up = CUDA.map(idxs_up) do pos0
        iseven(pos0) ? detail[div(pos0, 2) + 1] : zero(T)
    end

    reconstructed = CUDA.map(idxs_up) do pos0
        acc = zero(T)
        @inbounds for m in 0:(L - 1)
            idx = mod(pos0 - m, N)
            acc += h[m + 1] * approx_up[idx + 1]
            acc += g[m + 1] * detail_up[idx + 1]
        end
        acc
    end

    return reconstructed
end
"""
    Base.include_string(@__MODULE__, gpu_code, "Wave6G_gpu_kernels.jl")
end

_load_gpu_wavelet_kernels!()

function _cpu_tangent_from_gpu(tangent, template_cpu)
    val = tangent
    while val isa ChainRulesCore.AbstractThunk
        val = ChainRulesCore.unthunk(val)
    end

    if val isa ChainRulesCore.NoTangent || val isa ChainRulesCore.ZeroTangent
        return zeros(eltype(template_cpu), size(template_cpu))
    end

    ChainRulesCore.iszero(val) && return zeros(eltype(template_cpu), size(template_cpu))

    return Array(val)
end

function ChainRulesCore.rrule(
    ::typeof(_wavelet_analysis),
    signal::CUDA.CuArray{T,1},
    h::CUDA.CuArray{T,1},
    g::CUDA.CuArray{T,1}
) where {T}
    signal_cpu = Array(signal)
    h_cpu = Array(h)
    g_cpu = Array(g)
    approx_cpu, detail_cpu = _wavelet_analysis(signal_cpu, h_cpu, g_cpu)
    approx = cu(approx_cpu)
    detail = cu(detail_cpu)

    function pullback(Δ)
        Δapprox_thunk, Δdetail_thunk = Δ
        Δapprox_cpu = _cpu_tangent_from_gpu(Δapprox_thunk, approx_cpu)
        Δdetail_cpu = _cpu_tangent_from_gpu(Δdetail_thunk, detail_cpu)

        function cpu_forward(sig, hh, gg)
            _wavelet_analysis(sig, hh, gg)
        end

        _, back = Zygote.pullback(cpu_forward, signal_cpu, h_cpu, g_cpu)
        Δsig_cpu, Δh_cpu, Δg_cpu = back((Δapprox_cpu, Δdetail_cpu))

        return ChainRulesCore.NoTangent(), cu(Δsig_cpu), cu(Δh_cpu), cu(Δg_cpu)
    end

    return (approx, detail), pullback
end

function ChainRulesCore.rrule(
    ::typeof(_wavelet_synthesis),
    approx::CUDA.CuArray{T,1},
    detail::CUDA.CuArray{T,1},
    h::CUDA.CuArray{T,1},
    g::CUDA.CuArray{T,1}
) where {T}
    approx_cpu = Array(approx)
    detail_cpu = Array(detail)
    h_cpu = Array(h)
    g_cpu = Array(g)
    reconstructed_cpu = _wavelet_synthesis(approx_cpu, detail_cpu, h_cpu, g_cpu)
    reconstructed = cu(reconstructed_cpu)

    function pullback(Δrecon_thunk)
        Δrecon_cpu = _cpu_tangent_from_gpu(Δrecon_thunk, reconstructed_cpu)

        function cpu_forward(a, d, hh, gg)
            _wavelet_synthesis(a, d, hh, gg)
        end

        _, back = Zygote.pullback(cpu_forward, approx_cpu, detail_cpu, h_cpu, g_cpu)
        Δapprox_cpu, Δdetail_cpu, Δh_cpu, Δg_cpu = back(Δrecon_cpu)

        return ChainRulesCore.NoTangent(), cu(Δapprox_cpu), cu(Δdetail_cpu), cu(Δh_cpu), cu(Δg_cpu)
    end

    return reconstructed, pullback
end

function ChainRulesCore.rrule(
    ::typeof(_extract_filters),
    w::CUDA.CuArray{Tw,Nw},
    v::CUDA.CuArray{Tv,Nv}
) where {Tw,Nw,Tv,Nv}
    w_cpu = Array(w)
    v_cpu = Array(v)
    h_cpu, g_cpu = _extract_filters(w_cpu, v_cpu)
    h = cu(h_cpu)
    g = cu(g_cpu)

    function pullback(Δ)
        Δh_thunk, Δg_thunk = Δ
        Δh_cpu = _cpu_tangent_from_gpu(Δh_thunk, h_cpu)
        Δg_cpu = _cpu_tangent_from_gpu(Δg_thunk, g_cpu)

        function cpu_forward(w_arg, v_arg)
            _extract_filters(w_arg, v_arg)
        end

        _, back = Zygote.pullback(cpu_forward, w_cpu, v_cpu)
        Δw_cpu, Δv_cpu = back((Δh_cpu, Δg_cpu))

        return ChainRulesCore.NoTangent(), cu(Δw_cpu), cu(Δv_cpu)
    end

    return (h, g), pullback
end


# Função recursiva auxiliar para garantir a compatibilidade com Zygote (imutabilidade)
function _mera_analyze_recursive(
    current_signal::AbstractVector, 
    wC::AbstractVector, 
    vC::AbstractVector, 
    p::Int, 
    L::Int
)
    # Condição de parada (Base case)
    if p > L
        return (current_signal,), () # Retorna o sinal de aproximação final e uma tupla vazia (imutável)
    end
    
    h, g = _extract_filters(wC[p], vC[p])
    approx_p, detail_p = _wavelet_analysis(current_signal, h, g)

    # 4. Chamada recursiva para o próximo nível
    approx_list_rest, detail_list_rest = _mera_analyze_recursive(approx_p, wC, vC, p + 1, L)

    # 5. Acumulação IMUTÁVEL dos coeficientes (usando tuplas concatenadas)
    all_approxs = (current_signal, approx_list_rest...)
    all_details = (detail_p, detail_list_rest...)
    
    return all_approxs, all_details
end


function mera_analyze(
    data::AbstractVector{<:Number}, 
    wC::AbstractVector, 
    vC::AbstractVector
)
    L = length(wC)
    use_gpu = _cuda_available && L > 0 && (wC[1] isa CUDA.CuArray || vC[1] isa CUDA.CuArray)
    initial_signal = use_gpu ? cu(complex(data)) : complex(data)
    
    # Chamada da função recursiva (AD-compatible)
    coeffs_approx_tup, coeffs_detail_tup = _mera_analyze_recursive(initial_signal, wC, vC, 1, L)

    # Converte as tuplas de arrays para vetores (fora do caminho crítico do AD)
    coeffs_approx = collect(coeffs_approx_tup)
    coeffs_detail = collect(coeffs_detail_tup)

    return coeffs_approx, coeffs_detail
end

function mera_synthesize(coeffs_approx, coeffs_detail, wC::AbstractVector, vC::AbstractVector)
    L = length(wC)
    @assert length(coeffs_detail) == L "Número de níveis inconsistentes entre detalhes e filtros."
    @assert length(coeffs_approx) >= L + 1 "Lista de aproximações deve conter original + L níveis."

    reconstructed = coeffs_approx[end]

    for level in L:-1:1
        detail_vec = coeffs_detail[level]
        approx_vec = reconstructed

        @assert length(detail_vec) == length(approx_vec) "Comprimentos incompatíveis para síntese no nível $level."

        h, g = _extract_filters(wC[level], vC[level])
        reconstructed = _wavelet_synthesis(approx_vec, detail_vec, h, g)
    end

    return reconstructed
end


# ============================================================
# Variational Optimization API (sparsity schedule)
# ============================================================

mutable struct VariationalState
    data::Vector{Float32}
    mean::Float64
    std::Float64
    L::Int
    chi::Int
    chimid::Int
    wC::Vector{ComplexStorage}
    vC::Vector{ComplexStorage}
    uC::Vector{ComplexStorage}
    history::Vector{Any}
    last_loss::Float64
end

_to_complex32(tensors) = [ComplexF32.(tensor) for tensor in tensors]

function _haar_tensors(L::Int)
    inv_sqrt2 = ComplexF32(1 / sqrt(2))
    wC = Vector{Array{ComplexF32}}(undef, L)
    vC = Vector{Array{ComplexF32}}(undef, L)
    uC = Vector{Array{ComplexF32}}(undef, L)

    for level in 1:L
        w = zeros(ComplexF32, 4, 2, 2, 2)
        v = zeros(ComplexF32, 2, 4, 2)
        u = zeros(ComplexF32, 2, 2, 2, 2)

        w[1, 1, 1, 1] = inv_sqrt2
        w[1, 2, 1, 1] = inv_sqrt2
        w[2, 1, 1, 1] = inv_sqrt2
        w[2, 2, 1, 1] = -inv_sqrt2

        v[1, 1, 1] = one(ComplexF32)
        v[2, 4, 2] = one(ComplexF32)

        for i in 1:2
            u[i, i, i, i] = one(ComplexF32)
        end

        wC[level] = w
        vC[level] = v
        uC[level] = u
    end

    return (wC=wC, vC=vC, uC=uC)
end

function _initial_tensors(L::Int, chi::Int, chimid::Int, init::Symbol)
    if init == :haar
        if chi == 2 && chimid == 2
            tensors = _haar_tensors(L)
            return (wC=tensors.wC, vC=tensors.vC, uC=tensors.uC)
        else
            @warn "Warm start Haar disponível apenas para chi=2 e chimid=2; utilizando inicialização aleatória." chi=chi chimid=chimid
        end
    end

    w_init, v_init, u_init = Base.invokelatest(init_random_tensors, L, chi, chimid)
    return (wC=_to_complex32(w_init), vC=_to_complex32(v_init), uC=_to_complex32(u_init))
end

function variational_default_schedule()
    return [
        (numiter=200, lr=0.005),
        (numiter=150, lr=0.0025)
    ]
end

function prepare_variational_state(
    data::AbstractVector{<:Number};
    L::Int=5,
    chi::Int=4,
    chimid::Int=4,
    normalize::Bool=true,
    seed::Union{Nothing,Integer}=nothing,
    init::Symbol=:random
)
    _ensure_learning_engine!()

    seed !== nothing && Random.seed!(seed)

    data_f64 = Float64.(data)
    if normalize
        data_norm, μ, σ = zscore(data_f64)
    else
        data_norm = copy(data_f64)
        μ = 0.0
        σ = 1.0
    end

    tensors = _initial_tensors(L, chi, chimid, init)

    state = VariationalState(
        Float32.(data_norm),
        μ,
        σ,
        L,
        chi,
        chimid,
        tensors.wC,
        tensors.vC,
        tensors.uC,
        Any[],
        NaN
    )

    return state
end

function optimize_variational_schedule!(
    state::VariationalState;
    schedule=variational_default_schedule(),
    base_opts::Dict{String,Any}=Dict{String,Any}(),
    data::Union{Nothing,AbstractVector{<:Number}}=nothing
)
    _ensure_learning_engine!()

    global _engine_mode[] = :ENGINE

    losses = Float64[]
    data_vec = data === nothing ? state.data : Float32.(data)
    if data !== nothing
        state.data = data_vec
    end

    for (idx, stage) in enumerate(schedule)
        stage_L = haskey(stage, :L) ? stage.L : state.L
        stage_chi = haskey(stage, :chi) ? stage.chi : state.chi
        stage_chimid = haskey(stage, :chimid) ? stage.chimid : state.chimid
        stage_numiter = haskey(stage, :numiter) ? stage.numiter : get(base_opts, "numiter", 200)
        stage_lr = haskey(stage, :lr) ? stage.lr : get(base_opts, "lr", 0.001)
        stage_seed = haskey(stage, :seed) ? stage.seed : nothing
        stage_reinit = haskey(stage, :reinit) ? stage.reinit : false
        stage_extra_opts = haskey(stage, :opts) ? stage.opts : nothing
        stage_init = haskey(stage, :init) ? stage.init : :previous

        opts = Dict{String,Any}(base_opts)
        if stage_extra_opts !== nothing
            for (k, v) in stage_extra_opts
                opts[k] = v
            end
        end
        opts["numiter"] = stage_numiter
        opts["lr"] = stage_lr

        dims_changed = stage_L != state.L || stage_chi != state.chi || stage_chimid != state.chimid
        need_reinit = stage_reinit || dims_changed || stage_init != :previous

        initial_tensors = nothing
        if need_reinit
            stage_seed !== nothing && Random.seed!(stage_seed)
            init_mode = stage_init == :previous ? :random : stage_init
            tensors = _initial_tensors(stage_L, stage_chi, stage_chimid, init_mode)
            initial_tensors = (wC=tensors.wC, vC=tensors.vC, uC=tensors.uC)
        else
            initial_tensors = (wC=state.wC, vC=state.vC, uC=state.uC)
        end

        @info "Estágio $(idx): L=$(stage_L), χ=$(stage_chi), χ_mid=$(stage_chimid), iterações=$(stage_numiter), lr=$(stage_lr)"

        w_stage, v_stage, u_stage, final_loss = Base.invokelatest(
            optimize_wavelet_sparsity,
            data_vec;
            L=stage_L,
            chi=stage_chi,
            chimid=stage_chimid,
            opts=opts,
            initial=initial_tensors
        )

        state.L = stage_L
        state.chi = stage_chi
        state.chimid = stage_chimid
        state.wC = w_stage
        state.vC = v_stage
        state.uC = u_stage
        state.last_loss = Float64(final_loss)

        push!(losses, Float64(final_loss))
        push!(state.history, (stage=idx, settings=(L=stage_L, chi=stage_chi, chimid=stage_chimid, numiter=stage_numiter, lr=stage_lr, init=stage_init), loss=Float64(final_loss)))
    end

    return (loss=state.last_loss, losses=losses, state=state)
end

# ============================================================
# Integração com Engine de Aprendizado
# ============================================================

function learn_wavelet_filters(
    data::AbstractVector{<:Number};
    L::Int=5,
    chi::Int=4,
    chimid::Int=4,
    opts::Dict=Dict()
)
    _ensure_learning_engine!()
    try
        global _engine_mode[] = :ENGINE

        @info "learn_wavelet_filters (ENGINE detectado) — Iniciando otimização MERA-Sparsity."

        # Correção WORLD AGE
        wC_final, vC_final, uC_final, final_loss = Base.invokelatest(
            optimize_wavelet_sparsity,
            data; 
            L=L, 
            chi=chi, 
            chimid=chimid, 
            opts=opts
        )
        
        return (w=wC_final, v=vC_final, u=uC_final, loss=final_loss)

    catch err
        # Captura e exibe o erro completo de tipagem/runtime
        if isa(err, DimensionMismatch)
            @warn "Erro de runtime na otimização (DimensionMismatch). Retornando STUB." error=err
        elseif isa(err, LoadError) || isa(err, MethodError) || isa(err, SystemError) || isa(err, Exception)
            @warn "Falha de AD ou inclusão/carregamento do Engine MERA. Engine desativado." error=err
        else
            @warn "Erro inesperado no Engine MERA. Retornando STUB." error=err
        end

        global _engine_mode[] = :STUB
        @info "learn_wavelet_filters (STUB mode) — Retornando tensores placeholder."
        
        wC_stub = [zeros(ComplexF64, 4, 2, 4, 4) for _ in 1:L]
        vC_stub = [zeros(ComplexF64, 4, 4, 2) for _ in 1:L]
        uC_stub = [zeros(ComplexF64, 4, 4, 4, 4) for _ in 1:L]

        return (w=wC_stub, v=vC_stub, u=uC_stub, loss=100.0)
    end
end

function run_learning_validation(; 
    L::Int=5,
    chi::Int=4,
    chimid::Int=4,
    numiter::Int=200,
    lr::Float64=0.005,
    N::Int=512,
    seed::Union{Nothing,Integer}=nothing,
    return_details::Bool=false
)
    @info "Iniciando Teste de Validação do AD (Otimização de Esparsidade)"

    seed !== nothing && Random.seed!(seed)

    data_raw = randn(N) + sin.(collect(1:N) ./ 10)
    data_norm, _, _ = zscore(data_raw)

    opts = Dict("numiter" => numiter, "lr" => lr)

    results = learn_wavelet_filters(
        data_norm;
        L=L,
        chi=chi,
        chimid=chimid,
        opts=opts
    )

    final_loss = results.loss
    success = _engine_mode[] == :ENGINE && final_loss < 99.0

    @printf "\nRESULTADOS DA OTIMIZAÇÃO:\n"
    @printf "------------------------------------------------\n"
    @printf "Camadas MERA (L): %d\n" L
    @printf "Iterações AD: %d\n" numiter
    @printf "Loss Final (Esparsidade): %.6f\n" final_loss
    @printf "------------------------------------------------\n"

    return return_details ? (success=success, loss=final_loss) : success
end

# ============================================================
# Função de otimização Wavelet Sparsity (migrada do engine)
# ============================================================

function sparsity_loss(coeffs_detail)
    loss = 0.0f0
    count = 0.0f0
    for coeffs in coeffs_detail
        loss += sum(abs, coeffs)
        count += Float32(length(coeffs))
    end
    return count > 0 ? loss / count : loss
end

# Fallback definitions if engine not loaded
function TensorUpdateSVD(T, cut_axis::Int)
    ET = eltype(T)
    was_gpu = _cuda_available && isa(T, CUDA.CuArray)
    T_host = was_gpu ? Array(T) : T
    nd = ndims(T_host)
    @assert 1 <= cut_axis <= nd "cut_axis fora do intervalo do tensor"

    # Reordena para matricizar o eixo a ser cortado como última dimensão
    perm = [i for i in 1:nd if i != cut_axis]
    push!(perm, cut_axis)
    inv_perm = invperm(perm)

    T_perm = permutedims(T_host, perm)
    perm_sizes = size(T_perm)
    dim_col = perm_sizes[end]
    dim_row = Int(prod(perm_sizes) ÷ dim_col)

    if dim_row == 0
        error("TensorUpdateSVD: received degenerate reshape with sizes $(perm_sizes)")
    end

    A = reshape(T_perm, dim_row, dim_col)

    # Usa decomposição polar a partir de eigendecomposição para evitar instabilidades do LAPACK
    work_T = ET <: Real ? Float64 : ComplexF64
    A_work = Array{work_T}(A)
    for idx in eachindex(A_work)
        if !isfinite(A_work[idx])
            A_work[idx] = zero(A_work[idx])
        end
    end
    gram = Hermitian(A_work' * A_work)
    evals, evecs = eigen(gram)

    max_eval = maximum(evals)
    eps_factor = eps(real(float(one(work_T))))
    threshold = max_eval == 0 ? zero(max_eval) : max_eval * eps_factor

    inv_sqrt_vals = similar(evals)
    for i in eachindex(evals)
        if evals[i] > threshold
            inv_sqrt_vals[i] = 1 / sqrt(evals[i])
        else
            inv_sqrt_vals[i] = zero(evals[i])
        end
    end

    U = A_work * evecs * Diagonal(inv_sqrt_vals)
    U = reshape(U, perm_sizes[1:end-1]..., dim_col)

    T_updated = permutedims(U, inv_perm)
    T_updated = was_gpu ? cu(T_updated) : T_updated

    return T_updated
end

function init_random_tensors(L::Int, chi::Int, chimid::Int)
    d = 2
    wC = Vector{Array{ComplexF64}}(undef, L)
    vC = Vector{Array{ComplexF64}}(undef, L)
    uC = Vector{Array{ComplexF64}}(undef, L)

    for p in 1:L
        vC[p] = randn(ComplexF64, chi, d^2, chimid)
        wC[p] = randn(ComplexF64, d^2, chimid, chi, chi)
        uC[p] = randn(ComplexF64, chimid, chimid, chimid, chimid)

        vC[p] = TensorUpdateSVD(vC[p], 1)
        wC[p] = TensorUpdateSVD(wC[p], 1)
        uC[p] = TensorUpdateSVD(uC[p], 2)
    end
    
    return wC, vC, uC
end

function optimize_wavelet_sparsity(
    data::AbstractVector{<:Number};
    L::Int,
    chi::Int,
    chimid::Int,
    opts::Dict,
    initial::Union{Nothing,NamedTuple}=nothing
)
    _ensure_learning_engine!()

    # Define init_random_tensors if not loaded
    # Removed, now defined at module level
    numiter = get(opts, "numiter", 200)
    lr      = get(opts, "lr", 0.001)

    if initial !== nothing
        @assert haskey(initial, :wC) && haskey(initial, :vC) "Initial tensors must include wC and vC"
        w_base = initial.wC
        v_base = initial.vC
        u_base = haskey(initial, :uC) ? initial.uC : nothing
    else
        w_init, v_init, u_init = init_random_tensors(L, chi, chimid)
        w_base = w_init
        v_base = v_init
        u_base = u_init
    end

    if u_base === nothing
        _, _, u_tmp = init_random_tensors(L, chi, chimid)
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

    opt = Flux.Adam(lr)
    params = (wC = wC, vC = vC)
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

    for k in 1:numiter
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
        opt_state, params = Flux.update!(opt_state, params, grads)
        wC = params.wC
        vC = params.vC
        for p in 1:L
            wC[p] = TensorUpdateSVD(wC[p], 1)
            vC[p] = TensorUpdateSVD(vC[p], 1)
        end
        params = (wC = wC, vC = vC)
        final_loss = Float32(loss)
        if k % 20 == 0 || k == 1
            @printf "Iteração: %d/%d, Loss: %.6f\n" k numiter final_loss
        end
    end
    return wC, vC, uC, final_loss
end

end # module Wave6G
