# ============================================================
# engine/mera_utils.jl (FINAL: GARANTIA DE TIPAGEM)
# ============================================================

using LinearAlgebra
using Random

# Importação condicional de CUDA
let
    try
        @eval using CUDA
        @eval const _cuda_available = CUDA.has_cuda()
    catch e
        @warn "CUDA não disponível em mera_utils.jl, usando apenas CPU" error=e
        @eval const _cuda_available = false
        @eval module CUDA
            has_cuda() = false
            CuArray = Array
            set_runtime_version!(v) = nothing
            cu(x) = x  # Função identidade quando CUDA não disponível
        end
    end
end

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
        val = max(real(evals[i]), zero(evals[i]))
        inv_sqrt_vals[i] = val > threshold ? inv(sqrt(val)) : zero(val)
    end

    inv_sqrt = evecs * Diagonal(inv_sqrt_vals) * adjoint(evecs)
    Q_work = A_work * inv_sqrt
    A_iso = Array{ET}(Q_work)
    T_iso = reshape(A_iso, perm_sizes)

    result = permutedims(T_iso, inv_perm)
    return was_gpu ? cu(result) : result
end

function init_random_tensors(L::Int, chi::Int, chimid::Int)
    d = 2
    wC = Vector{Array{ComplexF64}}(undef, L)
    # Adaptação para GPU: arrays convertidos para CuArray se disponível
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
