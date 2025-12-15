using Pkg
Pkg.activate(".")

using Wavelets
using DataFrames
using CSV

println("Testando componentes básicos...")

# Testar wavelet
println("1. Testando wavelet...")
w = wavelet(WT.haar)
test_signal = rand(1024)
coeffs = dwt(test_signal, w)
println("   Wavelet Haar: OK")

# Testar threshold
println("2. Testando threshold...")
function threshold_wavelet(coeffs::Vector{Float64}, retain_ratio::Float64)
    N = length(coeffs)
    detail_indices = Int[]
    for level in 1:Wavelets.maxtransformlevels(N)
        append!(detail_indices, collect(Wavelets.detailrange(N, level)))
    end
    total = length(detail_indices)
    keep = retain_ratio >= 1 ? total : max(1, round(Int, retain_ratio * total))
    if keep < total
        mags = abs.(coeffs[detail_indices])
        top_idx = partialsortperm(mags, 1:keep; rev=true)
        keep_mask = falses(total)
        keep_mask[top_idx] .= true
        for (j, idx) in enumerate(detail_indices)
            keep_mask[j] || (coeffs[idx] = 0.0)
        end
    end
    return coeffs, keep, total
end

thresh, kept, total = threshold_wavelet(copy(coeffs), 0.1)
println("   Threshold: OK (kept $kept/$total)")

# Testar reconstrução
println("3. Testando reconstrução...")
recon = idwt(thresh, w)
println("   Reconstrução: OK")

println("✅ Todos os componentes básicos funcionam!")
