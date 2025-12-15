# ======================================================================
# test/test_reconstruction.jl
# ----------------------------------------------------------------------
# Validação determinística da reconstrução (MERA Haar)
# ======================================================================

using LinearAlgebra
using Printf

const INV_SQRT2 = 1 / sqrt(2)

function build_haar_tensors(L::Int)
    chi = 2
    chimid = 2
    wC = [zeros(ComplexF64, 4, chimid, chi, chi) for _ in 1:L]
    vC = [zeros(ComplexF64, chi, 4, chimid) for _ in 1:L]

    for level in 1:L
        w = wC[level]
        w[1, 1, 1, 1] = INV_SQRT2
        w[1, 2, 1, 1] = INV_SQRT2
        w[2, 1, 1, 1] = INV_SQRT2
        w[2, 2, 1, 1] = -INV_SQRT2

        v = vC[level]
        v[1, 1, 1] = 1.0
        v[2, 4, 2] = 1.0
    end

    return wC, vC
end

function reconstruction_metrics(original::AbstractVector, reconstructed::AbstractVector, coeffs_approx, coeffs_detail)
    rel_error = norm(reconstructed - original) / max(norm(original), eps())
    energy_original = sum(abs2, original)
    energy_coeffs = sum(abs2, coeffs_approx[end]) + sum(sum(abs2, d) for d in coeffs_detail)
    parseval_error = abs(energy_original - energy_coeffs) / max(energy_original, eps())
    mse = sum(abs2, original .- reconstructed) / length(original)
    max_val = maximum(abs.(original))
    psnr = mse < eps() ? Inf : 10.0 * log10(max_val^2 / mse)
    return rel_error, parseval_error, psnr
end

function run_reconstruction_test(L::Int, N_data::Int, chi::Int, chimid::Int)
    @info "Iniciando Teste de Reconstrução Determinística (Haar MERA)" L=L N=N_data chi=chi chimid=chimid

    expected_length = 4 * (1 << L)
    N = expected_length

    if N_data != expected_length
        @warn "Comprimento de sinal ajustado para compatibilidade com MERA binária." solicitado=N_data usado=expected_length
    end

    if (chi, chimid) != (2, 2)
        @warn "Parâmetros chi/chimid informados são ignorados no teste determinístico." chi=chi chimid=chimid
    end

    wC, vC = build_haar_tensors(L)
    data = ComplexF64.(collect(1:N))

    coeffs_approx, coeffs_detail = Wave6G.mera_analyze(data, wC, vC)
    reconstructed = Wave6G.mera_synthesize(coeffs_approx, coeffs_detail, wC, vC)

    rel_error, parseval_error, psnr_val = reconstruction_metrics(data, reconstructed, coeffs_approx, coeffs_detail)

    @printf "\nRESULTADOS DA VALIDAÇÃO:\n"
    @printf "------------------------------------------------\n"
    @printf "Comprimento Original: %d\n" length(data)
    @printf "Comprimento Reconstruído: %d\n" length(reconstructed)
    @printf "Erro Relativo (L2): %.2e\n" rel_error
    @printf "Erro Relativo de Parseval: %.2e\n" parseval_error
    @printf "PSNR (dB): %s\n" (psnr_val == Inf ? "Inf" : @sprintf("%.4f", psnr_val))
    @printf "------------------------------------------------\n"

    return rel_error < 1e-12 && parseval_error < 1e-12
end

L_test = 4
N_test = 4 * (1 << L_test)
chi_test = 2
chimid_test = 2

# success = run_reconstruction_test(L_test, N_test, chi_test, chimid_test)
# @info "Teste determinístico de reconstrução concluído." success=success
