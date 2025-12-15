# ======================================================================
# test/test_wavelet_aliases.jl
# ----------------------------------------------------------------------
# Verifica que aliases string/símbolo resolvem para as wavelets esperadas
# (incluindo Coiflet-3 e Symmlet-8) dentro das métricas do pipeline.
# ======================================================================

using Test
using Random
using Wavelets

function run_wavelet_alias_tests()
    Random.seed!(42)
    data = randn(4096)

    # Coiflet-3 (via coif6 filter in Wavelets.jl)
    w_coif = wavelet(WT.coif6)
    direct_c3 = Wave6G.wavelet_hurst_estimate(data; wavelet=w_coif)
    alias_c3 = Wave6G.wavelet_hurst_estimate(data; wavelet="c3")
    symbol_c3 = Wave6G.wavelet_hurst_estimate(data; wavelet=:C3)
    @test alias_c3.H ≈ direct_c3.H atol=1e-10
    @test symbol_c3.H ≈ direct_c3.H atol=1e-10

    direct_multi_c3 = Wave6G.wavelet_multifractal_metrics(data; wavelet=w_coif)
    alias_multi_c3 = Wave6G.wavelet_multifractal_metrics(data; wavelet="coiflet3")
    @test alias_multi_c3.alpha_width ≈ direct_multi_c3.alpha_width atol=1e-8
    @test alias_multi_c3.f_alpha_peak ≈ direct_multi_c3.f_alpha_peak atol=1e-8

    # Symmlet-8
    w_sym8 = wavelet(WT.sym8)
    direct_s8 = Wave6G.wavelet_hurst_estimate(data; wavelet=w_sym8)
    alias_s8 = Wave6G.wavelet_hurst_estimate(data; wavelet="s8")
    symbol_s8 = Wave6G.wavelet_hurst_estimate(data; wavelet=:symmlet8)
    @test alias_s8.H ≈ direct_s8.H atol=1e-10
    @test symbol_s8.H ≈ direct_s8.H atol=1e-10

    direct_multi_s8 = Wave6G.wavelet_multifractal_metrics(data; wavelet=w_sym8)
    alias_multi_s8 = Wave6G.wavelet_multifractal_metrics(data; wavelet="symlet8")
    @test alias_multi_s8.alpha_width ≈ direct_multi_s8.alpha_width atol=1e-8
    @test alias_multi_s8.f_alpha_peak ≈ direct_multi_s8.f_alpha_peak atol=1e-8

    return true
end
