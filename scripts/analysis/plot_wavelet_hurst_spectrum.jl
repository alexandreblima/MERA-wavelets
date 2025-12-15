using CSV
using DataFrames
using Statistics
using Plots
using Wavelets
using Wave6G
using Printf

"""
Plotar espectro wavelet para estimação do expoente de Hurst
Mostra como o método wavelet estima H através da análise de escalas
"""
function plot_wavelet_hurst_spectrum()
    # Carregar dados MAWI
    data_path = "data/MAWI_bytes_1ms.csv"
    if !isfile(data_path)
        @error "Arquivo MAWI não encontrado: $data_path"
        return
    end

    # Carregar série temporal
    data = CSV.read(data_path, DataFrame)
    series = vec(data[:, 1])  # Assumindo primeira coluna

    # Selecionar uma janela de exemplo (4096 amostras)
    window_size = 4096
    start_idx = 1
    window = Float64.(series[start_idx:start_idx+window_size-1])

    # Normalizar a janela
    window .-= mean(window)
    window ./= std(window)

    println("Janela selecionada: índices $start_idx:$(start_idx+window_size-1)")
    println("Tamanho: $(length(window)) amostras")

    # Calcular Hurst usando wavelet
    w_db4 = wavelet(WT.db4)
    H_estimated, scales, log_scales, log_energies = estimate_hurst_wavelet(window, w_db4)

    println("\nExpoente de Hurst estimado: H = $(round(H_estimated, digits=4))")

    # Plotar espectro wavelet
    p1 = plot(scales, exp.(log_energies),
              title="Espectro de Energia Wavelet",
              xlabel="Escala (j)", ylabel="Energia (S(j))",
              marker=:circle, linewidth=2, legend=false,
              yscale=:log10, xscale=:log2)

    # Plotar ajuste linear para H
    p2 = plot(log_scales, log_energies,
              title="Estimação do Hurst: log₂(S(j)) vs log₂(j)",
              xlabel="log₂(j) (Escala)", ylabel="log₂(S(j)) (Energia)",
              marker=:circle, linewidth=2, legend=false)

    # Adicionar linha de regressão
    slope = H_estimated + 0.5  # Para wavelet, slope = H + 1/2
    intercept = log_energies[1] - slope * log_scales[1]
    x_fit = range(minimum(log_scales), maximum(log_scales), length=100)
    y_fit = slope .* x_fit .+ intercept

    plot!(p2, x_fit, y_fit, linewidth=2, color=:red,
          label="Ajuste: slope = $(round(slope, digits=3))")

    # Adicionar anotação com valor de H
    annotate!(p2, maximum(log_scales)*0.7, maximum(log_energies)*0.9,
              text("H = $(round(H_estimated, digits=4))", :left, 12))

    # Plot combinado
    p_combined = plot(p1, p2, layout=(1,2), size=(1200, 500))
    savefig(p_combined, "results/wavelet_hurst_spectrum.png")

    @info "Gráfico salvo: results/wavelet_hurst_spectrum.png"
    @info "Hurst estimado: $H_estimated"

    return H_estimated, scales, log_scales, log_energies
end

"""
Função auxiliar para estimar Hurst via wavelet
Retorna H e dados intermediários para plotagem
"""
function estimate_hurst_wavelet(signal::Vector{Float64}, wavelet_type; min_level=4, max_level=12)
    # Transformada wavelet discreta
    wt = dwt(signal, wavelet_type)

    # Calcular energia em cada escala
    scales = collect(min_level:max_level)
    energies = Float64[]

    for j in scales
        # Extrair coeficientes da escala j
        coeffs = detailrange(length(signal), j)
        if !isempty(coeffs)
            energy = mean(abs.(wt[coeffs]).^2)
            push!(energies, energy)
        end
    end

    # Regressão linear em escala log-log
    log_scales = log2.(scales)
    log_energies = log2.(energies)

    # Ajuste linear: slope = H + 1/2 para wavelets
    slope = cov(log_scales, log_energies) / var(log_scales)
    H = slope - 0.5

    return H, scales, log_scales, log_energies
end

# Executar análise
if abspath(PROGRAM_FILE) == @__FILE__
    plot_wavelet_hurst_spectrum()
end