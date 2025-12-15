using CSV
using DataFrames
using Statistics
using Plots
using Printf
using Distributions
using Distributions

# Carregar todos os arquivos CSV de experimentos
function load_experiment_data()
    retain_rates = [0.01]  # ← APENAS 0.01 por enquanto
    data_frames = Dict{Float64, DataFrame}()

    for rate in retain_rates
        file_path = "results/mawi_metrics_100_nonoverlap_retain_$(rate).csv"
        if isfile(file_path)
            df = CSV.read(file_path, DataFrame)
            data_frames[rate] = df
            @info "Carregado" rate=rate rows=nrow(df)
        else
            @warn "Arquivo não encontrado" file=file_path
        end
    end

    return data_frames
end

# Calcular estatísticas com intervalos de confiança
function compute_statistics(data_frames::Dict{Float64, DataFrame})
    retain_rates = sort(collect(keys(data_frames)))
    methods = ["haar", "db4", "learned"]

    results = Dict{String, Any}()
    for method in methods
        results["retain_ratio"] = retain_rates

        # Para cada taxa de retenção, calcular estatísticas
        deltaH_means = Float64[]
        deltaH_stds = Float64[]
        deltaH_cis = Float64[]  # Intervalos de confiança

        psnr_means = Float64[]
        psnr_stds = Float64[]
        psnr_cis = Float64[]

        mf_means = Float64[]
        mf_stds = Float64[]
        mf_cis = Float64[]

        for rate in retain_rates
            # Delta H
            deltaH_data = data_frames[rate][!, "deltaH_$(method)"]
            push!(deltaH_means, mean(deltaH_data))
            push!(deltaH_stds, std(deltaH_data))
            push!(deltaH_cis, confidence_interval(deltaH_data))

            # PSNR
            psnr_data = data_frames[rate][!, "psnr_$(method)"]
            push!(psnr_means, mean(psnr_data))
            push!(psnr_stds, std(psnr_data))
            push!(psnr_cis, confidence_interval(psnr_data))

            # MF Width
            mf_data = data_frames[rate][!, "mf_width_$(method)"]
            push!(mf_means, mean(mf_data))
            push!(mf_stds, std(mf_data))
            push!(mf_cis, confidence_interval(mf_data))
        end

        results["deltaH_$(method)_mean"] = deltaH_means
        results["deltaH_$(method)_std"] = deltaH_stds
        results["deltaH_$(method)_ci"] = deltaH_cis

        results["psnr_$(method)_mean"] = psnr_means
        results["psnr_$(method)_std"] = psnr_stds
        results["psnr_$(method)_ci"] = psnr_cis

        results["mf_width_$(method)_mean"] = mf_means
        results["mf_width_$(method)_std"] = mf_stds
        results["mf_width_$(method)_ci"] = mf_cis
    end

    return results
end

# Calcular intervalo de confiança (95%) usando t-distribution
function confidence_interval(data::Vector{Float64}, confidence::Float64=0.95)
    n = length(data)
    mean_val = mean(data)
    std_val = std(data)

    # Para n >= 30, usar normal; senão t-student
    if n >= 30
        z = quantile(Normal(), (1 + confidence) / 2)
        margin = z * std_val / sqrt(n)
    else
        t = quantile(TDist(n-1), (1 + confidence) / 2)
        margin = t * std_val / sqrt(n)
    end

    return margin
end

# Plotar impacto no expoente de Hurst com intervalos de confiança
function plot_hurst_preservation(results::Dict{String, Any})
    retain_ratios = results["retain_ratio"]
    methods = ["haar", "db4", "learned"]
    method_names = Dict("haar" => "Haar", "db4" => "Daubechies-4", "learned" => "MERA Aprendido")

    p = plot(title="Impacto da Compressão na Preservação do Expoente de Hurst\n(Intervalos de Confiança 95%)",
             xlabel="Taxa de Retenção", ylabel="Delta H (Diferença do Original)",
             legend=:topright, size=(900, 600))

    for method in methods
        means = results["deltaH_$(method)_mean"]
        cis = results["deltaH_$(method)_ci"]

        # Plotar pontos com barras de erro
        scatter!(p, retain_ratios, means, yerror=cis,
                label="$(method_names[method])", marker=:circle, markersize=6, linewidth=2)
    end

    hline!(p, [0.0], label="Sem Diferença", linestyle=:dash, color=:black, linewidth=1)

    savefig(p, "results/hurst_preservation_with_ci.png")
    @info "Gráfico salvo" file="results/hurst_preservation_with_ci.png"
end

# Plotar PSNR vs Taxa de Retenção com intervalos de confiança
function plot_psnr_comparison(results::Dict{String, Any})
    retain_ratios = results["retain_ratio"]
    methods = ["haar", "db4", "learned"]
    method_names = Dict("haar" => "Haar", "db4" => "Daubechies-4", "learned" => "MERA Aprendido")

    p = plot(title="PSNR vs Taxa de Compressão\n(Intervalos de Confiança 95%)",
             xlabel="Taxa de Retenção", ylabel="PSNR (dB)",
             legend=:bottomright, size=(900, 600))

    for method in methods
        means = results["psnr_$(method)_mean"]
        cis = results["psnr_$(method)_ci"]

        scatter!(p, retain_ratios, means, yerror=cis,
                label="$(method_names[method])", marker=:square, markersize=6, linewidth=2)
    end

    savefig(p, "results/psnr_comparison_with_ci.png")
    @info "Gráfico salvo" file="results/psnr_comparison_with_ci.png"
end

# Plotar Largura Multifractal com intervalos de confiança
function plot_mf_width(results::Dict{String, Any})
    retain_ratios = results["retain_ratio"]
    methods = ["haar", "db4", "learned"]
    method_names = Dict("haar" => "Haar", "db4" => "Daubechies-4", "learned" => "MERA Aprendido")

    p = plot(title="Largura Multifractal vs Taxa de Compressão\n(Intervalos de Confiança 95%)",
             xlabel="Taxa de Retenção", ylabel="Largura Multifractal",
             legend=:topright, size=(900, 600))

    # Plotar linha para original (constante) - sem IC pois é uma referência
    if haskey(results, "mf_width_original_mean") && !isempty(results["mf_width_original_mean"])
        original_width = results["mf_width_original_mean"][1]
        hline!(p, [original_width], label="Original", linestyle=:dash, color=:black, linewidth=1)
    end

    for method in methods
        means = results["mf_width_$(method)_mean"]
        cis = results["mf_width_$(method)_ci"]

        scatter!(p, retain_ratios, means, yerror=cis,
                label="$(method_names[method])", marker=:diamond, markersize=6, linewidth=2)
    end

    savefig(p, "results/mf_width_comparison_with_ci.png")
    @info "Gráfico salvo" file="results/mf_width_comparison_with_ci.png"
end

# Função principal
function main()
    data_frames = load_experiment_data()
    if isempty(data_frames)
        @error "Nenhum arquivo de dados encontrado. Execute os experimentos primeiro."
        return
    end

    results = compute_statistics(data_frames)

    # Imprimir resumo estatístico com intervalos de confiança
    println("\n=== RESUMO ESTATÍSTICO COM INTERVALOS DE CONFIANÇA (95%) ===")
    for method in ["haar", "db4", "learned"]
        println("\nMétodo: $(method)")
        println("Delta H médio: $(round(results["deltaH_$(method)_mean"][1], digits=4)) ± $(round(results["deltaH_$(method)_ci"][1], digits=4))")
        println("PSNR médio: $(round(results["psnr_$(method)_mean"][1], digits=2)) ± $(round(results["psnr_$(method)_ci"][1], digits=2)) dB")
        println("MF Width médio: $(round(results["mf_width_$(method)_mean"][1], digits=4)) ± $(round(results["mf_width_$(method)_ci"][1], digits=4))")
    end

    # Gerar gráficos com intervalos de confiança
    plot_hurst_preservation(results)
    plot_psnr_comparison(results)
    plot_mf_width(results)

    @info "Análise completa! Gráficos com intervalos de confiança salvos em results/"
end

main()