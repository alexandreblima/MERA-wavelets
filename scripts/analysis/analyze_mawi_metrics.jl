using CSV
using DataFrames
using Plots

# Carrega os dados
csv_path = "results/mawi_metrics_full.csv"
df = CSV.read(csv_path, DataFrame)

# Gráfico 1: PSNR por janela para cada método
plot(df.window_id, df.psnr_learned, label="PSNR Learned", lw=2)
plot!(df.window_id, df.psnr_haar, label="PSNR Haar", lw=2)
plot!(df.window_id, df.psnr_db4, label="PSNR DB4", lw=2)
xlabel!("Window ID")
ylabel!("PSNR")
title!("PSNR por janela - Métodos Learned, Haar, DB4")
savefig("results/psnr_comparison.png")

# Gráfico 2: Expoente de Hurst por método
plot(df.window_id, df.H_learned, label="H Learned", lw=2)
plot!(df.window_id, df.H_haar, label="H Haar", lw=2)
plot!(df.window_id, df.H_db4, label="H DB4", lw=2)
plot!(df.window_id, df.H_original, label="H Original", lw=2)
xlabel!("Window ID")
ylabel!("Hurst Exponent")
title!("Expoente de Hurst por janela")
savefig("results/hurst_comparison.png")

# Gráfico 3: Largura multifractal por método
plot(df.window_id, df.mf_width_learned, label="Width Learned", lw=2)
plot!(df.window_id, df.mf_width_haar, label="Width Haar", lw=2)
plot!(df.window_id, df.mf_width_db4, label="Width DB4", lw=2)
plot!(df.window_id, df.mf_width_original, label="Width Original", lw=2)
xlabel!("Window ID")
ylabel!("Multifractal Width")
title!("Largura multifractal por janela")
savefig("results/mfwidth_comparison.png")

# Gráfico 4: Delta H por método
plot(df.window_id, df.deltaH_learned, label="DeltaH Learned", lw=2)
plot!(df.window_id, df.deltaH_haar, label="DeltaH Haar", lw=2)
plot!(df.window_id, df.deltaH_db4, label="DeltaH DB4", lw=2)
xlabel!("Window ID")
ylabel!("Delta H")
title!("Delta H por janela")
savefig("results/deltaH_comparison.png")

# Resumo estatístico
println("Resumo estatístico das principais métricas:")
println(describe(df[:, ["psnr_learned", "psnr_haar", "psnr_db4", "H_learned", "H_haar", "H_db4", "mf_width_learned", "mf_width_haar", "mf_width_db4", "deltaH_learned", "deltaH_haar", "deltaH_db4"]]))
