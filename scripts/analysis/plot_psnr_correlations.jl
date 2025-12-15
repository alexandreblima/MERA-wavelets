#!/usr/bin/env julia

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CSV
using DataFrames
using Statistics
using Plots

mera_path = get(ENV, "MERA_CSV", "results/mawi_parallel_1_100_metrics_t8.csv")
out1 = get(ENV, "OUT1", "results/corr_psnr_hurst.png")
out2 = get(ENV, "OUT2", "results/corr_psnr_alpha_width.png")

# Use only a specific retain for correlation (e.g., 0.1)
retain_focus = parse(Float64, get(ENV, "RETAIN", "0.1"))

df = CSV.read(mera_path, DataFrame)
df = df[coalesce.(df.window_id, 0) .> 0, :]
df = dropmissing(df, [:retain, :psnr, :hurst_H])

sub = df[abs.(df.retain .- retain_focus) .< 1e-9, :]

scatter(sub.hurst_H, sub.psnr;
    xlabel="Hurst H",
    ylabel="PSNR (dB)",
    title="PSNR vs Hurst (retain=$(retain_focus))",
    legend=false)
savefig(out1)

if :alpha_width in names(sub)
    scatter(sub.alpha_width, sub.psnr;
        xlabel="alpha_width",
        ylabel="PSNR (dB)",
        title="PSNR vs alpha_width (retain=$(retain_focus))",
        legend=false)
    savefig(out2)
end

println("Saved:", out1)
println("Saved:", out2)
