#!/usr/bin/env julia

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CSV
using DataFrames
using Statistics

mera_path = get(ENV, "MERA_CSV", "results/mawi_parallel_1_100_metrics_t8.csv")
base_path = get(ENV, "BASE_CSV", "results/wavelet_baseline_psnr.csv")
out_path = get(ENV, "OUT_CSV", "results/psnr_wide.csv")

# Note: This script currently joins only Haar and DB4 baselines for compatibility
# with older plotting scripts. For extended baselines (e.g., C3/S8/Bior4.4),
# prefer scripts/analysis/plot_psnr_comparison.jl with --waves.

m = CSV.read(mera_path, DataFrame)
mb = m[coalesce.(m.window_id, 0) .> 0, [:window_id, :retain, :psnr]]
rename!(mb, :psnr => :psnr_mera)

b = CSV.read(base_path, DataFrame)
b = b[coalesce.(b.window_id, 0) .> 0, [:window_id, :wavelet, :retain, :psnr]]
haar = b[b.wavelet .== "haar", [:window_id, :retain, :psnr]]
rename!(haar, :psnr => :psnr_haar)
db4 = b[b.wavelet .== "db4", [:window_id, :retain, :psnr]]
rename!(db4, :psnr => :psnr_db4)

wide = leftjoin(mb, haar, on=[:window_id, :retain])
wide = leftjoin(wide, db4, on=[:window_id, :retain])

CSV.write(out_path, wide)
println("Wrote: ", out_path)
