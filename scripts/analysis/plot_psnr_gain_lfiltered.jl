#!/usr/bin/env julia

# PSNR vs Compression Ratio (CR = 1/retain) and Gain (MERA - baseline),
# optionally filtering MERA rows by level L.

ENV["CUDA_VISIBLE_DEVICES"] = ""
ENV["JULIA_CUDA_USE_BINARYBUILDER"] = "false"

using Pkg
# Activate the project at the repo root (two levels up from scripts/analysis)
Pkg.activate(joinpath(@__DIR__, "../.."))

using ArgParse
using CSV
using DataFrames
using Statistics
using Printf
using Plots

function parse_cli()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--mera"; arg_type=String; required=true; help="CSV from MERA run (multi-retain)"
        "--baseline"; arg_type=String; required=true; help="CSV from baseline wavelets (haar, db4)"
        "--L"; arg_type=Int; default=5; help="Filter MERA rows to this level L (if column exists)"
        "--out-prefix"; arg_type=String; default="results/psnr_L"; help="Output prefix"
        "--title-suffix"; arg_type=String; default=""; help="Optional title suffix"
    end
    return parse_args(s)
end

function maybe_col(df::DataFrame, names_try::Vector{Symbol})
    # find a column by case/whitespace-insensitive match against any of names_try
    wanted = Set(lowercase.(String.(names_try)))
    for n in names(df)
        nm = lowercase(strip(String(n)))
        if nm in wanted
            return n
        end
    end
    return nothing
end

function load_mera_filtered(path::String, Lwanted::Int)
    df = CSV.read(path, DataFrame)
    # Drop summary rows if window_id exists
    c_wid = maybe_col(df, [:window_id, :Window_ID, :WindowId])
    if c_wid !== nothing
        df = df[coalesce.(df[!, c_wid], 0) .> 0, :]
    end
    # Filter by L if column exists (accept :L or :l)
    c_L = maybe_col(df, [:L, :l])
    if c_L !== nothing
        df = df[coalesce.(Int.(round.(Float64.(df[!, c_L]))), 0) .== Lwanted, :]
    end
    # Detect retain/psnr
    c_ret = maybe_col(df, [:retain, :Retain])
    c_psn = maybe_col(df, [:psnr, :PSNR])
    if c_ret === nothing || c_psn === nothing
        error("MERA CSV missing required columns (retain/psnr). Got: " * string(names(df)))
    end
    df = dropmissing(df, [c_ret, c_psn])
    g = groupby(df, c_ret)
    t = combine(g, c_psn => mean => :psnr_mean, c_psn => std => :psnr_std)
    # Ensure canonical column name :retain exists for downstream code
    if c_ret != :retain && (:retain ∉ names(t))
        rename!(t, c_ret => :retain)
    end
    sort!(t, :retain)
    return t
end

function load_baseline(path::String)
    df = CSV.read(path, DataFrame)
    c_wid = maybe_col(df, [:window_id, :Window_ID, :WindowId])
    if c_wid !== nothing
        df = df[coalesce.(df[!, c_wid], 0) .> 0, :]
    end
    c_wave = maybe_col(df, [:wavelet, :Wavelet])
    c_ret  = maybe_col(df, [:retain, :Retain])
    c_psn  = maybe_col(df, [:psnr, :PSNR])
    if c_wave === nothing || c_ret === nothing || c_psn === nothing
        error("Baseline CSV missing required columns (wavelet/retain/psnr). Got: " * string(names(df)))
    end
    df = dropmissing(df, [c_wave, c_ret, c_psn])
    g = groupby(df, [c_wave, c_ret])
    t = combine(g, c_psn => mean => :psnr_mean, c_psn => std => :psnr_std)
    # Canonicalize names for downstream
    if c_wave != :wavelet && (:wavelet ∉ names(t))
        rename!(t, c_wave => :wavelet)
    end
    if c_ret != :retain && (:retain ∉ names(t))
        rename!(t, c_ret => :retain)
    end
    sort!(t, [:wavelet, :retain])
    return t
end

function align_series(tm::DataFrame, tb::DataFrame)
    retains = collect(skipmissing(unique(tm.retain)))
    sort!(retains)
    cr = 1.0 ./ retains
    function pick_wave(t::DataFrame, wname::String)
        sub = t[t.wavelet .== wname, [:retain, :psnr_mean, :psnr_std]]
        mean = Float64[]; std = Float64[]
        for r in retains
            row = findfirst(x -> isapprox(x, r; atol=1e-12), sub.retain)
            if row === nothing
                push!(mean, NaN); push!(std, NaN)
            else
                push!(mean, sub.psnr_mean[row]); push!(std, sub.psnr_std[row])
            end
        end
        return (mean=mean, std=std)
    end
    mera_mean = [ tm.psnr_mean[findfirst(==(r), tm.retain)] for r in retains ]
    mera_std  = [ tm.psnr_std[ findfirst(==(r), tm.retain)] for r in retains ]
    haar = pick_wave(tb, "haar")
    db4  = pick_wave(tb, "db4")
    series = [
        (label="MERA (learned)", mean=mera_mean, std=mera_std),
        (label="Haar", mean=haar.mean, std=haar.std),
        (label="DB4", mean=db4.mean, std=db4.std),
    ]
    baseline_means = [
        (label="MERA - Haar", mean=haar.mean),
        (label="MERA - DB4", mean=db4.mean),
    ]
    return retains, cr, series, mera_mean, baseline_means
end

function plot_psnr_vs_cr(cr, series, outpath::String; title_suffix="")
    default(fontfamily="sans", size=(980, 560))
    plt = plot(legend=:bottomleft, xlabel="Compression ratio (CR = 1/retain)", ylabel="PSNR (dB)")
    if !isempty(title_suffix)
        plot!(title="PSNR vs CR " * title_suffix)
    else
        plot!(title="PSNR vs Compression Ratio")
    end
    for s in series
        plot!(plt, cr, s.mean, yerror=s.std, marker=:circle, label=s.label)
    end
    mkpath(dirname(outpath))
    savefig(plt, outpath)
end

function plot_gain_vs_cr(cr, mera_mean, baseline_means, outpath::String; title_suffix="")
    default(fontfamily="sans", size=(980, 560))
    plt = plot(legend=:bottomleft, xlabel="Compression ratio (CR = 1/retain)", ylabel="ΔPSNR (dB)")
    if !isempty(title_suffix)
        plot!(title="Gain (MERA - baseline) vs CR " * title_suffix)
    else
        plot!(title="Gain (MERA - baseline) vs Compression Ratio")
    end
    for b in baseline_means
        gain = mera_mean .- b.mean
        plot!(plt, cr, gain, marker=:square, label=b.label)
    end
    mkpath(dirname(outpath))
    savefig(plt, outpath)
end

function run()
    args = parse_cli()
    @assert isfile(args["mera"]) "MERA CSV not found: $(args["mera"])"
    @assert isfile(args["baseline"]) "Baseline CSV not found: $(args["baseline"])"

    Lwanted = Int(args["L"])  # filtering L per CLI
    tm = load_mera_filtered(args["mera"], Lwanted)
    tb = load_baseline(args["baseline"])

    retains, cr, series, mera_mean, baseline_means = align_series(tm, tb)

    suffix = @sprintf "(L=%d)" Lwanted
    if !isempty(args["title-suffix"]) 
        suffix *= " " * String(args["title-suffix"]) 
    end

    # If out-prefix not provided/left default, place outputs next to MERA CSV and
    # infer L from the MERA CSV directory name L_<L> when available.
    out_prefix = String(args["out-prefix"]) 
    infer_L_from_path = false
    if out_prefix == "results/psnr_L" || isempty(out_prefix)
        mera_dir = dirname(String(args["mera"]))
        out_prefix = joinpath(mera_dir, "psnr_L")
        infer_L_from_path = true
    end
    # Use path-inferred L only for filename suffix (do not change filtering L)
    Lname = Lwanted
    if infer_L_from_path
        for part in splitpath(dirname(String(args["mera"])) )
            m = match(r"^L_(\d+)$", part)
            if m !== nothing
                Lname = parse(Int, m.captures[1])
                break
            end
        end
    end
    base = string(out_prefix) * @sprintf "_L%d" Lname
    out1 = base * "_psnr_vs_cr.png"
    out2 = base * "_gain_vs_cr.png"

    plot_psnr_vs_cr(cr, series, out1; title_suffix=suffix)
    plot_gain_vs_cr(cr, mera_mean, baseline_means, out2; title_suffix=suffix)

    # Export also PDF/SVG
    for ext in (".pdf", ".svg")
        try
            plot_psnr_vs_cr(cr, series, replace(out1, ".png"=>ext); title_suffix=suffix)
            plot_gain_vs_cr(cr, mera_mean, baseline_means, replace(out2, ".png"=>ext); title_suffix=suffix)
        catch e
            @warn "Failed to export $ext" error=e
        end
    end

    println("Saved: ", out1)
    println("Saved: ", out2)
    println("Saved: ", replace(out1, ".png"=>".pdf"))
    println("Saved: ", replace(out1, ".png"=>".svg"))
    println("Saved: ", replace(out2, ".png"=>".pdf"))
    println("Saved: ", replace(out2, ".png"=>".svg"))
end

run()
