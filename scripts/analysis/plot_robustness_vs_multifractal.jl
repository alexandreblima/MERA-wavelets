#!/usr/bin/env julia

ENV["CUDA_VISIBLE_DEVICES"] = ""
ENV["JULIA_CUDA_USE_BINARYBUILDER"] = "false"

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using ArgParse
using CSV
using DataFrames
using Statistics
using Plots
using Printf

function parse_cli()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--mera"; arg_type=String; required=true; help="CSV MERA com métricas (hurst, alpha_width)"
        "--wide"; arg_type=String; required=true; help="CSV wide com psnr_mera/psnr_haar/psnr_db4 (por janela e retain)"
        "--retain"; arg_type=Float64; default=0.1
        "--outdir"; arg_type=String; default="results"
    end
    return parse_args(s)
end

function pearsonr(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    xf = collect(skipmissing(x)); yf = collect(skipmissing(y))
    n = min(length(xf), length(yf))
    n < 3 && return NaN
    xf = xf[1:n]; yf = yf[1:n]
    cx = xf .- mean(xf); cy = yf .- mean(yf)
    den = std(xf) * std(yf)
    den ≈ 0 && return NaN
    return sum(cx .* cy) / ((n - 1) * den)
end

function fit_line(x::Vector{<:Real}, y::Vector{<:Real})
    n = length(x)
    n < 2 && return (0.0, mean(y))
    X = [ones(n) x]
    # OLS beta = (X'X)^(-1) X'y
    β = (X'X) \ (X'y)
    return (β[2], β[1])  # slope, intercept
end

function run()
    args = parse_cli()
    @assert isfile(args["mera"]) "MERA CSV not found"
    @assert isfile(args["wide"]) "Wide CSV not found"

    mera = CSV.read(args["mera"], DataFrame)
    mera = mera[coalesce.(mera.window_id, 0) .> 0, :]
    mera = dropmissing(mera, [:retain, :psnr, :hurst_H])

    wide = CSV.read(args["wide"], DataFrame)
    wide = dropmissing(wide, [:retain])

    rsel = args["retain"]
    mm = mera[abs.(mera.retain .- rsel) .< 1e-9, [:window_id, :retain, :hurst_H, :alpha_width]]
    ww = wide[abs.(wide.retain .- rsel) .< 1e-9, :]

    df = innerjoin(mm, ww; on=[:window_id, :retain])

    df[:, :gain_haar] = df.psnr_mera .- df.psnr_haar
    df[:, :gain_db4]  = df.psnr_mera .- df.psnr_db4

    mkpath(args["outdir"])

    function scatter_with_fit(x, y, xlabel, ylabel, title, outpath)
        xclean = collect(skipmissing(x)); yclean = collect(skipmissing(y))
        n = min(length(xclean), length(yclean))
        xclean = xclean[1:n]; yclean = yclean[1:n]
        r = pearsonr(xclean, yclean)
        m, b = fit_line(xclean, yclean)
        default(fontfamily="sans", size=(900, 520))
        plt = scatter(xclean, yclean; xlabel=xlabel, ylabel=ylabel, title=title, legend=false)
        xs = range(minimum(xclean), maximum(xclean), length=100)
        plot!(plt, xs, m .* xs .+ b, color=:red, label=nothing)
        ann = @sprintf("r = %.3f", r)
        annotate!(plt, minimum(xclean), maximum(yclean), text(ann, :left, 10))
        savefig(plt, outpath)
        # Also export PDF/SVG
        base = splitext(outpath)[1]
        try
            savefig(plt, base * ".pdf")
            savefig(plt, base * ".svg")
        catch e
            @warn "Failed to export PDF/SVG" error=e
        end
    end

    out1 = joinpath(args["outdir"], Printf.@sprintf("gain_haar_vs_hurst_retain_%.2f.png", rsel))
    out2 = joinpath(args["outdir"], Printf.@sprintf("gain_db4_vs_hurst_retain_%.2f.png", rsel))
    out3 = joinpath(args["outdir"], Printf.@sprintf("gain_haar_vs_alpha_width_retain_%.2f.png", rsel))
    out4 = joinpath(args["outdir"], Printf.@sprintf("gain_db4_vs_alpha_width_retain_%.2f.png", rsel))

    scatter_with_fit(df.hurst_H, df.gain_haar, "Hurst H", "ΔPSNR (MERA−Haar)", Printf.@sprintf("Retain=%.2f", rsel), out1)
    scatter_with_fit(df.hurst_H, df.gain_db4,  "Hurst H", "ΔPSNR (MERA−DB4)",  Printf.@sprintf("Retain=%.2f", rsel), out2)

    if :alpha_width in names(df)
    scatter_with_fit(df.alpha_width, df.gain_haar, "alpha_width", "ΔPSNR (MERA−Haar)", Printf.@sprintf("Retain=%.2f", rsel), out3)
    scatter_with_fit(df.alpha_width, df.gain_db4,  "alpha_width", "ΔPSNR (MERA−DB4)",  Printf.@sprintf("Retain=%.2f", rsel), out4)
    end

    println("Saved:", out1)
    println("Saved:", out2)
    println("Saved:", out3)
    println("Saved:", out4)
end

run()
