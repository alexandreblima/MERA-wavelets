#!/usr/bin/env julia

ENV["CUDA_VISIBLE_DEVICES"] = ""
ENV["JULIA_CUDA_USE_BINARYBUILDER"] = "false"

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using ArgParse
using CSV
using DataFrames
using Statistics

function parse_cli()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--mera"; arg_type=String; required=true
        "--wide"; arg_type=String; required=true
        "--retains"; arg_type=String; default="0.01,0.02,0.05,0.1"
        "--output"; arg_type=String; default="results/robustness_correlations.csv"
    end
    return parse_args(s)
end

function pearsonr(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    x = collect(skipmissing(x)); y = collect(skipmissing(y))
    n = min(length(x), length(y))
    n < 3 && return (NaN, 0)
    x = x[1:n]; y = y[1:n]
    cx = x .- mean(x); cy = y .- mean(y)
    den = std(x) * std(y)
    if den ≈ 0
        return (NaN, n)
    end
    return (sum(cx .* cy) / ((n - 1) * den), n)
end

function fit_line(x::Vector{<:Real}, y::Vector{<:Real})
    x = collect(skipmissing(x)); y = collect(skipmissing(y))
    n = min(length(x), length(y))
    n < 2 && return (NaN, NaN)
    x = x[1:n]; y = y[1:n]
    X = [ones(n) x]
    β = (X'X) \ (X'y)
    return (β[2], β[1])  # slope, intercept
end

function run()
    args = parse_cli()
    mera = CSV.read(args["mera"], DataFrame)
    mera = mera[coalesce.(mera.window_id, 0) .> 0, :]
    wide = CSV.read(args["wide"], DataFrame)
    # Round retains to 2 decimals to avoid FP mismatches
    if :retain in names(mera)
        mera[:, :retain] = round.(Float64.(mera.retain), digits=2)
    end
    if :retain in names(wide)
        wide[:, :retain] = round.(Float64.(wide.retain), digits=2)
    end

    retains = [parse(Float64, strip(x)) for x in split(args["retains"], ",") if !isempty(strip(x))]

    rows = DataFrame(baseline=String[], metric=String[], retain=Float64[], r=Float64[], slope=Float64[], intercept=Float64[], n=Int[])

    for rsel in retains
        rsel2 = round(rsel, digits=2)
        mm = mera[mera.retain .== rsel2, [:window_id, :retain, :hurst_H, :alpha_width]]
        ww = wide[wide.retain .== rsel2, :]
        df = innerjoin(mm, ww; on=[:window_id, :retain])
        if isempty(df)
            continue
        end
        df[:, :gain_haar] = df.psnr_mera .- df.psnr_haar
        df[:, :gain_db4]  = df.psnr_mera .- df.psnr_db4
        # Drop rows missing any of the required fields
        dropmissing!(df, [:gain_haar, :gain_db4])

        for (base, gaincol) in (("haar", :gain_haar), ("db4", :gain_db4))
            for metric in ("hurst_H", "alpha_width")
                if Symbol(metric) ∈ names(df)
                    r, n = pearsonr(df[:, Symbol(metric)], df[:, gaincol])
                    slope, intercept = fit_line(df[:, Symbol(metric)], df[:, gaincol])
                    push!(rows, (base, metric, rsel2, r, slope, intercept, n))
                end
            end
        end
    end

    mkpath(dirname(args["output"]))
    CSV.write(args["output"], rows)
    println("Wrote: ", args["output"]) 
end

run()
