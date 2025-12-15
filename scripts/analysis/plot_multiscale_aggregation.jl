#!/usr/bin/env julia

ENV["CUDA_VISIBLE_DEVICES"] = ""
ENV["JULIA_CUDA_USE_BINARYBUILDER"] = "false"

using Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))

using ArgParse
using CSV
using DataFrames
using Statistics
using Plots
using Printf

function parse_cli()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--data"; arg_type=String; required=true; help="CSV with raw time series (first numeric column will be used)"
        "--fs"; arg_type=Float64; default=1000.0; help="Sampling rate in Hz (default assumes 1 ms sampling)"
        "--scales-ms"; arg_type=String; default="1,2,4,8,16"; help="Comma-separated aggregation scales in milliseconds"
        "--agg"; arg_type=String; default="mean"; help="Aggregation: mean|sum"
        "--max-points"; arg_type=Int; default=5000; help="Downsample each aggregated series to at most this many points for plotting"
        "--no-original"; action=:store_true; help="Do not plot the original series as the top panel"
        "--normalize-panels"; action=:store_true; help="Z-score each panel for visual comparability (plotting only)"
        "--title"; arg_type=String; default=""; help="Optional title suffix"
        "--out"; arg_type=String; default=""; help="Output path (default: results/<trace_base>/multiscale_aggregation.(png|pdf))"
    end
    return parse_args(s)
end

function load_first_numeric_column(path::String)
    df = CSV.read(path, DataFrame)
    for c in eachcol(df)
        v = try
            Float64.(collect(skipmissing(c)))
        catch
            continue
        end
        if !isempty(v)
            return v
        end
    end
    error("No numeric column found in: " * path)
end

function aggregate_series(x::Vector{Float64}, factor::Int; agg::String="mean")
    n = length(x)
    m = fld(n, factor)
    if m == 0
        return x
    end
    y = reshape(@view(x[1:(m*factor)]), factor, m)
    if agg == "mean"
        return vec(mean(y; dims=1))
    elseif agg == "sum"
        return vec(sum(y; dims=1))
    else
        error("Unknown aggregation: " * agg)
    end
end

function downsample(x::Vector{Float64}, max_points::Int)
    n = length(x)
    if n <= max_points
        return x
    end
    idxs = round.(Int, range(1, n; length=max_points))
    return x[idxs]
end

function main()
    args = parse_cli()
    data_path = String(args["data"])
    @assert isfile(data_path) "Data file not found: $(data_path)"
    x = load_first_numeric_column(data_path)
    x = Float64.(x)
    x = x[isfinite.(x)]

    fs = Float64(args["fs"]) # samples per second
    # parse scales in ms and convert to integer aggregation factors
    scales_ms = parse.(Int, split(String(args["scales-ms"]), ","))
    factors = Int.(round.(fs .* (scales_ms ./ 1000)))
    factors = unique(filter(>=(1), factors))

    agg = lowercase(String(args["agg"]))
    maxpts = Int(args["max-points"])

    trace_file = splitpath(data_path) |> last
    trace_base = splitext(trace_file)[1]
    out = String(get(args, "out", ""))
    if isempty(out)
        out = joinpath("results", trace_base, "multiscale_aggregation.png")
    end
    mkpath(dirname(out))

    ttl_suffix = String(get(args, "title", ""))
    ttl = isempty(ttl_suffix) ? "Multi-scale aggregation" : "Multi-scale aggregation — " * ttl_suffix

    include_orig = !(haskey(args, "no-original") && args["no-original"] == true)
    normalize = haskey(args, "normalize-panels") && args["normalize-panels"] == true

    n_panels = include_orig ? length(factors) + 1 : length(factors)
    plt = plot(layout=(n_panels, 1), size=(1000, 250*n_panels), grid=true)

    # helper: optional z-score for plotting only
    zplot(v) = begin
        if !normalize
            return v
        end
        μ = mean(v); σ = std(v); σ = σ < eps(Float64) ? 1.0 : σ
        return (v .- μ) ./ σ
    end

    # Original (downsampled for display)
    if include_orig
        x0 = downsample(x, maxpts)
        plot!(plt[1], zplot(x0), label=@sprintf("original (%.0f Hz)", fs), title=ttl, legend=:topright)
    end

    # Aggregated scales
    for (i, f) in enumerate(factors)
        xa = aggregate_series(x, f; agg=agg)
        xa = downsample(xa, maxpts)
        Δt_ms = 1000 * f / fs
        panel_idx = include_orig ? (i + 1) : i
        plot!(plt[panel_idx], zplot(xa), label=@sprintf("%.0f ms (factor %d)", Δt_ms, f), legend=:topright)
    end

    png_path = endswith(lowercase(out), ".png") ? out : replace(out, r"\.(pdf|svg)$" => ".png")
    savefig(plt, png_path)
    # optional PDF/SVG next to it
    pdf_path = replace(png_path, ".png" => ".pdf")
    svg_path = replace(png_path, ".png" => ".svg")
    try
        savefig(plt, pdf_path)
    catch
    end
    try
        savefig(plt, svg_path)
    catch
    end
    @info "Saved multiscale aggregation plot" png=png_path pdf=pdf_path svg=svg_path factors=factors
end

main()
