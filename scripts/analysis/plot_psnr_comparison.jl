#!/usr/bin/env julia

ENV["CUDA_VISIBLE_DEVICES"] = ""
ENV["JULIA_CUDA_USE_BINARYBUILDER"] = "false"

using Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))

using ArgParse
using CSV
using DataFrames
using Statistics
using Random
using Plots

function parse_cli()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--mera"; arg_type=String; required=true; help="CSV from MERA run (multi-retain)"
        "--baseline"; arg_type=String; required=true; help="CSV from baseline wavelets (supports: haar, db4, c3, s8, bior4.4)"
        "--waves"; arg_type=String; default="haar,db4,c3,s8,bior4.4"; help="comma-separated baseline wavelets to include"
        "--out1"; arg_type=String; default="results/psnr_vs_retain.png"
        "--out2"; arg_type=String; default="results/psnr_gain_vs_retain.png"
        "--bootstrap-block"; arg_type=Int; default=8; help="block length (in windows) used for bootstrap (min 1)"
        "--bootstrap-samples"; arg_type=Int; default=2000; help="number of bootstrap replicates (0 = fall back to normal approximation)"
        "--bootstrap-seed"; arg_type=Int; default=-1; help="seed (>=0) for reproducible bootstrap"
        "--ci"; action=:store_true; help="enable confidence-interval error bars on plots"
        "--no-ci"; action=:store_true; help="force-disable confidence-interval error bars (legacy flag)"
        "--no-title"; action=:store_true; help="suppress plot titles"
        "--zoom-max"; arg_type=Float64; default=0.0; help="if > 0, also emit zoomed plots for retains ≤ this value"
    end
    return parse_args(s)
end

function load_mera_df(path::String)
    df = CSV.read(path, DataFrame)
    if :window_id in names(df)
        df = df[coalesce.(df.window_id, 0) .> 0, :]
    end
    df = dropmissing(df, [:retain, :psnr])
    return DataFrame(retain=Float64.(df.retain), window_id=Int.(df.window_id), psnr=Float64.(df.psnr))
end

function load_baseline_df(path::String)
    df = CSV.read(path, DataFrame)
    df = df[coalesce.(df.window_id, 0) .> 0, :]
    df = dropmissing(df, [:wavelet, :retain, :psnr])
    return DataFrame(
        wavelet=String.(df.wavelet),
        retain=Float64.(df.retain),
        window_id=Int.(df.window_id),
        psnr=Float64.(df.psnr)
    )
end

function moving_block_bootstrap_mean_ci(values::Vector{Float64}, block_size::Int, n_boot::Int; seed::Union{Nothing, Int}=nothing)
    n = length(values)
    @assert n > 0 "Vector of values must be non-empty"
    μ = mean(values)
    σ = n > 1 ? std(values) : 0.0
    if n == 1
        return (mean=μ, ci_low=μ, ci_high=μ, std=σ)
    end
    if n_boot <= 0
        z = 1.959963984540054
        err = z * σ / sqrt(n)
        return (mean=μ, ci_low=μ - err, ci_high=μ + err, std=σ)
    end
    bs = clamp(block_size, 1, n)
    rng = isnothing(seed) ? MersenneTwister() : MersenneTwister(seed)
    total_blocks = n - bs + 1
    blocks = [values[i:i+bs-1] for i in 1:total_blocks]
    sample = Vector{Float64}(undef, n)
    boot_means = Vector{Float64}(undef, n_boot)
    for b in 1:n_boot
        filled = 0
        while filled < n
            block = blocks[rand(rng, 1:total_blocks)]
            for val in block
                filled += 1
                sample[filled] = val
                if filled == n
                    break
                end
            end
        end
        boot_means[b] = mean(sample)
    end
    lo = quantile(boot_means, 0.025)
    hi = quantile(boot_means, 0.975)
    return (mean=μ, ci_low=lo, ci_high=hi, std=σ)
end

function summarize_by_retain(df::DataFrame, retains::Vector{Float64}; value_col::Symbol, block_size::Int, n_boot::Int, seed_base::Union{Nothing, Int}=nothing, seed_stride::Int=1)
    means = Float64[]
    ci_low = Float64[]
    ci_high = Float64[]
    for (idx, r) in enumerate(retains)
        rows = df[df.retain .== r, :]
        vals = collect(skipmissing(Float64.(rows[!, value_col])))
        isempty(vals) && error("Nenhum valor encontrado para retain=$(r)")
        seed = isnothing(seed_base) ? nothing : seed_base + seed_stride * (idx - 1)
        summary = moving_block_bootstrap_mean_ci(vals, block_size, n_boot; seed=seed)
        push!(means, summary.mean)
        push!(ci_low, summary.ci_low)
        push!(ci_high, summary.ci_high)
    end
    return (mean=means, ci_low=ci_low, ci_high=ci_high)
end

function gain_series_for_wavelet(wavelet::AbstractString, mera_df::DataFrame, baseline_df::DataFrame, retains::Vector{Float64}, block_size::Int, n_boot::Int, seed_base::Union{Nothing, Int}=nothing)
    wavelet_str = String(wavelet)
    sub_base = baseline_df[baseline_df.wavelet .== wavelet_str, [:retain, :window_id, :psnr]]
    means = Float64[]
    ci_low = Float64[]
    ci_high = Float64[]
    for (idx, r) in enumerate(retains)
        sub_mera = mera_df[mera_df.retain .== r, [:window_id, :psnr]]
        sub_single = sub_base[sub_base.retain .== r, :]
        mera_sel = select(dropmissing(sub_mera, [:window_id, :psnr]), :window_id, :psnr => :psnr_mera)
        base_sel = select(dropmissing(sub_single, [:window_id, :psnr]), :window_id, :psnr => :psnr_baseline)
        joined = innerjoin(mera_sel, base_sel, on=:window_id)
        diffs = collect(Float64.(joined.psnr_mera .- joined.psnr_baseline))
        isempty(diffs) && error("Sem pares MERA/baseline para retain=$(r), wavelet=$(wavelet_str)")
        seed = isnothing(seed_base) ? nothing : seed_base + 1000 * (idx - 1)
        summary = moving_block_bootstrap_mean_ci(diffs, block_size, n_boot; seed=seed)
        push!(means, summary.mean)
        push!(ci_low, summary.ci_low)
        push!(ci_high, summary.ci_high)
    end
    return (mean=means, ci_low=ci_low, ci_high=ci_high)
end

function extract_means(series, mask::Union{Nothing, Vector{Bool}}=nothing)
    vals = Float64[]
    for s in series
        if mask === nothing
            append!(vals, s.mean)
        else
            for (idx, keep) in enumerate(mask)
                keep || continue
                push!(vals, s.mean[idx])
            end
        end
    end
    return vals
end

function compute_y_limits(series; mask::Union{Nothing, Vector{Bool}}=nothing)
    values = Float64[]
    values = extract_means(series, mask)
    y_min = minimum(values)
    y_max = maximum(values)
    if y_min == y_max
        ε = max(1.0, abs(y_min) * 0.05)
        y_min -= ε
        y_max += ε
    else
        margin = 0.05 * (y_max - y_min)
        y_min -= margin
        y_max += margin
    end
    return (y_min, y_max)
end

function compute_y_limits_gain(gain_series; mask::Union{Nothing, Vector{Bool}}=nothing)
    values = Float64[]
    if mask === nothing
        for g in gain_series
            append!(values, g.mean)
        end
    else
        for g in gain_series
            for (idx, keep) in enumerate(mask)
                keep || continue
                push!(values, g.mean[idx])
            end
        end
    end
    y_min = minimum(values)
    y_max = maximum(values)
    if y_min == y_max
        ε = max(0.5, abs(y_min) * 0.1)
        y_min -= ε
        y_max += ε
    else
        margin = 0.10 * (y_max - y_min)
        y_min -= margin
        y_max += margin
    end
    return (y_min, y_max)
end

function plot_psnr(retains, series, outpath::String; plot_ci::Bool=true, plot_title::String="PSNR vs retain", zoom_max::Union{Nothing, Float64}=nothing, ylims_override::Union{Nothing, Tuple{Float64, Float64}}=nothing)
    default(fontfamily="sans", size=(900, 520))
    plt = plot(legend=:bottomright, xlabel="retain", ylabel="PSNR (dB)", title=plot_title)
    mask = zoom_max === nothing ? nothing : [r <= zoom_max + 1e-12 for r in retains]
    for s in series
        if plot_ci
            lower = max.(0.0, s.mean .- s.ci_low)
            upper = max.(0.0, s.ci_high .- s.mean)
            plot!(plt, retains, s.mean, yerror=(lower, upper), marker=:circle, label=s.label)
        else
            plot!(plt, retains, s.mean, marker=:circle, label=s.label)
        end
    end
    if ylims_override !== nothing
        ylims!(plt, ylims_override...)
    else
        yl = compute_y_limits(series; mask=mask)
        ylims!(plt, yl...)
    end
    if zoom_max !== nothing
        xlims!(plt, (0, zoom_max))
    end
    savefig(plt, outpath)
end

function plot_gain(retains, gain_series, outpath::String; plot_ci::Bool=true, plot_title::String="MERA - baseline (mean ± IC)", zoom_max::Union{Nothing, Float64}=nothing, ylims_override::Union{Nothing, Tuple{Float64, Float64}}=nothing)
    default(fontfamily="sans", size=(900, 520))
    plt = plot(legend=:bottomright, xlabel="retain", ylabel="ΔPSNR (dB)", title=plot_title)
    mask = zoom_max === nothing ? nothing : [r <= zoom_max + 1e-12 for r in retains]
    for g in gain_series
        if plot_ci
            lower = max.(0.0, g.mean .- g.ci_low)
            upper = max.(0.0, g.ci_high .- g.mean)
            plot!(plt, retains, g.mean, yerror=(lower, upper), marker=:square, label=g.label)
        else
            plot!(plt, retains, g.mean, marker=:square, label=g.label)
        end
    end
    if ylims_override !== nothing
        ylims!(plt, ylims_override...)
    else
        yl = compute_y_limits_gain(gain_series; mask=mask)
        ylims!(plt, yl...)
    end
    if zoom_max !== nothing
        xlims!(plt, (0, zoom_max))
    end
    savefig(plt, outpath)
end

function run()
    args = parse_cli()
    @assert isfile(args["mera"]) "MERA CSV not found"
    @assert isfile(args["baseline"]) "Baseline CSV not found"

    mera_dir = dirname(args["mera"])
    if args["out1"] == "results/psnr_vs_retain.png" || isempty(String(args["out1"]))
        args["out1"] = joinpath(mera_dir, "psnr_vs_retain.png")
    end
    if args["out2"] == "results/psnr_gain_vs_retain.png" || isempty(String(args["out2"]))
        args["out2"] = joinpath(mera_dir, "psnr_gain_vs_retain.png")
    end

    function detect_L_suffix(dirpath::String)
        for part in splitpath(dirpath)
            m = match(r"^L_(\d+)$", part)
            if m !== nothing
                return string("_L_", m.captures[1])
            end
        end
        return ""
    end
    Lsuf = detect_L_suffix(mera_dir)
    if !isempty(Lsuf)
        default1 = joinpath(mera_dir, "psnr_vs_retain.png")
        default2 = joinpath(mera_dir, "psnr_gain_vs_retain.png")
        if args["out1"] == default1
            base, ext = splitext(default1)
            args["out1"] = base * Lsuf * ext
        end
        if args["out2"] == default2
            base, ext = splitext(default2)
            args["out2"] = base * Lsuf * ext
        end
    end

    mera_df = load_mera_df(args["mera"])
    baseline_df = load_baseline_df(args["baseline"])

    retains = sort(unique(mera_df.retain))

    block_size = max(1, Int(args["bootstrap-block"]))
    n_boot = Int(args["bootstrap-samples"])
    seed_arg = Int(args["bootstrap-seed"])
    seed_base = seed_arg >= 0 ? seed_arg : nothing
    show_ci = get(args, "ci", false)
    if get(args, "no-ci", false)
        show_ci = false
    end
    plot_ci = show_ci
    plot_title = get(args, "no-title", false) ? "" : nothing
    zoom_val = Float64(args["zoom-max"])
    zoom_max = zoom_val > 0 ? zoom_val : nothing
    has_zoom = zoom_max !== nothing && any(r -> r <= zoom_max + 1e-12, retains)

    mera_summary = summarize_by_retain(mera_df, retains; value_col=:psnr, block_size=block_size, n_boot=n_boot, seed_base=seed_base, seed_stride=7)

    wave_names = [strip(w) for w in split(args["waves"], ",") if !isempty(strip(w))]
    label_map = Dict(
        "haar" => "Haar",
        "db4" => "DB4",
        "c3" => "Coiflet-3",
        "s8" => "Symmlet-8",
        "bior4.4" => "Biorthogonal-4.4",
        "bior44" => "Biorthogonal-4.4"
    )

    series = NamedTuple[]
    push!(series, (label="MERA (learned)", mean=mera_summary.mean, ci_low=mera_summary.ci_low, ci_high=mera_summary.ci_high))

    gain_series = NamedTuple[]
    for (idx, name) in enumerate(wave_names)
        name_str = String(name)
        sub = baseline_df[baseline_df.wavelet .== name_str, :]
        isempty(sub) && error("Wavelet $(name_str) não encontrado no CSV de baseline")
        seed_offset = seed_base === nothing ? nothing : seed_base + 10000 * idx
        summary = summarize_by_retain(sub, retains; value_col=:psnr, block_size=block_size, n_boot=n_boot, seed_base=seed_offset, seed_stride=11)
        label = get(label_map, name_str, name_str)
        push!(series, (label=label, mean=summary.mean, ci_low=summary.ci_low, ci_high=summary.ci_high))

        gain_seed = seed_base === nothing ? nothing : seed_base + 20000 * idx
        gain = gain_series_for_wavelet(name_str, mera_df, baseline_df, retains, block_size, n_boot, gain_seed)
        push!(gain_series, (label="MERA - " * label, mean=gain.mean, ci_low=gain.ci_low, ci_high=gain.ci_high))
    end

    mkpath(dirname(args["out1"]))
    title_psnr = plot_title === nothing ? "PSNR vs retain" : plot_title
    default_gain_title = plot_ci ? "MERA - baseline (mean ± IC)" : "MERA - baseline"
    title_gain = plot_title === nothing ? default_gain_title : plot_title
    out1_path = args["out1"]
    out2_path = args["out2"]
    out1_base, out1_ext = splitext(out1_path)
    out2_base, out2_ext = splitext(out2_path)
    plot_psnr(retains, series, out1_path; plot_ci=plot_ci, plot_title=title_psnr)
    plot_gain(retains, gain_series, out2_path; plot_ci=plot_ci, plot_title=title_gain)
    if has_zoom
        plot_psnr(retains, series, out1_base * "_zoom" * out1_ext; plot_ci=plot_ci, plot_title=title_psnr, zoom_max=zoom_max)
        plot_gain(retains, gain_series, out2_base * "_zoom" * out2_ext; plot_ci=plot_ci, plot_title=title_gain, zoom_max=zoom_max)
    end

    b1 = out1_base
    b2 = out2_base
    try
        plot_psnr(retains, series, b1 * ".pdf"; plot_ci=plot_ci, plot_title=title_psnr)
        plot_psnr(retains, series, b1 * ".svg"; plot_ci=plot_ci, plot_title=title_psnr)
        plot_gain(retains, gain_series, b2 * ".pdf"; plot_ci=plot_ci, plot_title=title_gain)
        plot_gain(retains, gain_series, b2 * ".svg"; plot_ci=plot_ci, plot_title=title_gain)
        if has_zoom
            plot_psnr(retains, series, b1 * "_zoom.pdf"; plot_ci=plot_ci, plot_title=title_psnr, zoom_max=zoom_max)
            plot_psnr(retains, series, b1 * "_zoom.svg"; plot_ci=plot_ci, plot_title=title_psnr, zoom_max=zoom_max)
            plot_gain(retains, gain_series, b2 * "_zoom.pdf"; plot_ci=plot_ci, plot_title=title_gain, zoom_max=zoom_max)
            plot_gain(retains, gain_series, b2 * "_zoom.svg"; plot_ci=plot_ci, plot_title=title_gain, zoom_max=zoom_max)
        end
    catch e
        @warn "Failed to export PDF/SVG" error=e
    end

    println("Saved: ", args["out1"])
    println("Saved: ", args["out2"])
    println("Saved: ", b1 * ".pdf")
    println("Saved: ", b1 * ".svg")
    println("Saved: ", b2 * ".pdf")
    println("Saved: ", b2 * ".svg")
end

run()
