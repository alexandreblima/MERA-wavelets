#!/usr/bin/env julia

# Summarize ΔH (H_reconstructed - H_orig) by retain and plot with bootstrap CIs

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
        "--csv"; arg_type=String; help="Path to hurst_compare_retains.csv"; required=true
        "--out-prefix"; arg_type=String; default=""; help="Output prefix (directory + base). Defaults next to CSV."
        "--title"; arg_type=String; default="ΔH vs retain";
        "--bootstrap-block"; arg_type=Int; default=8; help="block length (in windows) for the bootstrap"
        "--bootstrap-samples"; arg_type=Int; default=2000; help="number of bootstrap replicates (0 = fallback to normal approximation)"
        "--bootstrap-seed"; arg_type=Int; default=-1; help="seed (>=0) for reproducible bootstrap"
        "--ci"; action=:store_true; help="enable confidence-interval error bars on the plot"
        "--no-ci"; action=:store_true; help="force-disable confidence-interval error bars (legacy flag)"
        "--no-title"; action=:store_true; help="suppress the plot title"
        "--zoom-max"; arg_type=Float64; default=0.0; help="if > 0, also emit a zoomed plot using retains ≤ this value"
    end
    return parse_args(s)
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

function collect_values(df::AbstractDataFrame, col::Symbol)
    vals = df[!, col]
    return collect(skipmissing(Float64.(vals)))
end

function infer_out_prefix(csvpath::String)
    dir = dirname(csvpath)
    base = "hurst_preservation"
    parent = splitpath(dir) |> last
    if startswith(parent, "L_")
        base *= "_" * replace(parent, "L_"=>"L_")
    end
    return joinpath(dir, base)
end

function render_plot(retains::Vector{Float64}, summaries::Dict{Symbol, Vector{NamedTuple}}; series_syms, label_map, marker_map, plot_ci::Bool, plot_title::String, out_prefix::String, zoom_max::Union{Nothing, Float64}=nothing)
    idxs = collect(eachindex(retains))
    if zoom_max !== nothing
        tol = 1e-12
        idxs = [i for (i, r) in enumerate(retains) if r <= zoom_max + tol]
        isempty(idxs) && return nothing
        retains = retains[idxs]
    end

    default(size=(900, 600))
    p = plot(title=plot_title, xlabel="retain", ylabel="ΔH (recon - orig)", legend=:topright)
    for sym in series_syms
        data = summaries[sym][idxs]
        means = [d.mean for d in data]
        if plot_ci
            lower = [max(0.0, data[i].mean - data[i].ci_low) for i in eachindex(data)]
            upper = [max(0.0, data[i].ci_high - data[i].mean) for i in eachindex(data)]
            plot!(
                p,
                retains,
                means;
                yerror = (lower, upper),
                label = get(label_map, sym, String(sym)),
                marker = get(marker_map, sym, :circle),
                ms = 6
            )
        else
            plot!(
                p,
                retains,
                means;
                label = get(label_map, sym, String(sym)),
                marker = get(marker_map, sym, :circle),
                ms = 6
            )
        end
    end
    if zoom_max !== nothing
        xlims!(p, (0, zoom_max))
    end
    png_path = out_prefix * ".png"
    pdf_path = out_prefix * ".pdf"
    svg_path = out_prefix * ".svg"
    savefig(p, png_path)
    savefig(p, pdf_path)
    savefig(p, svg_path)
    return (png_path, pdf_path, svg_path)
end

function main()
    args = parse_cli()
    csvpath = args["csv"]
    df = CSV.read(csvpath, DataFrame)

    required = [:retain, :deltaH_haar, :deltaH_learn]
    for c in required
        hasproperty(df, c) || error("Missing column $(c) in $(csvpath)")
    end

    baseline_order = ["db4", "c3", "s8", "bior4_4"]
    optional_syms = Symbol[]
    for base in baseline_order
        sym = Symbol("deltaH_" * base)
        hasproperty(df, sym) && push!(optional_syms, sym)
    end
    for name in propertynames(df)
        sname = String(name)
        if startswith(sname, "deltaH_") && !(name in optional_syms) && name ∉ (:deltaH_haar, :deltaH_learn)
            push!(optional_syms, name)
        end
    end

    series_syms = vcat([:deltaH_haar, :deltaH_learn], optional_syms)

    block_size = max(1, Int(args["bootstrap-block"]))
    n_boot = Int(args["bootstrap-samples"])
    seed_arg = Int(args["bootstrap-seed"])
    seed_base = seed_arg >= 0 ? seed_arg : nothing
    show_ci = get(args, "ci", false)
    if get(args, "no-ci", false)
        show_ci = false
    end
    plot_ci = show_ci
    plot_title = get(args, "no-title", false) ? "" : String(args["title"])
    zoom_val = Float64(args["zoom-max"])
    zoom_max = zoom_val > 0 ? zoom_val : nothing

    g = groupby(df, :retain)
    retains = Float64[]
    summaries = Dict{Symbol, Vector{NamedTuple}}()
    for sym in series_syms
        summaries[sym] = Vector{NamedTuple}(undef, length(g))
    end

    for (idx, sub) in enumerate(g)
        push!(retains, first(sub.retain))
        for (j, sym) in enumerate(series_syms)
            vals = collect_values(sub, sym)
            isempty(vals) && error("Nenhum valor para coluna $(sym) na retain=$(first(sub.retain))")
            seed = isnothing(seed_base) ? nothing : seed_base + 1000 * (j - 1) + idx
            summaries[sym][idx] = moving_block_bootstrap_mean_ci(vals, block_size, n_boot; seed=seed)
        end
    end

    order = sortperm(retains)
    retains = retains[order]
    for sym in keys(summaries)
        summaries[sym] = summaries[sym][order]
    end

    out_prefix = isempty(args["out-prefix"]) ? infer_out_prefix(csvpath) : args["out-prefix"]
    summary = DataFrame(retain = retains)
    for sym in series_syms
        data = summaries[sym]
        means = [x.mean for x in data]
        ci_low = [x.ci_low for x in data]
        ci_high = [x.ci_high for x in data]
        summary[!, Symbol(string(sym), "_mean")] = means
        summary[!, Symbol(string(sym), "_ci_low")] = ci_low
        summary[!, Symbol(string(sym), "_ci_high")] = ci_high
    end
    summary_path = out_prefix * "_summary.csv"
    CSV.write(summary_path, summary)

    label_map = Dict(
        :deltaH_haar => "Haar (MERA)",
        :deltaH_learn => "Learned (MERA)",
        :deltaH_db4 => "DB4 (DWT)",
        :deltaH_c3 => "C3 (DWT)",
        :deltaH_s8 => "S8 (DWT)",
        :deltaH_bior4_4 => "Bior4.4 (DWT)"
    )
    marker_map = Dict(
        :deltaH_haar => :circle,
        :deltaH_learn => :square,
        :deltaH_db4 => :diamond,
        :deltaH_c3 => :utriangle,
        :deltaH_s8 => :star5,
        :deltaH_bior4_4 => :hexagon
    )
    plot_main = render_plot(retains, summaries; series_syms=series_syms, label_map=label_map, marker_map=marker_map, plot_ci=plot_ci, plot_title=plot_title, out_prefix=out_prefix)
    if zoom_max !== nothing
        zoom_prefix = out_prefix * "_zoom"
        plot_zoom = render_plot(retains, summaries; series_syms=series_syms, label_map=label_map, marker_map=marker_map, plot_ci=plot_ci, plot_title=plot_title, out_prefix=zoom_prefix, zoom_max=zoom_max)
        if plot_zoom === nothing
            @warn "Zoom plot skipped: no retains ≤ $(zoom_max)"
        end
    end
    if plot_main !== nothing
        png_path, _, _ = plot_main
        @info "Saved" summary=basename(summary_path) png=basename(png_path)
    else
        @warn "Main plot skipped: no retain values to plot"
    end
end

main()
