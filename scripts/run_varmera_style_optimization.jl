#!/usr/bin/env julia

# CPU-only runner: doVarMERA-inspired optimization over MERA tensors
# - No NVIDIA GPU usage
# - Progress logs per stage and iteration (via engine prints)
# - Measures runtime per window and writes incremental CSV

ENV["CUDA_VISIBLE_DEVICES"] = ""
ENV["JULIA_CUDA_USE_BINARYBUILDER"] = "false"

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using ArgParse
using CSV
using DataFrames
using Statistics
using Printf
using Wave6G
using Wavelets
using Dates
using LinearAlgebra # for BLAS.set_num_threads
using Random
using SHA
using JLD2

function parse_cli()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--data"; arg_type=String; default="data/MAWI_bytes_1ms.csv";
        "--window-size"; arg_type=Int; default=1024
        "--step"; arg_type=Int; default=1024
        "--num-windows"; arg_type=Int; default=0
        "--start-window"; arg_type=Int; default=1
        "--retain"; arg_type=Float64; default=0.1
        "--retains"; arg_type=String; default="0.01,0.02,0.05,0.1"; help="comma-separated list of retains (fractions) for PSNR sweep"
        "--mera-L"; arg_type=Int; default=5
        "--mera-chi"; arg_type=Int; default=2
        "--mera-chimid"; arg_type=Int; default=2
        "--stage1-iters"; arg_type=Int; default=50
        "--stage1-lr"; arg_type=Float64; default=5e-3
        "--stage2-iters"; arg_type=Int; default=50
        "--stage2-lr"; arg_type=Float64; default=2.5e-3
        "--init"; arg_type=String; default="haar"; help="initialization: haar|random"
        "--param"; arg_type=String; default="mera"; help="parametrization: mera|angle"
        "--sparsity-weight"; arg_type=Float64; default=1.0; help="weight for L1 sparsity term"
        "--mse-weight"; arg_type=Float64; default=0.0; help="weight for MSE term (0 disables)"
        "--early-stop"; action=:store_true; help="enable early stopping based on tolerance/patience"
        "--tol"; arg_type=Float64; default=1e-4; help="relative tolerance for early stopping"
        "--patience"; arg_type=Int; default=5; help="patience (iterations without improvement)"
        "--min-iter"; arg_type=Int; default=10; help="minimum iterations before checking early stop"
        "--output"; arg_type=String; default="results/varmera_test.csv"
        "--threads"; arg_type=Int; default=Threads.nthreads()
        "--parallel-windows"; action=:store_true
        "--blas-threads"; arg_type=Int; default=0; help="if >0, set BLAS.set_num_threads(n)"
        "--disable-intra-window"; action=:store_true; help="set W6G_INTRA_THREADS=0 to avoid nested parallelism (threads only across windows)"
        "--seed"; arg_type=Int; default=12345
        "--save-models"; action=:store_true; help="save per-window model checkpoints (wC/vC) as JLD2 next to output"
        "--models-dir"; arg_type=String; default=""; help="optional custom directory to save models (defaults to <dirname(output)>/models)"
        "--analysis-wavelet"; arg_type=String; default="db4"; help="Wavelet para métricas Hurst/multifractal (aliases: db4, c3, s8, haar, ...)"
        "--baseline-waves"; arg_type=String; default="db4,c3,s8,bior4.4"; help="Wavelets DWT para baselines ΔH/PSNR (aliases: db4, c3|coiflet3, s8|sym8, bior4.4|cdf97)"
    end
    return parse_args(s)
end

psnr(original, reconstructed) = begin
    mse = mean(abs2, reconstructed .- original)
    max_val = maximum(abs.(original))
    mse < eps() || max_val == 0 ? Inf : 10.0 * log10(max_val^2 / mse)
end

const DEFAULT_BASELINE_WAVES = ("db4", "c3", "s8", "bior4.4")

function normalize_wavelet_alias(name::AbstractString)
    s = lowercase(strip(name))
    s = replace(s, ' ' => "")
    s = replace(s, '-' => "")
    s = replace(s, '.' => "")
    if isempty(s)
        return ""
    elseif s == "haar" || s == "db1" || s == "daubechies1"
        return "haar"
    elseif s == "db4" || s == "daubechies4"
        return "db4"
    elseif s == "c3" || s == "coif3" || s == "coiflet3" || s == "coif6"
        return "c3"
    elseif s == "s8" || s == "sym8" || s == "symmlet8" || s == "symlet8"
        return "s8"
    elseif s == "bior44" || s == "biorthogonal44" || s == "biorthogonal44" || s == "cdf97"
        return "bior4.4"
    else
        return s
    end
end

function wavelet_column_key(alias::String)
    replace(alias, "." => "_")
end

function baseline_wavelet(alias::String)
    if alias == "haar"
        return wavelet(Wavelets.WT.haar)
    elseif alias == "db4"
        return wavelet(Wavelets.WT.db4)
    elseif alias == "c3"
        return wavelet(Wavelets.WT.coif6)
    elseif alias == "s8"
        return wavelet(Wavelets.WT.sym8)
    elseif alias == "bior4.4"
        return Wavelets.WT.wavelet(Wavelets.WT.cdf97, Wavelets.WT.LiftingTransform())
    else
        error("Unsupported baseline wavelet alias: $alias")
    end
end

function threshold_mera(details::Vector, retain::Float64)
    filtered = similar(details)
    kept = 0
    total = 0
    for i in eachindex(details)
        v = details[i]
        total += length(v)
        k = clamp(Int(ceil(retain * length(v))), 0, length(v))
        if k == 0
            filtered[i] = fill!(similar(v), zero(eltype(v)))
        else
            idx = partialsortperm(abs.(v), rev=true, 1:k)
            m = zeros(eltype(v), length(v))
            m[idx] .= v[idx]
            filtered[i] = m
            kept += k
        end
    end
    return filtered, kept, total
end

function threshold_dwt_wavelet(coeffs::Vector{Float64}, N::Int, retain::Float64, w)
    c = copy(coeffs)
    max_levels = Wavelets.maxtransformlevels(N)
    kept = 0
    total = 0
    for level in 1:max_levels
        rng = Wavelets.detailrange(N, level)
        len = length(rng)
        total += len
        k = clamp(Int(ceil(retain * len)), 0, len)
        if k == 0
            c[rng] .= 0.0
            continue
        end
        v = c[rng]
        idx = partialsortperm(abs.(v), rev=true, 1:k)
        mask = zeros(Bool, len)
        mask[idx] .= true
        for j in 1:len
            c[rng[j]] = mask[j] ? v[j] : 0.0
        end
        kept += k
    end
    xr = idwt(c, w)
    return Float64.(xr), kept, total
end

hurst_default(x::AbstractVector{<:Real}) = Wave6G.wavelet_hurst_estimate(Float64.(x); min_level=4, max_level=12, model=:fgn).H

function multifractal_metrics(x::AbstractVector{<:Real})
    Wave6G.wavelet_multifractal_metrics(Float64.(x); min_level=2, max_level=12)
end

function format_eta(sec::Real)
    s = max(0.0, float(sec))
    h = floor(Int, s / 3600)
    m = floor(Int, (s - h * 3600) / 60)
    r = round(Int, s - h * 3600 - m * 60)
    return @sprintf("%02d:%02d:%02d", h, m, r)
end

function run_window(window_id::Int, signal::AbstractVector{<:Real}, cfg::Dict{String,Any})
    try
        # Hard-disable CUDA inside Wave6G engine
        try
            Base.eval(Wave6G, :(const _cuda_available = false))
        catch
        end

        L = cfg["mera-L"]
        chi = cfg["mera-chi"]
        chimid = cfg["mera-chimid"]
        init_sym = cfg["init"] == "haar" ? :haar : :random

    @info "[window=$(window_id)] starting" L=L chi=chi chimid=chimid Threads=Threads.nthreads()
        t0 = time()

        original = Float64.(signal)
        N = length(original)

        # Prepare state (normalizes data internally)
        state = Wave6G.prepare_variational_state(original; L=L, chi=chi, chimid=chimid, normalize=true, init=init_sym)

        # doVarMERA-inspired schedule: two stages; engine logs per-iteration progress
        schedule = [
            (numiter=cfg["stage1-iters"], lr=cfg["stage1-lr"], reinit=false, init=:previous,
             opts=Dict("param"=>cfg["param"],
                       "early_stop"=>get(cfg, "early-stop", false),
                       "tol"=>get(cfg, "tol", 1e-4),
                       "patience"=>get(cfg, "patience", 5),
                       "min_iter"=>get(cfg, "min-iter", 10),
                       "sparsity_weight"=>get(cfg, "sparsity-weight", 1.0),
                       "mse_weight"=>get(cfg, "mse-weight", 0.0))),
            (numiter=cfg["stage2-iters"], lr=cfg["stage2-lr"], reinit=false, init=:previous,
             opts=Dict("param"=>cfg["param"],
                       "early_stop"=>get(cfg, "early-stop", false),
                       "tol"=>get(cfg, "tol", 1e-4),
                       "patience"=>get(cfg, "patience", 5),
                       "min_iter"=>get(cfg, "min-iter", 10),
                       "sparsity_weight"=>get(cfg, "sparsity-weight", 1.0),
                       "mse_weight"=>get(cfg, "mse-weight", 0.0)))
        ]

        # Run optimization (AD with per-iteration SVD projection under the hood)
        res = Wave6G.optimize_variational_schedule!(state; schedule=schedule)
        # Extract effective iterations used per stage (when early stopping is enabled)
        it1_used = try
            length(res.state.history) >= 1 ? res.state.history[1].iters_used : cfg["stage1-iters"]
        catch
            cfg["stage1-iters"]
        end
        it2_used = try
            length(res.state.history) >= 2 ? res.state.history[2].iters_used : cfg["stage2-iters"]
        catch
            cfg["stage2-iters"]
        end
        wC = res.state.wC; vC = res.state.vC; uC = res.state.uC

        # Perfect reconstruction diagnostic (ρ = 1, no thresholding)
        coeffs_approx_learn, coeffs_detail_learn = Wave6G.mera_analyze(Float32.(original), wC, vC)
        reconstructed_pr = Wave6G.mera_synthesize(coeffs_approx_learn, coeffs_detail_learn, wC, vC)
        pr_residual = Float64.(real.(reconstructed_pr)) .- original
        pr_mse = mean(abs2, pr_residual)
        σ_orig = std(original)
        σ_orig = σ_orig < eps() ? 1.0 : σ_orig
        pr_relative_error = sqrt(pr_mse) / σ_orig
        @info "[window=$(window_id)] PR check" pr_mse=pr_mse pr_relative_error=pr_relative_error

        # Optionally save per-window model (checkpoint)
        if get(cfg, "save-models", false)
            models_dir = get(cfg, "models-dir", "")
            if !isempty(models_dir)
                try
                    mkpath(models_dir)
                    model_path = joinpath(models_dir, @sprintf("window_%d_%s.jld2", window_id, get(cfg, "param", "mera")))
                    meta = Dict(
                        :L => L,
                        :chi => chi,
                        :chimid => chimid,
                        :param => get(cfg, "param", "mera"),
                        :seed => get(cfg, "seed", missing),
                        :window_id => window_id,
                        :timestamp => string(Dates.now()),
                    )
                    @save model_path wC vC uC meta
                    @info "Saved model" path=model_path
                catch e
                    @warn "Failed to save model" error=e
                end
            end
        end

        # Analyze once (learned + Haar)
        haar_tensors = Wave6G._haar_tensors(L)
        coeffs_approx_haar, coeffs_detail_haar = Wave6G.mera_analyze(Float32.(original), haar_tensors.wC, haar_tensors.vC)

        # Window analytics (Hurst & Multifractal) on original
        analysis_wavelet = get(cfg, "analysis-wavelet", "db4")
        hurst_orig = Wave6G.wavelet_hurst_estimate(original; wavelet=analysis_wavelet).H
        multi_orig = Wave6G.wavelet_multifractal_metrics(original; wavelet=analysis_wavelet)

        baseline_aliases = get(cfg, "baseline_aliases", collect(DEFAULT_BASELINE_WAVES))
        baseline_wavelets = Dict{String, Any}()
        coeffs_dwt = Dict{String, Vector{Float64}}()
        for alias in baseline_aliases
            w = baseline_wavelet(alias)
            baseline_wavelets[alias] = w
            if alias != "haar"
                coeffs_dwt[alias] = dwt(original, w)
            end
        end

        # Retain sweep
        retains = cfg["retains_vec"]
        rows = Vector{Dict{Symbol,Any}}(undef, length(retains))
        for (j, retain) in enumerate(retains)
            t_ret = time()

            filtered_haar, kept_haar, total_haar = threshold_mera(coeffs_detail_haar, retain)
            rec_haar = Wave6G.mera_synthesize(coeffs_approx_haar, filtered_haar, haar_tensors.wC, haar_tensors.vC)
            x_haar = Float64.(real.(rec_haar))
            H_haar = hurst_default(x_haar)
            M_haar = multifractal_metrics(x_haar)
            psnr_haar = psnr(original, x_haar)
            deltaH_haar = H_haar - hurst_orig

            filtered_learn, kept_learn, total_learn = threshold_mera(coeffs_detail_learn, retain)
            rec_learn = Wave6G.mera_synthesize(coeffs_approx_learn, filtered_learn, wC, vC)
            x_learn = Float64.(real.(rec_learn))
            H_learn = hurst_default(x_learn)
            M_learn = multifractal_metrics(x_learn)
            psnr_learn = psnr(original, x_learn)
            deltaH_learn = H_learn - hurst_orig

            row = Dict{Symbol,Any}()
            row[:window_id] = window_id
            row[:retain] = retain
            row[:L] = L
            row[:chi] = chi
            row[:chimid] = chimid
            row[:iters1] = cfg["stage1-iters"]
            row[:iters2] = cfg["stage2-iters"]
            row[:iters1_used] = it1_used
            row[:iters2_used] = it2_used
            row[:threads] = Threads.nthreads()
            row[:analysis_wavelet] = analysis_wavelet
            row[:param] = get(cfg, "param", "mera")
            row[:seed] = get(cfg, "seed", missing)
            row[:git_commit] = get(cfg, "git_commit", missing)
            row[:git_dirty] = get(cfg, "git_dirty", missing)
            row[:manifest_sha256] = get(cfg, "manifest_sha256", missing)
            row[:seconds] = round(time() - t_ret, digits=3)
            row[:error] = ""

            row[:H_orig] = hurst_orig
            row[:H_haar] = H_haar
            row[:H_learn] = H_learn
            row[:deltaH_haar] = deltaH_haar
            row[:deltaH_learn] = deltaH_learn
            row[:psnr_haar] = psnr_haar
            row[:psnr_learn] = psnr_learn
            row[:kept_haar] = kept_haar
            row[:total_haar] = total_haar
            row[:kept_learn] = kept_learn
            row[:total_learn] = total_learn

            row[:psnr] = psnr_learn
            row[:deltaH] = deltaH_learn
            row[:kept] = kept_learn
            row[:total] = total_learn
            row[:hurst_H] = hurst_orig
            row[:hurst_H_recon] = H_learn
            row[:pr_mse] = pr_mse
            row[:pr_relative_error] = pr_relative_error

            row[:alpha_min] = multi_orig.alpha_min
            row[:alpha_max] = multi_orig.alpha_max
            row[:alpha_width] = multi_orig.alpha_width
            row[:alpha_peak] = multi_orig.alpha_peak
            row[:f_alpha_peak] = multi_orig.f_alpha_peak
            row[:alpha_width_orig] = multi_orig.alpha_width
            row[:alpha_peak_orig] = multi_orig.alpha_peak
            row[:f_alpha_peak_orig] = multi_orig.f_alpha_peak

            row[:alpha_width_haar] = M_haar.alpha_width
            row[:alpha_peak_haar] = M_haar.alpha_peak
            row[:f_alpha_peak_haar] = M_haar.f_alpha_peak
            row[:delta_alpha_width_haar] = M_haar.alpha_width - multi_orig.alpha_width

            row[:alpha_width_learn] = M_learn.alpha_width
            row[:alpha_peak_learn] = M_learn.alpha_peak
            row[:f_alpha_peak_learn] = M_learn.f_alpha_peak
            row[:delta_alpha_width_learn] = M_learn.alpha_width - multi_orig.alpha_width

            for alias in baseline_aliases
                alias == "haar" && continue
                w = baseline_wavelets[alias]
                coeffs_base = coeffs_dwt[alias]
                x_wave, kept_wave, total_wave = threshold_dwt_wavelet(coeffs_base, N, retain, w)
                H_wave = hurst_default(x_wave)
                M_wave = multifractal_metrics(x_wave)
                psnr_wave = psnr(original, x_wave)
                prefix = wavelet_column_key(alias)
                row[Symbol("H_" * prefix)] = H_wave
                row[Symbol("deltaH_" * prefix)] = H_wave - hurst_orig
                row[Symbol("psnr_" * prefix)] = psnr_wave
                row[Symbol("kept_" * prefix)] = kept_wave
                row[Symbol("total_" * prefix)] = total_wave
                row[Symbol("alpha_width_" * prefix)] = M_wave.alpha_width
                row[Symbol("alpha_peak_" * prefix)] = M_wave.alpha_peak
                row[Symbol("f_alpha_peak_" * prefix)] = M_wave.f_alpha_peak
                row[Symbol("delta_alpha_width_" * prefix)] = M_wave.alpha_width - multi_orig.alpha_width
            end

            rows[j] = row
        end
        secs = round(time() - t0, digits=3)
        @info "[window=$(window_id)] finished" seconds=secs
        return rows
    catch e
        @error "window error" window_id exception=(e, catch_backtrace())
        return [Dict(:window_id=>window_id, :retain=>missing, :psnr=>missing, :kept=>missing, :total=>missing, :seconds=>missing, :L=>missing, :chi=>missing, :chimid=>missing, :iters1=>missing, :iters2=>missing, :threads=>Threads.nthreads(), :hurst_H=>missing, :alpha_min=>missing, :alpha_max=>missing, :alpha_width=>missing, :alpha_peak=>missing, :f_alpha_peak=>missing, :analysis_wavelet=>get(cfg, "analysis-wavelet", "db4"), :pr_mse=>missing, :pr_relative_error=>missing, :error=>string(e))]
    end
end

function main()
    args = parse_cli()
    # Seed for reproducibility
    try
        Random.seed!(args["seed"])
    catch e
        @warn "Failed to set seed" error=e
    end
    # Optionally inform threads; cannot set Threads at runtime, but we log it
    @info "Threads active" n=Threads.nthreads()
    if get(args, "disable-intra-window", false)
        ENV["W6G_INTRA_THREADS"] = "0"
        @info "Intra-window threading disabled via ENV[W6G_INTRA_THREADS]=0"
    end
        # BLAS threads control (optional)
        if get(args, "blas-threads", 0) > 0
            try
                BLAS.set_num_threads(args["blas-threads"])
                @info "BLAS threads set" n=args["blas-threads"]
            catch e
                @warn "Failed to set BLAS threads" error=e
            end
        end
    # Parse retains vector
    retains_vec = begin
        rs = get(args, "retains", "0.01,0.02,0.05,0.1,0.2,0.4,0.8")
        v = [parse(Float64, strip(x)) for x in split(rs, ",") if !isempty(strip(x))]
        isempty(v) && (v = [get(args, "retain", 0.1)])
        v
    end

    baseline_aliases = begin
        ws = get(args, "baseline-waves", join(DEFAULT_BASELINE_WAVES, ","))
        parsed = String[]
        for token in split(ws, ",")
            alias = normalize_wavelet_alias(token)
            isempty(alias) && continue
            alias in parsed && continue
            push!(parsed, alias)
        end
        for mandatory in DEFAULT_BASELINE_WAVES
            mandatory in parsed || push!(parsed, mandatory)
        end
        parsed
    end
    args["baseline_aliases"] = baseline_aliases

    dataset = Wave6G.prepare_window_dataset(args["data"]; window_size=args["window-size"], step=args["step"], normalize=true)
    analysis_wavelet = get(args, "analysis-wavelet", "db4")

    global_series = dataset.truncated
    if !(global_series isa AbstractVector) || isempty(global_series)
        @warn "Global Hurst estimation skipped: empty series after truncation" data=args["data"]
    else
        global_stats = Wave6G.wavelet_hurst_estimate_with_ci(global_series; wavelet=analysis_wavelet, min_level=2, max_level=nothing, model=:fgn)
        trace_dir = dirname(args["output"])
        mkpath(trace_dir)
        global_hurst_path = joinpath(trace_dir, "hurst_global_summary.csv")
        num_levels = global_stats.num_levels
        level_min = num_levels > 0 ? minimum(global_stats.levels) : missing
        level_max = num_levels > 0 ? maximum(global_stats.levels) : missing
        removed_val = hasproperty(dataset, :removed) ? dataset.removed : missing
        global_df = DataFrame(
            H = [global_stats.H],
            ci_low = [global_stats.ci_low],
            ci_high = [global_stats.ci_high],
            slope = [global_stats.slope],
            intercept = [global_stats.intercept],
            se_H = [global_stats.se_H],
            se_slope = [global_stats.se_slope],
            residual_variance = [global_stats.residual_variance],
            residual_std = [global_stats.residual_std],
            dof = [global_stats.dof],
            num_levels = [num_levels],
            level_min = [level_min],
            level_max = [level_max],
            min_level_param = [2],
            max_level_param = [missing],
            alpha = [0.05],
            wavelet = [string(analysis_wavelet)],
            series_length = [length(global_series)],
            removed = [removed_val]
        )
        CSV.write(global_hurst_path, global_df)
        @info "Global Hurst estimate (entire trace)" H=global_stats.H ci_low=global_stats.ci_low ci_high=global_stats.ci_high num_levels=global_stats.num_levels path=global_hurst_path
    end

    windows = dataset.windows
    start = max(1, args["start-window"])
    n = args["num-windows"] > 0 ? min(args["num-windows"], length(windows)-start+1) : length(windows)-start+1

    mkpath(dirname(args["output"]))
    isfile(args["output"]) && rm(args["output"]; force=true)

    # inject retains into args (mutable Dict)
    args["retains_vec"] = retains_vec
    # Gather repo metadata
    proj_root = abspath(joinpath(@__DIR__, ".."))
    git_commit = try
        readchomp(`git -C $(proj_root) rev-parse HEAD`)
    catch
        missing
    end
    git_dirty = try
        st = readchomp(`git -C $(proj_root) status --porcelain`)
        !isempty(strip(st))
    catch
        missing
    end
    manifest_sha = try
        mp = joinpath(proj_root, "Manifest.toml")
        isfile(mp) ? bytes2hex(sha256(read(mp))) : missing
    catch
        missing
    end
    args["git_commit"] = git_commit
    args["git_dirty"] = git_dirty
    args["manifest_sha256"] = manifest_sha

    t_global = time()
    done = Threads.Atomic{Int}(0)
    eta_lock = ReentrantLock()
    print_progress = function ()
        d = done[]
        elapsed = time() - t_global
        avg = d > 0 ? (elapsed / d) : 0.0
        rem = max(n - d, 0)
        eta = rem * avg
        @info "Progress" done=d total=n elapsed=round(elapsed, digits=1) eta=format_eta(eta)
    end
    if get(args, "parallel-windows", false) && n > 1 && Threads.nthreads() > 1
        write_lock = ReentrantLock()
        Threads.@threads for i in 1:n
            idx = start + i - 1
            rows = run_window(idx, windows[idx], args)
            df_rows = DataFrame(rows)
            Base.lock(write_lock) do
                CSV.write(args["output"], df_rows; append=isfile(args["output"]))
            end
            Threads.atomic_add!(done, 1)
            Base.lock(eta_lock) do
                print_progress()
            end
        end
    else
        for i in 1:n
            idx = start + i - 1
            rows = run_window(idx, windows[idx], args)
            df_rows = DataFrame(rows)
            CSV.write(args["output"], df_rows; append=isfile(args["output"]))
            Threads.atomic_add!(done, 1)
            print_progress()
        end
    end
    total_secs = round(time() - t_global, digits=3)
    @info "All done" windows=n seconds=total_secs output=args["output"]
    # Summary row omitted to keep CSV rectangular for downstream tooling
end

main()
