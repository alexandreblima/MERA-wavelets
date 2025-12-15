#!/usr/bin/env julia
# Compatibility shim: forward to the renamed runner
Base.include(Main, joinpath(@__DIR__, "mera_mawi_runner.jl"))#!/usr/bin/env julia

# Strongly disable CUDA usage in this run to avoid GPU initialization paths
ENV["CUDA_VISIBLE_DEVICES"] = ""
ENV["JULIA_CUDA_USE_BINARYBUILDER"] = "false"

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

## (ENV already set above)

using ArgParse
using CSV
using DataFrames
using Statistics
using Printf
using Wave6G

function parse_cli()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--data"; arg_type=String; default="data/small_test.csv"
        "--window-size"; arg_type=Int; default=4096
        "--step"; arg_type=Int; default=4096
        "--retain"; arg_type=Float64; default=0.1
        "--retains"; arg_type=String; default="0.01,0.02,0.05,0.1"
        "--start-window"; arg_type=Int; default=1
        "--num-windows"; arg_type=Int; default=1
        "--mera-L"; arg_type=Int; default=3; help="Número de níveis (L). Para séries 1D, L=3 é um ponto inicial rápido/estável."
        "--mera-chi"; arg_type=Int; default=2; help="Dimensão de ligação χ (site coarse-grained). Recomendado χ=2 para séries 1D."
        "--mera-chimid"; arg_type=Int; default=2; help="Dimensão intermediária χ_mid (canal do disentangler). Recomendado χ_mid=2 para séries 1D."
        "--stage1-iters"; arg_type=Int; default=10
        "--stage1-lr"; arg_type=Float64; default=5e-3
        "--stage2-iters"; arg_type=Int; default=10
        "--stage2-lr"; arg_type=Float64; default=2.5e-3
        "--output"; arg_type=String; default="results/debug_test.csv"
        "--use-gpu"; action=:store_true
        "--warm-start-haar"; action=:store_true
        "--train"; action=:store_true
        "--analysis-wavelet"; arg_type=String; default="db4"; help="Wavelet para métricas Hurst/multifractal (aliases: db4, c3, s8, haar, ...)"
    end
    return parse_args(s)
end

psnr(original, reconstructed) = begin
    mse = mean(abs2, reconstructed .- original)
    max_val = maximum(abs.(original))
    mse < eps() || max_val == 0 ? Inf : 10.0 * log10(max_val^2 / mse)
end

to_host(x) = x

function format_eta(sec::Real)
    s = max(0.0, float(sec))
    h = floor(Int, s / 3600)
    m = floor(Int, (s - h * 3600) / 60)
    r = round(Int, s - h * 3600 - m * 60)
    return @sprintf("%02d:%02d:%02d", h, m, r)
end

function run_window_optimized(window_id, signal, cfg)
    try
        t0 = time()
        println("Starting window $window_id")
        original = Float64.(signal)
        L = cfg["mera-L"]
        wC = nothing; vC = nothing
        if get(cfg, "train", false)
            # Force CPU-only for training to avoid CUDA paths
            try
                Base.eval(Wave6G, :(const _cuda_available = false))
            catch err
                # ignore redefinition warnings
            end
            # Tiny training schedule (CPU-only): fast and safe
            base_state = Wave6G.prepare_variational_state(original; L=L, chi=cfg["mera-chi"], chimid=cfg["mera-chimid"], normalize=true, init=(cfg["warm-start-haar"] ? :haar : :random))
            schedule = [ (numiter=cfg["stage1-iters"], lr=cfg["stage1-lr"], reinit=false, init=:previous),
                         (numiter=cfg["stage2-iters"], lr=cfg["stage2-lr"], reinit=false, init=:previous) ]
            res = Wave6G.optimize_variational_schedule!(base_state; schedule=schedule)
            wC = res.state.wC; vC = res.state.vC
        else
            # Haar fallback (no training)
            tensors = Wave6G._haar_tensors(L)
            wC = tensors.wC; vC = tensors.vC
        end
        # Analyze once (CPU-only)
        coeffs_approx, coeffs_detail = Wave6G.mera_analyze(Float32.(original), wC, vC)

        # Window analytics on original signal
        analysis_wavelet = get(cfg, "analysis-wavelet", "db4")
        hurst = Wave6G.wavelet_hurst_estimate(original; wavelet=analysis_wavelet)
        multi = Wave6G.wavelet_multifractal_metrics(original; wavelet=analysis_wavelet)

        # Determine retains vector
        retains_vec = get(cfg, "retains_vec", nothing)
        if retains_vec === nothing
            retains_vec = [get(cfg, "retain", 0.1)]
        end

        total = sum(length, coeffs_detail)
        rows = Vector{Dict}(undef, length(retains_vec))
        for (j, retain) in enumerate(retains_vec)
            kept = 0
            filtered_detail = Vector{typeof(coeffs_detail[1])}(undef, length(coeffs_detail))
            for i in eachindex(coeffs_detail)
                v = coeffs_detail[i]
                k = clamp(Int(ceil(retain * length(v))), 0, length(v))
                if k == 0
                    filtered_detail[i] = fill!(similar(v), zero(eltype(v)))
                else
                    idx = partialsortperm(abs.(v), rev=true, 1:k)
                    m = zeros(eltype(v), length(v))
                    m[idx] .= v[idx]
                    filtered_detail[i] = m
                end
                kept += k
            end
            reconstructed = Wave6G.mera_synthesize(coeffs_approx, filtered_detail, wC, vC)
            rec_host = to_host(reconstructed)
            rec_real64 = Float64.(real.(rec_host))
            p = psnr(original, rec_real64)
            rows[j] = Dict(
                :window_id=>window_id,
                :retain=>retain,
                :psnr=>p,
                :kept=>kept,
                :total=>total,
                :seconds=>round(time() - t0, digits=3),
                :L=>L,
                :chi=>cfg["mera-chi"],
                :chimid=>cfg["mera-chimid"],
                :iters1=>cfg["stage1-iters"],
                :iters2=>cfg["stage2-iters"],
                :threads=>Threads.nthreads(),
                :hurst_H=>hurst.H,
                :alpha_min=>multi.alpha_min,
                :alpha_max=>multi.alpha_max,
                :alpha_width=>multi.alpha_width,
                :alpha_peak=>multi.alpha_peak,
                :f_alpha_peak=>multi.f_alpha_peak,
                :analysis_wavelet=>analysis_wavelet,
                :error=>""
            )
        end
        secs = round(time() - t0, digits=3)
        @info "Finished window" id=window_id seconds=secs
        return rows
    catch e
        @error "window error" window_id exception=(e, catch_backtrace())
        return [Dict(:window_id=>window_id, :retain=>missing, :psnr=>missing, :kept=>missing, :total=>missing, :seconds=>missing, :L=>missing, :chi=>missing, :chimid=>missing, :iters1=>missing, :iters2=>missing, :threads=>Threads.nthreads(), :hurst_H=>missing, :alpha_min=>missing, :alpha_max=>missing, :alpha_width=>missing, :alpha_peak=>missing, :f_alpha_peak=>missing, :error=>string(e))]
    end
end

function main()
    args = parse_cli()
    # Parse retains vector if provided
    retains_vec = begin
        rs = get(args, "retains", "")
        if !isempty(rs)
            [parse(Float64, strip(x)) for x in split(rs, ",") if !isempty(strip(x))]
        else
            nothing
        end
    end

    dataset = Wave6G.prepare_window_dataset(args["data"]; window_size=args["window-size"], step=args["step"], normalize=true)
    windows = dataset.windows
    start = max(1, args["start-window"])
    n = args["num-windows"] > 0 ? min(args["num-windows"], length(windows)-start+1) : length(windows)-start+1
    # Prepare output for incremental writing
    mkpath(dirname(args["output"]))
    if isfile(args["output"]) 
        # start fresh to avoid mixing with previous runs
        rm(args["output"]; force=true)
    end
    # Inject retains into args if present
    if retains_vec !== nothing
        args["retains_vec"] = retains_vec
    end

    t0 = time()
    for i in 1:n
        idx = start + i - 1
        rows = run_window_optimized(idx, windows[idx], args)
        # Write all retain rows for this window
        df_rows = DataFrame(rows)
        CSV.write(args["output"], df_rows; append=isfile(args["output"]))
        # Progress/ETA
        elapsed = time() - t0
        done = i
        avg = elapsed / max(done, 1)
        remaining = n - done
        eta = remaining * avg
        @info "Progress" done=done total=n elapsed=round(elapsed, digits=1) eta=format_eta(eta)
    end
    total_secs = round(time() - t0, digits=3)
    @info "Wrote $(n) windows (multi-retain) to $(args["output"]) in $(total_secs)s"

    # Append summary row with wall-clock
    try
        summary_row = DataFrame([Dict(
            :window_id => 0,
            :retain => missing,
            :psnr => missing,
            :kept => missing,
            :total => missing,
            :seconds => total_secs,
            :L => args["mera-L"],
            :chi => args["mera-chi"],
            :chimid => args["mera-chimid"],
            :iters1 => args["stage1-iters"],
            :iters2 => args["stage2-iters"],
            :threads => Threads.nthreads(),
            :hurst_H => missing,
            :alpha_min => missing,
            :alpha_max => missing,
            :alpha_width => missing,
            :alpha_peak => missing,
            :f_alpha_peak => missing,
            :error => ""
        )])
        CSV.write(args["output"], summary_row; append=true)
    catch e
        @warn "Failed to append summary row" error=e
    end
end

main()
