#!/usr/bin/env julia

ENV["CUDA_VISIBLE_DEVICES"] = ""
ENV["JULIA_CUDA_USE_BINARYBUILDER"] = "false"

using Pkg
# Activate the repository root project (two levels up from scripts/analysis)
Pkg.activate(joinpath(@__DIR__, "../.."))

using ArgParse, CSV, DataFrames, Statistics, Printf, Wave6G, Wavelets

function parse_cli()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--data"; arg_type=String; default="data/MAWI_bytes_1ms.csv"
        "--window-size"; arg_type=Int; default=4096
        "--step"; arg_type=Int; default=4096
        "--num-windows"; arg_type=Int; default=20
        "--start-window"; arg_type=Int; default=1
        "--retain-min"; arg_type=Float64; default=0.01
        "--retain-max"; arg_type=Float64; default=0.10
        "--retain-steps"; arg_type=Int; default=10
        "--mera-L"; arg_type=Int; default=3
        "--mera-chi"; arg_type=Int; default=2
        "--mera-chimid"; arg_type=Int; default=2
        "--iters1"; arg_type=Int; default=50
        "--lr1"; arg_type=Float64; default=5e-3
        "--iters2"; arg_type=Int; default=50
        "--lr2"; arg_type=Float64; default=2.5e-3
        "--init"; arg_type=String; default="haar"
        "--output"; arg_type=String; default="results/hurst_compare_retains.csv"
    end
    return parse_args(s)
end

psnr(x,y) = begin
    mse = mean(abs2, y .- x)
    max_val = maximum(abs.(x))
    mse < eps() || max_val == 0 ? Inf : 10.0 * log10(max_val^2 / mse)
end

function threshold_levels(detail, retain)
    kept = 0; total = 0
    out = Vector{typeof(detail[1])}(undef, length(detail))
    for i in eachindex(detail)
        v = detail[i]
        total += length(v)
        k = clamp(Int(ceil(retain * length(v))), 0, length(v))
        if k == 0
            out[i] = fill!(similar(v), zero(eltype(v)))
            continue
        end
        idx = partialsortperm(abs.(v), rev=true, 1:k)
        m = zeros(eltype(v), length(v))
        m[idx] .= v[idx]
        kept += k
        out[i] = m
    end
    return out, kept, total
end

function hurst(x::AbstractVector{<:Real})
    return Wave6G.wavelet_hurst_estimate(Float64.(x); min_level=4, max_level=12, model=:fgn).H
end

function multifractal(x::AbstractVector{<:Real})
    m = Wave6G.wavelet_multifractal_metrics(Float64.(x); min_level=2, max_level=12)
    return (; alpha_width=m.alpha_width, alpha_peak=m.alpha_peak, f_alpha_peak=m.f_alpha_peak)
end

function threshold_db_wavelet(x::AbstractVector{<:Real}, w; retain::Float64)
    # DWT coefficients and per-level threshold on detail bands
    N = length(x)
    coeffs = dwt(Float64.(x), w)
    coeffs_thr = copy(coeffs)
    kept = 0; total = 0
    max_levels = Wavelets.maxtransformlevels(N)
    for level in 1:max_levels
        rng = Wavelets.detailrange(N, level)
        if isempty(rng)
            continue
        end
        v = coeffs[rng]
        total += length(v)
        k = clamp(Int(ceil(retain * length(v))), 0, length(v))
        if k == 0
            coeffs_thr[rng] .= 0.0
            continue
        end
        idx_local = partialsortperm(abs.(v), rev=true, 1:k)
        mask = zeros(Bool, length(v))
        mask[idx_local] .= true
        # zero out others
        for (j, val) in enumerate(v)
            coeffs_thr[rng][j] = mask[j] ? val : 0.0
        end
        kept += k
    end
    x_rec = idwt(coeffs_thr, w)
    return Float64.(x_rec), kept, total
end

function main()
    args = parse_cli()
    # Hard-disable CUDA
    try Base.eval(Wave6G, :(const _cuda_available = false)) catch; end

    dataset = Wave6G.prepare_window_dataset(args["data"]; window_size=args["window-size"], step=args["step"], normalize=true)
    windows = dataset.windows
    start = max(1, args["start-window"])
    nwin = min(args["num-windows"], length(windows)-start+1)

    retains = range(args["retain-min"], args["retain-max"], length=args["retain-steps"])

    mkpath(dirname(args["output"]))
    isfile(args["output"]) && rm(args["output"]; force=true)

    # Precompute Haar tensors
    wv_haar = Wave6G._haar_tensors(args["mera-L"])

    # Prepare learned state once per window (to keep comparable per window)
    total_tasks = nwin * length(retains)
    task_idx = 0
    t_global = time()
    for i in 1:nwin
        win_id = start + i - 1
        x = Float64.(windows[win_id])
        H_orig = hurst(x)
        M_orig = multifractal(x)

        # Train learned tensors on this window
        state = Wave6G.prepare_variational_state(x; L=args["mera-L"], chi=args["mera-chi"], chimid=args["mera-chimid"], normalize=true, init=(args["init"]=="haar" ? :haar : :random))
        schedule = [ (numiter=args["iters1"], lr=args["lr1"], reinit=false, init=:previous),
                     (numiter=args["iters2"], lr=args["lr2"], reinit=false, init=:previous) ]
        res = Wave6G.optimize_variational_schedule!(state; schedule=schedule)
        w_learn, v_learn = res.state.wC, res.state.vC

        # Analyze once per method and sweep retains
    ca_h, cd_h = Wave6G.mera_analyze(Float32.(x), wv_haar.wC, wv_haar.vC)
    ca_l, cd_l = Wave6G.mera_analyze(Float32.(x), w_learn, v_learn)
    w_db4 = wavelet(Wavelets.WT.db4)
    w_c3 = wavelet(Wavelets.WT.coif6)          # Coiflet-3 (order 6 in Wavelets.jl)
    w_s8 = wavelet(Wavelets.WT.sym8)           # Symmlet-8 (sym8)
    w_bior44 = Wavelets.WT.wavelet(Wavelets.WT.cdf97, Wavelets.WT.LiftingTransform())  # Biorthogonal 4.4

        for r in retains
            task_idx += 1
            # progress percent
            pct = round(100.0 * task_idx / total_tasks; digits=1)
            @info "Progress" percent=pct window=i retain=r

            # Haar (MERA)
            fd_h, kept_h, total_h = threshold_levels(cd_h, r)
            rec_h = Wave6G.mera_synthesize(ca_h, fd_h, wv_haar.wC, wv_haar.vC)
            xh = Float64.(real.(rec_h))
            H_h = hurst(xh)
            psnr_h = psnr(x, xh)
            M_h = multifractal(xh)

            # Learned
            fd_l, kept_l, total_l = threshold_levels(cd_l, r)
            rec_l = Wave6G.mera_synthesize(ca_l, fd_l, w_learn, v_learn)
            xl = Float64.(real.(rec_l))
            H_l = hurst(xl)
            psnr_l = psnr(x, xl)
            M_l = multifractal(xl)

            # DB4 DWT baseline
            xd, kept_d, total_d = threshold_db_wavelet(x, w_db4; retain=r)
            H_d = hurst(xd)
            psnr_d = psnr(x, xd)
            M_d = multifractal(xd)

            # C3 DWT baseline
            xc3, kept_c3, total_c3 = threshold_db_wavelet(x, w_c3; retain=r)
            H_c3 = hurst(xc3)
            psnr_c3 = psnr(x, xc3)
            M_c3 = multifractal(xc3)

            # S8 DWT baseline
            xs8, kept_s8, total_s8 = threshold_db_wavelet(x, w_s8; retain=r)
            H_s8 = hurst(xs8)
            psnr_s8 = psnr(x, xs8)
            M_s8 = multifractal(xs8)

            # Bior4.4 DWT baseline
            xbior, kept_bior, total_bior = threshold_db_wavelet(x, w_bior44; retain=r)
            H_bior = hurst(xbior)
            psnr_bior = psnr(x, xbior)
            M_bior = multifractal(xbior)

            df = DataFrame([(
                window=win_id,
                retain=r,
                L=args["mera-L"],
                chi=args["mera-chi"],
                chimid=args["mera-chimid"],
                iters1=args["iters1"],
                iters2=args["iters2"],
                init=args["init"],
                H_orig=H_orig,
                H_haar=H_h,
                H_learn=H_l,
                H_db4=H_d,
                H_c3=H_c3,
                H_s8=H_s8,
                H_bior4_4=H_bior,
                deltaH_haar=H_h - H_orig,
                deltaH_learn=H_l - H_orig,
                deltaH_db4=H_d - H_orig,
                deltaH_c3=H_c3 - H_orig,
                deltaH_s8=H_s8 - H_orig,
                deltaH_bior4_4=H_bior - H_orig,
                psnr_haar=psnr_h,
                psnr_learn=psnr_l,
                psnr_db4=psnr_d,
                psnr_c3=psnr_c3,
                psnr_s8=psnr_s8,
                psnr_bior4_4=psnr_bior,
                kept_haar=kept_h,
                total_haar=total_h,
                kept_learn=kept_l,
                total_learn=total_l,
                kept_db4=kept_d,
                total_db4=total_d,
                kept_c3=kept_c3,
                total_c3=total_c3,
                kept_s8=kept_s8,
                total_s8=total_s8,
                kept_bior4_4=kept_bior,
                total_bior4_4=total_bior,
                alpha_width_orig=M_orig.alpha_width,
                alpha_width_haar=M_h.alpha_width,
                alpha_width_learn=M_l.alpha_width,
                alpha_width_db4=M_d.alpha_width,
                alpha_width_c3=M_c3.alpha_width,
                alpha_width_s8=M_s8.alpha_width,
                alpha_width_bior4_4=M_bior.alpha_width,
                delta_alpha_width_haar=M_h.alpha_width - M_orig.alpha_width,
                delta_alpha_width_learn=M_l.alpha_width - M_orig.alpha_width,
                delta_alpha_width_db4=M_d.alpha_width - M_orig.alpha_width,
                delta_alpha_width_c3=M_c3.alpha_width - M_orig.alpha_width,
                delta_alpha_width_s8=M_s8.alpha_width - M_orig.alpha_width,
                delta_alpha_width_bior4_4=M_bior.alpha_width - M_orig.alpha_width,
                f_alpha_peak_orig=M_orig.f_alpha_peak,
                f_alpha_peak_haar=M_h.f_alpha_peak,
                f_alpha_peak_learn=M_l.f_alpha_peak,
                f_alpha_peak_db4=M_d.f_alpha_peak,
                f_alpha_peak_c3=M_c3.f_alpha_peak,
                f_alpha_peak_s8=M_s8.f_alpha_peak,
                f_alpha_peak_bior4_4=M_bior.f_alpha_peak,
            )])
            CSV.write(args["output"], df; append=isfile(args["output"]))
        end
    end
    @info "Done" windows=nwin seconds=round(time()-t_global, digits=3) output=args["output"]
end

main()
