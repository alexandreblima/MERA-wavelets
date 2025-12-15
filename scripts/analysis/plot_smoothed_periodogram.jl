#!/usr/bin/env julia

ENV["CUDA_VISIBLE_DEVICES"] = ""
ENV["JULIA_CUDA_USE_BINARYBUILDER"] = "false"

using Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))

using ArgParse
using CSV
using DataFrames
using Statistics
const HAS_FFTW = let ok=false
    try
        @eval using FFTW
        ok = true
    catch
        @warn "FFTW not available; falling back to naive DFT (slower)"
        ok = false
    end
    ok
end
using Printf
using Plots

function parse_cli()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--data"; arg_type=String; required=true; help="CSV with raw time series (first numeric column will be used)"
        "--fs"; arg_type=Float64; default=1000.0; help="Sampling rate in Hz (default assumes 1 ms sampling)"
        "--method"; arg_type=String; default="welch"; help="Smoothing: welch|logbin|daniell"
        "--segment-length"; arg_type=Int; default=4096; help="Welch/Daniell: segment length (samples)"
        "--overlap"; arg_type=Float64; default=0.5; help="Welch/Daniell: fractional overlap [0,1)"
        "--window"; arg_type=String; default="hann"; help="Window: hann|hamming|rect"
        "--logbin-bins"; arg_type=Int; default=50; help="Number of log-frequency bins (for method=logbin)"
        "--fit-frac"; arg_type=Float64; default=0.1; help="Fraction of lowest nonzero frequencies to fit slope β in log-log"
        "--detrend"; action=:store_true; help="Detrend (remove mean) before PSD"
        "--title"; arg_type=String; default=""; help="Optional title suffix"
        "--no-title"; action=:store_true; help="Suppress plot title"
        "--out-prefix"; arg_type=String; default=""; help="Output prefix (default: results/<trace_base>/periodogram)"
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

function get_window(name::String, L::Int)
    n = lowercase(name)
    if n == "hann"
        return 0.5 .- 0.5 .* cos.(2π .* (0:L-1) ./ (L-1))
    elseif n == "hamming"
        return 0.54 .- 0.46 .* cos.(2π .* (0:L-1) ./ (L-1))
    elseif n == "rect"
        return ones(L)
    else
        error("Unknown window: " * name)
    end
end

"""
Welch PSD estimate.
Returns (f, Pxx) with f in Hz.
"""
function welch_psd(x::AbstractVector{<:Real}; fs::Real, L::Int, overlap::Real, window::String)
    x = collect(float(x))
    N = length(x)
    x = x .- mean(x)
    step = max(1, Int(round(L * (1 - overlap))))
    w = get_window(window, L)
    U = sum(w.^2)
    segs = Int(fld(N - L, step)) + 1
    # accumulate one-sided PSD length K = floor(L/2)+1
    K = fld(L,2) + 1
    acc = zeros(Float64, K)
    for i in 0:(segs-1)
        s = 1 + i*step
        seg = @view x[s:(s+L-1)]
        xw = seg .* w
        if HAS_FFTW
            X = FFTW.fft(xw)
            P = abs2.(X) ./ (fs * U)
            # one-sided
            acc .+= P[1:K]
        else
            # naive DFT for k=0..K-1
            P = zeros(Float64, K)
            for k in 0:(K-1)
                ω = 2π * k / L
                csum = 0.0 + 0.0im
                @inbounds @simd for n in 0:(L-1)
                    csum += xw[n+1] * cis(-ω * n)
                end
                P[k+1] = abs2(csum) / (fs * U)
            end
            acc .+= P
        end
    end
    Pxx = acc ./ segs
    f = collect(range(0, stop=fs/2, length=K))
    return (collect(f), Pxx)
end

"""
Daniell kernel: simple moving-average smoothing over raw periodogram computed on segments.
"""
function daniell_psd(x::AbstractVector{<:Real}; fs::Real, L::Int, overlap::Real, window::String, span::Int=7)
    f, P = welch_psd(x; fs=fs, L=L, overlap=overlap, window=window)
    k = max(1, span)
    Psm = similar(P)
    n = length(P)
    for i in 1:n
        lo = max(1, i - k)
        hi = min(n, i + k)
        Psm[i] = mean(@view P[lo:hi])
    end
    return (f, Psm)
end

"""
Log-frequency binning with robust median in each bin.
"""
function logbin_psd(x::AbstractVector{<:Real}; fs::Real, L::Int, overlap::Real, window::String, nbins::Int)
    f, P = welch_psd(x; fs=fs, L=L, overlap=overlap, window=window)
    # exclude f=0
    idx = findall(>(0), f)
    f1 = f[idx]
    P1 = P[idx]
    logf = log10.(f1)
    edges = range(minimum(logf), stop=maximum(logf), length=nbins+1)
    fb = Float64[]; Pb = Float64[]
    for j in 1:nbins
        mask = (logf .>= edges[j]) .& (logf .< edges[j+1])
        if any(mask)
            push!(fb, 10^mean(view(logf, findall(mask))))
            push!(Pb, median(view(P1, findall(mask))))
        end
    end
    return (fb, Pb)
end

function fit_slope_loglog(f::AbstractVector{<:Real}, P::AbstractVector{<:Real}; frac::Real=0.1)
    idx = findall(>(0), f)
    f = collect(f[idx]); P = collect(P[idx])
    n = length(f)
    m = max(5, Int(round(frac * n)))
    sel = 1:m
    lx = log10.(f[sel]); ly = log10.(P[sel])
    β = cov(lx, ly) / var(lx)
    # line for plotting anchored at first point
    a = ly[1] - β*lx[1]
    return β, a
end

function main()
    args = parse_cli()
    data_path = String(args["data"])
    @assert isfile(data_path) "Data file not found: $(data_path)"
    x = load_first_numeric_column(data_path)
    x = Float64.(x)
    if get(args, "detrend", false) || true
        x .-= mean(x)
    end
    fs = Float64(args["fs"])
    L = Int(args["segment-length"])
    ov = Float64(args["overlap"])
    win = String(args["window"])
    method = lowercase(String(args["method"]))

    f = Float64[]; P = Float64[]
    if method == "welch"
        f, P = welch_psd(x; fs=fs, L=L, overlap=ov, window=win)
    elseif method == "daniell"
        f, P = daniell_psd(x; fs=fs, L=L, overlap=ov, window=win)
    elseif method == "logbin"
        f, P = logbin_psd(x; fs=fs, L=L, overlap=ov, window=win, nbins=Int(args["logbin-bins"]))
    else
        error("Unknown method=$(method)")
    end

    β, a = fit_slope_loglog(f, P; frac=Float64(args["fit-frac"]))

    trace_file = splitpath(data_path) |> last
    trace_base = splitext(trace_file)[1]
    out_prefix = String(get(args, "out-prefix", ""))
    if isempty(out_prefix)
        out_prefix = joinpath("results", trace_base, "periodogram")
    end
    mkpath(dirname(out_prefix))

    ttl_suffix = String(get(args, "title", ""))
    title_enabled = !get(args, "no-title", false)
    ttl = title_enabled ? (isempty(ttl_suffix) ? "Smoothed periodogram" : "Smoothed periodogram — " * ttl_suffix) : ""

    plt = plot(xaxis=:log10, yaxis=:log10, legend=:topleft, grid=true,
               xlabel="Frequency (Hz)", ylabel="Power spectral density", title=ttl)
    # For log axes, exclude DC bin (f==0) from the main plotted curve
    idx_pos = findall(>(0), f)
    fplot = f[idx_pos]; Pplot = P[idx_pos]
    plot!(plt, fplot, Pplot, label=uppercase(method))
    # Add slope guide line over the first decade
    idx = idx_pos
    f1 = f[idx]
    fline = [f1[1], f1[end]]
    yline = 10 .^ (a .+ (log10.(fline) .* β))
    plot!(plt, fline, yline, label=@sprintf("slope β = %.2f", β), linestyle=:dash, color=:red)

    png_path = out_prefix * ".png"
    pdf_path = out_prefix * ".pdf"
    svg_path = out_prefix * ".svg"
    savefig(plt, png_path)
    try
        savefig(plt, pdf_path)
    catch
    end
    try
        savefig(plt, svg_path)
    catch
    end
    @info "Saved smoothed periodogram" png=png_path pdf=pdf_path svg=svg_path method=method β=β
end

main()
