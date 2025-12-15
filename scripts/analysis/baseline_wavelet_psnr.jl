#!/usr/bin/env julia

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
using Wave6G
using Wavelets

function parse_cli()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--data"; arg_type=String; default="data/MAWI_bytes_1ms.csv"
        "--window-size"; arg_type=Int; default=4096
        "--step"; arg_type=Int; default=4096
        "--start-window"; arg_type=Int; default=1
        "--num-windows"; arg_type=Int; default=100
        "--retains"; arg_type=String; default="0.01,0.02,0.05,0.1,0.2,0.4,0.8"
        "--waves"; arg_type=String; default="haar,db4,c3,s8,bior4.4"; help="comma-separated wavelets (aliases supported): haar, db4, c3|coif3|coiflet3, s8|sym8|symmlet8, bior4.4|bior44|biorthogonal4.4|cdf97"
        "--mera-L"; arg_type=Int; default=5; help="MERA L parameter for output directory structure"
        "--output"; arg_type=String; default=""
    end
    return parse_args(s)
end

psnr(original, reconstructed) = begin
    mse = mean(abs2, reconstructed .- original)
    max_val = maximum(abs.(original))
    mse < eps() || max_val == 0 ? Inf : 10.0 * log10(max_val^2 / mse)
end

function get_wavelet(name::AbstractString)
    # Normalize: lowercase, remove spaces, hyphens, and dots for robust aliasing
    s = lowercase(strip(name))
    s = replace(s, ' ' => "")
    s = replace(s, '-' => "")
    s = replace(s, '.' => "")

    # Common aliases:
    # - Haar
    if s == "haar" || s == "db1"
        return wavelet(WT.haar)

    # - Daubechies-4
    elseif s == "db4" || s == "daubechies4"
        return wavelet(WT.db4)

    # - Coiflet-3 (C3). Wavelets.jl exposes coif2, coif4, coif6, coif8 where coif6 corresponds to Coiflet-3 order
    elseif s == "c3" || s == "coif3" || s == "coiflet3"
        return wavelet(WT.coif6)

    # - Symmlet-8 (aka Symlet-8). Wavelets.jl uses sym8
    elseif s == "s8" || s == "sym8" || s == "symmlet8" || s == "symlet8"
        return wavelet(WT.sym8)

    # - Biorthogonal 4.4 (bior4.4). In Wavelets.jl we can use CDF 9/7 (cdf97) via lifting.
    #   Many papers refer to BiOrthogonal 4.4; CDF 9/7 is the standard biorthogonal used in JPEG2000.
    elseif s == "bior44" || s == "biorthogonal44" || s == "biorthogonal44" || s == "cdf97" || s == "bior4.4"
        return Wavelets.WT.wavelet(Wavelets.WT.cdf97, Wavelets.WT.LiftingTransform())

    else
        error("Unsupported wavelet: $name (normalized: $s)")
    end
end

function threshold_per_level!(coeffs::Vector{Float64}, N::Int, retain::Float64, w)
    # Leave approximation band (level = max) unchanged; threshold details at all levels
    max_levels = Wavelets.maxtransformlevels(N)
    total = 0
    kept = 0
    for level in 1:max_levels
        rng = Wavelets.detailrange(N, level)
        total += length(rng)
        len = length(rng)
        k = clamp(Int(ceil(retain * len)), 0, len)
        if k == 0
            coeffs[rng] .= 0.0
            continue
        end
        v = coeffs[rng]
        idx = partialsortperm(abs.(v), rev=true, 1:k)
        mask = zeros(Bool, len)
        mask[idx] .= true
        for (j, on) in enumerate(mask)
            coeffs[rng[j]] = on ? v[j] : 0.0
        end
        kept += k
    end
    return kept, total
end

function run()
    args = parse_cli()
    retains = [parse(Float64, strip(x)) for x in split(args["retains"], ",") if !isempty(strip(x))]
    waves = [strip(x) for x in split(args["waves"], ",") if !isempty(strip(x))]

    ds = Wave6G.prepare_window_dataset(args["data"]; window_size=args["window-size"], step=args["step"], normalize=true)
    windows = ds.windows
    start = max(1, args["start-window"])
    n = args["num-windows"] > 0 ? min(args["num-windows"], length(windows)-start+1) : length(windows)-start+1

    # Default output under results/<trace_base>/L_<L>/wavelet_baseline_psnr.csv
    out = String(args["output"])
    if isempty(out)
        trace_file = splitpath(String(args["data"])) |> last
        trace_base = splitext(trace_file)[1]
        L = args["mera-L"]
        out = joinpath("results", trace_base, "L_$(L)", "wavelet_baseline_psnr.csv")
    end
    mkpath(dirname(out))
    isfile(out) && rm(out; force=true)

    t0 = time()
    done = 0
    for i in 1:n
        idx = start + i - 1
        x = Float64.(windows[idx])
        N = length(x)
        for wave_name in waves
            w = get_wavelet(wave_name)
            coeffs = dwt(x, w)
            for retain in retains
                c = copy(coeffs)
                t1 = time()
                kept, total = threshold_per_level!(c, N, retain, w)
                xr = idwt(c, w)
                p = psnr(x, xr)
                secs = round(time() - t1, digits=3)
                row = DataFrame([Dict(
                    :window_id => idx,
                    :wavelet => wave_name,
                    :retain => retain,
                    :psnr => p,
                    :kept => kept,
                    :total => total,
                    :seconds => secs
                )])
                CSV.write(out, row; append=isfile(out))
            end
        end
        done += 1
        elapsed = time() - t0
        avg = elapsed / done
        rem = n - done
        eta = rem * avg
        @info "Progress (wavelet baseline)" done=done total=n elapsed=round(elapsed,digits=1) eta=@sprintf("%02d:%02d:%02d", floor(Int,eta/3600), floor(Int,(eta%3600)/60), floor(Int,eta%60))
    end
    total_secs = round(time() - t0, digits=3)
    CSV.write(out, DataFrame([Dict(:window_id=>0, :wavelet=>"", :retain=>missing, :psnr=>missing, :kept=>missing, :total=>missing, :seconds=>total_secs)]); append=true)
end

run()
