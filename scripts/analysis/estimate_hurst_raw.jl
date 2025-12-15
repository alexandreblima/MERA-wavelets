#!/usr/bin/env julia

ENV["CUDA_VISIBLE_DEVICES"] = ""
ENV["JULIA_CUDA_USE_BINARYBUILDER"] = "false"

using Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))

using ArgParse
using CSV
using DataFrames
using Statistics
using Printf
using Optim

# Optional FFTW: try to load; if missing, we disable Local Whittle
const HAS_FFTW = let ok=false
    try
        @eval using FFTW
        ok = true
    catch
        @warn "FFTW não disponível; Local Whittle será desabilitado"
        ok = false
    end
    ok
end

function parse_cli()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--data"; arg_type=String; required=true; help="CSV com série temporal bruta (primeira coluna numérica será usada)"
        "--window-size"; arg_type=Int; default=4096; help="tamanho da janela para bootstrap do DFA"
        "--step"; arg_type=Int; default=4096; help="passo entre janelas"
        "--num-windows"; arg_type=Int; default=0; help="0 = usar todas possíveis; >0 = limitar"
        "--dfa-order"; arg_type=Int; default=1; help="ordem do polinômio no DFA (1=linear)"
        "--bootstrap"; arg_type=Int; default=500; help="reamostragens bootstrap das janelas para IC do DFA (0 = desabilita)"
        "--mera-L"; arg_type=Int; default=5; help="MERA L parameter for output directory structure"
        "--out"; arg_type=String; default=""; help="caminho de saída do resumo (CSV)"
        "--title"; arg_type=String; default=""; help="rótulo opcional"
        "--with-lw"; action=:store_true; help="habilitar estimativa Local Whittle (espectral); requer FFTW ou DFT ingênuo"
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
    error("Nenhuma coluna numérica encontrada em: " * path)
end

function make_windows(x::Vector{Float64}; window_size::Int, step::Int, num_windows::Int)
    n = length(x)
    idxs = collect(1:step:(n - window_size + 1))
    if num_windows > 0
        idxs = idxs[1:min(num_windows, length(idxs))]
    end
    return [x[i:(i+window_size-1)] for i in idxs]
end

# Simple DFA (order m=1 default). Returns slope alpha.
function dfa_alpha(x::Vector{Float64}; order::Int=1)
    n = length(x)
    y = cumsum(x .- mean(x))
    smin = 16
    smax = max(smin*2, fld(n, 8))
    if smax <= smin
        return NaN
    end
    scales = unique(round.(Int, exp.(range(log(smin), log(smax), length=12))))
    Fs = Float64[]
    Ss = Float64[]
    for s in scales
        m = fld(n, s)
        if m < 4
            continue
        end
        F2 = 0.0
        for k in 0:(m-1)
            seg = @view y[(k*s+1):((k+1)*s)]
            t = collect(1:s)
            if order == 1
                T = [ones(s) t]
            else
                # fallback: polynomial up to order (simple Vandermonde)
                T = hcat([t .^ p for p in 0:order]...)
            end
            coeff = T \ seg
            trend = T * coeff
            res = seg .- trend
            F2 += mean(res .^ 2)
        end
        F = sqrt(F2 / m)
        push!(Fs, F)
        push!(Ss, s)
    end
    if length(Ss) < 4
        return NaN
    end
    lx = log.(Ss); ly = log.(Fs)
    α = cov(lx, ly) / var(lx)
    return α
end

# Local Whittle estimator (semi-paramétrico)
function local_whittle_H_optional(x::Vector{Float64}; enabled::Bool=false)
    if !enabled
        return (NaN, NaN, 0)
    end
    n = length(x)
    x0 = x .- mean(x)
    # Compute first m spectral ordinates either via FFTW or naive DFT
    m = max(10, min(fld(n,10), floor(Int, sqrt(n))))
    λ = [2π*k/n for k in 1:m]
    I0 = similar(λ)
    if HAS_FFTW
        X = FFTW.fft(x0)
        I = abs2.(X)[2:fld(n,2)]
        I0 .= I[1:m]
    else
        # Naive DFT for the first m frequencies
        @warn "Local Whittle: usando DFT ingênuo (sem FFTW), pode ser mais lento"
        for k in 1:m
            ω = 2π * k / n
            csum = 0.0 + 0.0im
            for t in 1:n
                csum += x0[t] * cis(-ω * t)
            end
            I0[k] = abs2(csum)
        end
    end
    function obj(d)
        if d <= -0.49 || d >= 0.49
            return Inf
        end
        w = λ .^ (2d)
        return log(mean(w .* I0)) - 2d*mean(log.(λ))
    end
    res = Optim.optimize(obj, -0.49, 0.49)
    d̂ = Optim.minimizer(res)
    Ĥ = d̂ + 0.5
    se = 0.5 / sqrt(m)
    return (Ĥ, se, m)
end

function bootstrap_ci(vals::Vector{Float64}; B::Int=500, alpha::Float64=0.05)
    n = length(vals)
    if n == 0 || B <= 0
        return (NaN, NaN)
    end
    means = similar(vals, B)
    for b in 1:B
        idx = rand(1:n, n)
        means[b] = mean(view(vals, idx))
    end
    sort!(means)
    lo = means[max(1, floor(Int, (alpha/2)*B))]
    hi = means[min(B, ceil(Int, (1-alpha/2)*B))]
    return (lo, hi)
end

function run()
    args = parse_cli()
    data_path = String(args["data"])
    @assert isfile(data_path) "Arquivo de dados não encontrado: $(data_path)"
    x = load_first_numeric_column(data_path)
    x = Float64.(x)
    x = x[isfinite.(x)]
    x .-= mean(x)

    # DFA por janelas para IC robusto
    ws = Int(args["window-size"])
    st = Int(args["step"])
    nw = Int(args["num-windows"])
    wins = make_windows(x; window_size=ws, step=st, num_windows=nw)
    alphas = Float64[]
    for w in wins
        a = dfa_alpha(w; order=Int(args["dfa-order"]))
        if isfinite(a)
            push!(alphas, a)
        end
    end
    H_dfa_mean = mean(alphas)
    H_dfa_std  = std(alphas)
    H_dfa_ci   = bootstrap_ci(alphas; B=Int(args["bootstrap"]))

    # Local Whittle opcional no traço completo
    with_lw = haskey(args, "with-lw") && args["with-lw"] == true
    H_lw, H_lw_se, m = local_whittle_H_optional(x; enabled=with_lw)
    H_lw_ci = (isfinite(H_lw) && isfinite(H_lw_se)) ? (H_lw - 1.96*H_lw_se, H_lw + 1.96*H_lw_se) : (NaN, NaN)

    # Montar saída
    trace_file = splitpath(data_path) |> last
    trace_base = splitext(trace_file)[1]
    out = String(args["out"])
    if isempty(out)
        L = args["mera-L"]
        out = joinpath("results", trace_base, "L_$(L)", "hurst_raw_summary.csv")
    end
    mkpath(dirname(out))
    df = DataFrame([(
        label=String(get(args, "title", "")),
        N=length(x),
        dfa_windows=length(alphas),
        H_dfa_mean=H_dfa_mean,
        H_dfa_std=H_dfa_std,
        H_dfa_ci_low=H_dfa_ci[1],
        H_dfa_ci_high=H_dfa_ci[2],
        H_lw=H_lw,
        H_lw_se=H_lw_se,
        H_lw_ci_low=H_lw_ci[1],
        H_lw_ci_high=H_lw_ci[2],
        lw_m=m,
        ws=ws,
        step=st
    )])
    CSV.write(out, df)
    @info "H estimado diretamente do traço (DFA e Local Whittle opcional)" out=out H_dfa_mean=round(H_dfa_mean,digits=4) H_dfa_ci=H_dfa_ci H_lw=(isfinite(H_lw) ? round(H_lw,digits=4) : NaN) H_lw_ci=H_lw_ci
end

run()
