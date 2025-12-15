#!/usr/bin/env julia

using ArgParse, CSV, DataFrames, Statistics, Printf
using Wave6G
using Random
using Plots

function parse_cli()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--data"; arg_type=String; required=true; help="CSV do trace original"
        "--out"; arg_type=String; default="multifractal_diagnosis.csv"
        "--plot"; arg_type=String; default="multifractal_spectrum.png"
        "--n_boot"; arg_type=Int; default=100; help="número de amostras para bootstrapping"
        "--seed"; arg_type=Int; default=12345
    end
    return parse_args(s)
end

function load_trace(path)
    # Assume CSV com uma coluna de série temporal
    df = CSV.read(path, DataFrame)
    col = names(df)[1]
    return collect(skipmissing(df[!, col]))
end

function multifractal_metrics(signal)
    m = Wave6G.wavelet_multifractal_metrics(signal)
    return m
end

function bootstrap_metrics(signal; n_boot=100, seed=12345)
    Random.seed!(seed)
    N = length(signal)
    metrics = []
    for i in 1:n_boot
        idx = rand(1:N, N)
        s = signal[idx]
        push!(metrics, multifractal_metrics(s))
    end
    return metrics
end

function summarize_boot(metrics)
    # Extrai IC para cada parâmetro
    getv(f) = [getfield(m, f) for m in metrics]
    summary = Dict()
    for f in ("alpha_min", "alpha_max", "alpha_width", "alpha_peak", "f_alpha_peak")
        v = getv(Symbol(f))
        summary[f * "_mean"] = mean(v)
        summary[f * "_lower"] = quantile(v, 0.025)
        summary[f * "_upper"] = quantile(v, 0.975)
    end
    return summary
end

function diagnose_multifractality(alpha_width_mean, alpha_width_lower)
    # Critério simples: multifractal se largura média > 0.1 e IC não inclui zero
    if alpha_width_mean > 0.1 && alpha_width_lower > 0.0
        return "multifractal"
    else
        return "monofractal"
    end
end

function plot_spectrum(signal, outpath)
    m = multifractal_metrics(signal)
    Plots.default(fontfamily="sans", size=(900, 520))
    plt = plot(m.alpha, m.f_alpha, xlabel="α", ylabel="f(α)", title="Espectro multifractal", legend=false)
    savefig(plt, outpath)
end

function run()
    args = parse_cli()
    signal = load_trace(args["data"])
    m = multifractal_metrics(signal)
    metrics_boot = bootstrap_metrics(signal; n_boot=args["n_boot"], seed=args["seed"])
    summary = summarize_boot(metrics_boot)
    diagnosis = diagnose_multifractality(summary["alpha_width_mean"], summary["alpha_width_lower"])
    # Salvar CSV
    df = DataFrame([merge(summary, Dict(
        "diagnosis" => diagnosis,
        "alpha_min" => m.alpha_min,
        "alpha_max" => m.alpha_max,
        "alpha_width" => m.alpha_width,
        "alpha_peak" => m.alpha_peak,
        "f_alpha_peak" => m.f_alpha_peak
    ))])
    CSV.write(args["out"], df)
    # Salvar gráfico
    plot_spectrum(signal, args["plot"])
    println("Diagnóstico: ", diagnosis)
    println("Salvo em: ", args["out"])
    println("Gráfico: ", args["plot"])
end

run()
