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

function parse_cli()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--mera"; arg_type=String; required=true; help="CSV de saída do MERA (com coluna hurst_h)"
        "--L"; arg_type=Int; default=0; help="Filtrar por nível L (0 = sem filtro)"
        "--bootstrap"; arg_type=Int; default=1000; help="Número de reamostragens bootstrap (0 = desabilita)"
        "--out"; arg_type=String; default=""; help="Saída (CSV); default: ao lado do arquivo MERA"
        "--title"; arg_type=String; default=""; help="Rótulo opcional para o resumo"
    end
    return parse_args(s)
end

function maybe_col(df::DataFrame, names_try::Vector{Symbol})
    wanted = Set(lowercase.(String.(names_try)))
    for n in names(df)
        if lowercase(strip(String(n))) in wanted
            return n
        end
    end
    return nothing
end

function bootstrap_ci_mean(x::Vector{Float64}; B::Int=1000, alpha::Float64=0.05)
    n = length(x)
    if n == 0 || B <= 0
        return (NaN, NaN)
    end
    means = Vector{Float64}(undef, B)
    for b in 1:B
        idx = rand(1:n, n)
        means[b] = mean(view(x, idx))
    end
    sort!(means)
    lo_idx = max(1, floor(Int, (alpha/2) * B))
    hi_idx = min(B, ceil(Int, (1 - alpha/2) * B))
    return (means[lo_idx], means[hi_idx])
end

function run()
    args = parse_cli()
    path = String(args["mera"])
    @assert isfile(path) "MERA CSV não encontrado: $(path)"
    df = CSV.read(path, DataFrame)

    c_wid = maybe_col(df, [:window_id])
    if c_wid !== nothing
        df = df[coalesce.(df[!, c_wid], 0) .> 0, :]
    end

    # Filtrar por L, se solicitado e existir
    Lwanted = Int(args["L"])
    if Lwanted > 0
        c_L = maybe_col(df, [:L, :l])
        @assert c_L !== nothing "Coluna L não encontrada para filtrar"
        df = df[coalesce.(Int.(round.(Float64.(df[!, c_L]))), 0) .== Lwanted, :]
    end

    c_h = maybe_col(df, [:hurst_h, :hurst, :H])
    @assert c_h !== nothing "Coluna de Hurst (hurst_h) não encontrada no CSV"
    h = skipmissing(Float64.(df[!, c_h])) |> collect
    n = length(h)
    @assert n > 0 "Sem amostras de H para calcular o resumo"

    h_mean = mean(h)
    h_std  = std(h)
    se = h_std / sqrt(n)
    # IC normal aproximado (referência)
    z = 1.96
    ci_norm = (h_mean - z*se, h_mean + z*se)
    # IC bootstrap (padrão)
    B = Int(args["bootstrap"])
    ci_boot = bootstrap_ci_mean(h; B=B, alpha=0.05)

    # Preparar saída
    summ = DataFrame([
        (; label=String(get(args, "title", "")), n=n, h_mean=h_mean, h_std=h_std,
           se=se, ci_norm_low=ci_norm[1], ci_norm_high=ci_norm[2],
           ci_boot_low=ci_boot[1], ci_boot_high=ci_boot[2], L=(Lwanted>0 ? Lwanted : missing))
    ])

    out = String(args["out"])
    if isempty(out)
        basedir = dirname(path)
        suffix = Lwanted > 0 ? @sprintf("_L%d", Lwanted) : ""
        out = joinpath(basedir, "hurst_summary" * suffix * ".csv")
    end
    mkpath(dirname(out))
    CSV.write(out, summ)

    @info "Hurst summary" n=n mean=round(h_mean,digits=5) std=round(h_std,digits=5) se=round(se,digits=5) ci_norm=ci_norm ci_boot=ci_boot out=out
end

run()
