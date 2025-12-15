#!/usr/bin/env julia

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using ArgParse
using CSV
using DataFrames
using Statistics

function parse_cli()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--input"; arg_type=String; required=true
        "--output"; arg_type=String; default=""
    end
    return parse_args(s)
end

function summarize(df::DataFrame)
    # Separate data rows and summary row
    data = dropmissing(df, :window_id)
    data = data[data.window_id .> 0, :]

    has_seconds = :seconds in names(data)

    # Per-window time (max seconds per window_id)
    win_time = combine(groupby(data, :window_id), :seconds => maximum => :win_secs)
    avg_win = nrow(win_time) > 0 ? mean(skipmissing(win_time.win_secs)) : missing

    # Per-retain aggregates
    keep_ratio = if (:kept in names(data)) && (:total in names(data))
        data.kept ./ data.total
    else
        fill(missing, nrow(data))
    end
    data[:, :keep_ratio] = keep_ratio

    g = groupby(data, :retain)
    per_retain = combine(g,
        :psnr => mean => :psnr_mean,
        :psnr => std => :psnr_std,
        :keep_ratio => mean => :keep_mean,
        :keep_ratio => std => :keep_std,
        nrow => :rows
    )
    sort!(per_retain, :retain)

    # Pull wall-clock from footer if present
    wall = try
        footer = df[coalesce.(df.window_id, 0) .== 0, :]
        nrow(footer) > 0 && :seconds in names(footer) ? footer.seconds[end] : missing
    catch
        missing
    end

    return per_retain, avg_win, wall
end

function main()
    args = parse_cli()
    f = args["input"]
    @assert isfile(f) "Input file not found: $(f)"
    df = CSV.read(f, DataFrame)
    per_retain, avg_win, wall = summarize(df)

    println("Summary per retain:")
    show(per_retain, allcols=true, allrows=true); println()
    println("Average per-window seconds:", avg_win)
    println("Wall-clock seconds (from footer):", wall)

    if !isempty(args["output"]) 
        CSV.write(args["output"], per_retain)
    end
end

main()
