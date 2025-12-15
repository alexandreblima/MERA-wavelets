# ============================================================
# src/MAWI.jl — utilitários de ingestão/particionamento para MAWI
# ============================================================

module MAWI

using DelimitedFiles
using Statistics
using Wavelets

export load_mawi_series, truncate_dyadic, window_series,
       normalize_windows, prepare_window_dataset,
        wavelet_hurst_estimate, wavelet_hurst_estimate_with_ci,
        wavelet_multifractal_metrics

function load_mawi_series(path::AbstractString; limit::Union{Nothing,Int}=nothing)
    data = readdlm(path, ',', Float64)
    values = vec(data)
    if limit === nothing
        return values
    end
    return values[1:limit]
end

function truncate_dyadic(values::AbstractVector{<:Number})
    N = length(values)
    L = floor(Int, log2(N))
    M = Int(1) << L
    truncated = Float64.(values[1:M])
    removed = N - M
    return truncated, M, removed
end

function window_series(values::AbstractVector{<:Number}; window_size::Int, step::Int=window_size)
    N = length(values)
    @assert window_size ≤ N "window_size must not exceed series length"
    @assert step ≥ 1 "step must be positive"
    windows = Vector{Vector{Float64}}()
    for start in 1:step:(N - window_size + 1)
        push!(windows, Float64.(values[start:(start + window_size - 1)]))
    end
    return windows
end

function normalize_windows(windows::AbstractVector{<:AbstractVector{<:Number}})
    normalized = Vector{Vector{Float32}}(undef, length(windows))
    stats = Vector{Tuple{Float64, Float64}}(undef, length(windows))
    for (i, window) in enumerate(windows)
        w = Float64.(window)
        μ = mean(w)
        σ = std(w)
        σ = σ < eps() ? 1.0 : σ
        normalized[i] = Float32.((w .- μ) ./ σ)
        stats[i] = (μ, σ)
    end
    return normalized, stats
end

function prepare_window_dataset(path::AbstractString;
        limit::Union{Nothing,Int}=nothing,
        window_size::Int,
        step::Int=window_size,
        normalize::Bool=true)

    series = load_mawi_series(path; limit=limit)
    truncated, length_pow2, removed = truncate_dyadic(series)
    windows = window_series(truncated; window_size=window_size, step=step)
    norm_windows = windows
    stats = nothing
    if normalize
        norm_windows, stats = normalize_windows(windows)
    end
    return (; windows=norm_windows, stats, truncated, length_pow2, removed)
end

const _WAVELET_ALIAS_RESOLVERS = Dict{String, Function}(
    "haar" => () -> wavelet(Wavelets.WT.haar),
    "db1" => () -> wavelet(Wavelets.WT.haar),
    "daubechies1" => () -> wavelet(Wavelets.WT.haar),
    "db4" => () -> wavelet(Wavelets.WT.db4),
    "daubechies4" => () -> wavelet(Wavelets.WT.db4),
    "c3" => () -> wavelet(Wavelets.WT.coif6),
    "coif3" => () -> wavelet(Wavelets.WT.coif6),
    "coiflet3" => () -> wavelet(Wavelets.WT.coif6),
    "coif6" => () -> wavelet(Wavelets.WT.coif6),
    "s8" => () -> wavelet(Wavelets.WT.sym8),
    "sym8" => () -> wavelet(Wavelets.WT.sym8),
    "symmlet8" => () -> wavelet(Wavelets.WT.sym8),
    "symlet8" => () -> wavelet(Wavelets.WT.sym8),
    "bior44" => () -> Wavelets.WT.wavelet(Wavelets.WT.cdf97, Wavelets.WT.LiftingTransform()),
    "bior4.4" => () -> Wavelets.WT.wavelet(Wavelets.WT.cdf97, Wavelets.WT.LiftingTransform()),
    "biorthogonal44" => () -> Wavelets.WT.wavelet(Wavelets.WT.cdf97, Wavelets.WT.LiftingTransform()),
    "biorthogonal4.4" => () -> Wavelets.WT.wavelet(Wavelets.WT.cdf97, Wavelets.WT.LiftingTransform()),
    "cdf97" => () -> Wavelets.WT.wavelet(Wavelets.WT.cdf97, Wavelets.WT.LiftingTransform())
)

function _normalize_wavelet_name(name::AbstractString)
    s = lowercase(strip(name))
    s = replace(s, ' ' => "")
    s = replace(s, '-' => "")
    s = replace(s, '.' => "")
    return s
end

function _resolve_wavelet(w)
    if w isa Wavelets.WT.FilterWavelet || w isa Wavelets.WT.LSWavelet
        return w
    elseif w isa Wavelets.WT.OrthoWaveletClass || w isa Wavelets.WT.BiOrthoWaveletClass
        return wavelet(w)
    elseif w isa Symbol
        return _resolve_wavelet(String(w))
    elseif w isa AbstractString
        key = _normalize_wavelet_name(w)
        resolver = get(_WAVELET_ALIAS_RESOLVERS, key, nothing)
        resolver === nothing && error("Unsupported wavelet specification: $w (normalized: $key)")
        return resolver()
    else
        error("Unsupported wavelet specification: $w")
    end
end

function _ols_slope(xs::AbstractVector{<:Real}, ys::AbstractVector{<:Real})
    @assert length(xs) == length(ys) && !isempty(xs)
    x̄ = mean(xs)
    ȳ = mean(ys)
    num = sum((xs .- x̄) .* (ys .- ȳ))
    den = sum((xs .- x̄) .^ 2)
    den ≈ 0 && error("Degenerate regression slope: zero variance in scales")
    return num / den
end

function wavelet_hurst_estimate(values::AbstractVector{<:Real};
        wavelet=Wavelets.wavelet(Wavelets.WT.db4),
        min_level::Int=2,
        max_level::Union{Nothing,Int}=nothing,
        model::Symbol=:fgn)

    N = length(values)
    @assert N > 0 "Input series cannot be empty"

    w = _resolve_wavelet(wavelet)
    max_levels = Wavelets.maxtransformlevels(N)
    level_hi = max_level === nothing ? max_levels : min(max_level, max_levels)
    level_hi = max(level_hi, min_level)

    coeffs = dwt(Float64.(values), w)

    levels = collect(min_level:level_hi)
    energies = Float64[]
    valid_levels = Int[]

    for level in levels
        range = Wavelets.detailrange(N, level)
        band = coeffs[range]
        energy = mean(abs2, band)
        if isfinite(energy) && energy > 0
            push!(energies, energy)
            push!(valid_levels, level)
        end
    end

    @assert !isempty(energies) "No valid detail energies to estimate H"

    slope = _ols_slope(Float64.(valid_levels), log2.(energies))
    H = model === :fbm ? (slope - 1.0) / 2 : (slope + 1.0) / 2
    return (; H, slope, levels=valid_levels, energies)
end

const _GAUSSIAN_Z95 = 1.959963984540054

function _hurst_gaussian_stats(levels::AbstractVector, energies::AbstractVector, slope::Real, H::Real; alpha::Float64=0.05)
    n = length(levels)
    if n < 3
        return (; intercept=NaN, se_slope=NaN, se_H=NaN,
                 residual_variance=NaN, residual_std=NaN, dof=n - 2,
                 ci_low=NaN, ci_high=NaN, num_levels=n)
    end
    levels_f = Float64.(levels)
    energies_f = Float64.(energies)
    y = log2.(energies_f)
    x̄ = mean(levels_f)
    ȳ = mean(y)
    intercept = ȳ - slope * x̄
    residuals = y .- (intercept .+ slope .* levels_f)
    dof = n - 2
    sse = sum(residuals .^ 2)
    σ² = dof > 0 ? sse / dof : NaN
    sxx = sum((levels_f .- x̄) .^ 2)
    if !(dof > 0 && sxx > 0 && isfinite(σ²))
        return (; intercept=intercept, se_slope=NaN, se_H=NaN,
                 residual_variance=σ², residual_std=isfinite(σ²) ? sqrt(σ²) : NaN, dof=dof,
                 ci_low=NaN, ci_high=NaN, num_levels=n)
    end
    se_slope = sqrt(σ² / sxx)
    se_H = se_slope / 2
    z = _GAUSSIAN_Z95
    delta = z * se_H
    ci_low = clamp(H - delta, 0.0, 1.0)
    ci_high = clamp(H + delta, 0.0, 1.0)
    return (; intercept=intercept, se_slope=se_slope, se_H=se_H,
             residual_variance=σ², residual_std=sqrt(σ²), dof=dof,
             ci_low=ci_low, ci_high=ci_high, num_levels=n)
end

function wavelet_hurst_estimate_with_ci(values::AbstractVector{<:Real};
        wavelet=Wavelets.wavelet(Wavelets.WT.db4),
        min_level::Int=2,
        max_level::Union{Nothing,Int}=nothing,
        model::Symbol=:fgn,
        alpha::Float64=0.05)
    res = wavelet_hurst_estimate(values; wavelet=wavelet, min_level=min_level, max_level=max_level, model=model)
    stats = _hurst_gaussian_stats(res.levels, res.energies, res.slope, res.H; alpha=alpha)
    return (; res..., stats...)
end

function wavelet_multifractal_metrics(
    values::AbstractVector{<:Real};
    wavelet=Wavelets.wavelet(Wavelets.WT.db4),
    min_level::Int=2,
    max_level::Union{Nothing,Int}=nothing,
    q_values::AbstractVector{<:Real}=collect(-5:5),
    eps_val::Real=1e-12
)
    N = length(values)
    N == 0 && error("Input series cannot be empty")

    w = _resolve_wavelet(wavelet)
    max_levels = Wavelets.maxtransformlevels(N)
    level_hi = max_level === nothing ? max_levels : min(max_level, max_levels)
    level_hi = max(level_hi, min_level)
    levels = collect(min_level:level_hi)
    isempty(levels) && return (; alpha_min=NaN, alpha_max=NaN, alpha_width=NaN, alpha_peak=NaN, f_alpha_peak=NaN)

    coeffs = dwt(Float64.(values), w)
    abs_details = [abs.(coeffs[Wavelets.detailrange(N, level)]) for level in levels]

    q_list = collect(q_values)
    isempty(q_list) && return (; alpha_min=NaN, alpha_max=NaN, alpha_width=NaN, alpha_peak=NaN, f_alpha_peak=NaN)

    zeta_vals = fill(NaN, length(q_list))

    for (idx_q, q) in enumerate(q_list)
        log_stats = Float64[]
        level_subset = Float64[]
        for (jdx, detail) in enumerate(abs_details)
            isempty(detail) && continue
            detail_shift = detail .+ eps_val
            s_q = if q == 0
                exp(mean(log.(detail_shift)))
            else
                mean(detail_shift .^ q)
            end
            if isfinite(s_q) && s_q > 0
                push!(log_stats, log2(s_q))
                push!(level_subset, levels[jdx])
            end
        end
        if length(level_subset) ≥ 2
            slope = _ols_slope(level_subset, log_stats)
            zeta_vals[idx_q] = -slope
        end
    end

    valid_idx = findall(!isnan, zeta_vals)
    length(valid_idx) ≥ 3 || return (; alpha_min=NaN, alpha_max=NaN, alpha_width=NaN, alpha_peak=NaN, f_alpha_peak=NaN)

    qs = Float64.(q_list[valid_idx])
    zetas = zeta_vals[valid_idx]
    alpha = similar(zetas)

    for i in eachindex(qs)
        if 1 < i < length(qs)
            dq = qs[i + 1] - qs[i - 1]
            dq ≈ 0 && (alpha[i] = NaN; continue)
            alpha[i] = (zetas[i + 1] - zetas[i - 1]) / dq
        elseif i == 1
            dq = qs[i + 1] - qs[i]
            dq ≈ 0 && (alpha[i] = NaN; continue)
            alpha[i] = (zetas[i + 1] - zetas[i]) / dq
        else
            dq = qs[i] - qs[i - 1]
            dq ≈ 0 && (alpha[i] = NaN; continue)
            alpha[i] = (zetas[i] - zetas[i - 1]) / dq
        end
    end

    good = findall(!isnan, alpha)
    isempty(good) && return (; alpha_min=NaN, alpha_max=NaN, alpha_width=NaN, alpha_peak=NaN, f_alpha_peak=NaN)

    qs = qs[good]
    zetas = zetas[good]
    alpha = alpha[good]
    f_alpha = qs .* alpha .- zetas .+ 1

    idx_peak = argmax(f_alpha)
    return (
        alpha_min = minimum(alpha),
        alpha_max = maximum(alpha),
        alpha_width = maximum(alpha) - minimum(alpha),
        alpha_peak = alpha[idx_peak],
        f_alpha_peak = f_alpha[idx_peak]
    )
end

end # module
