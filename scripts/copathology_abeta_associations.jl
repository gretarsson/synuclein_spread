#!/usr/bin/env julia

using PathoSpread
using CSV
using DataFrames
using Statistics
using Plots
using Measures
using HypothesisTests
using StatsBase
using MCMCChains

const APP_PATH = "simulations/syn_app_DIFFGA_RETRO.jls"
const ABETA40_PATH = "data/syn_tau_abeta/ab40_pathology_mpff.csv"
const ABETA42_PATH = "data/syn_tau_abeta/ab42_pathology_mpff.csv"
const OUTPUT_DIR = get(ENV, "COPATHOLOGY_OUTDIR", "figures/copathology_updated/amyloid_beta")
const CSV_DIR = joinpath(OUTPUT_DIR, "csv")
const UPDATE_ALPHA = 0.001
const FILTER_UPDATED = get(ENV, "COPATHOLOGY_FILTER_UPDATED", "true") == "true"
const EXCLUDE_SEED_SITES = get(ENV, "COPATHOLOGY_EXCLUDE_SEEDS", "false") == "true"
const SEED_SITE_BASES = Set(["CA1", "CA3", "DG"])

const FIG_SIZE = (2200, 600)
const LABEL_SIZE = 18
const TICK_SIZE = 13
const TITLE_SIZE = 16
const ANNOT_SIZE = 10


function fdr_bh(pvals::AbstractVector{<:Real})
    N = length(pvals)
    p = Float64.(pvals)
    q = fill(Float64(NaN), N)
    valid = .!isnan.(p)
    any(valid) || return q

    pv = p[valid]
    ord = sortperm(pv)
    p_sorted = pv[ord]
    q_raw = p_sorted .* (length(p_sorted) ./ (1:length(p_sorted)))

    q_sorted = similar(q_raw)
    q_sorted[end] = min(q_raw[end], 1.0)
    for i in (length(q_raw) - 1):-1:1
        q_sorted[i] = min(q_raw[i], q_sorted[i + 1])
    end

    q_valid = similar(q_sorted)
    q_valid[ord] = q_sorted
    q[valid] = q_valid
    return q
end


function setup_defaults()
    PathoSpread.setup_plot_theme!(font="Arial", base=18, lw=3, markersize=10, dpi=300)
    default(
        size = FIG_SIZE,
        guidefontsize = LABEL_SIZE,
        tickfontsize = TICK_SIZE,
        titlefontsize = TITLE_SIZE,
        legendfontsize = 12,
        left_margin = 4mm,
        right_margin = 4mm,
        top_margin = 4mm,
        bottom_margin = 4mm,
    )
end


function region_means_from_wide(path::AbstractString)
    df = CSV.read(path, DataFrame)
    means = Dict{String, Float64}()
    nobs = Dict{String, Int}()
    for c in names(df)[3:end]
        nonmissing = collect(skipmissing(df[!, c]))
        isempty(nonmissing) && continue
        if !(eltype(nonmissing) <: Number)
            continue
        end
        vals = Float64.(nonmissing)
        means[String(c)] = mean(vals)
        nobs[String(c)] = length(vals)
    end
    return means, nobs
end

strip_hemi(region::AbstractString) = (startswith(region, "i") || startswith(region, "c")) ? region[2:end] : String(region)
is_seed_region(region::AbstractString) = strip_hemi(region) in SEED_SITE_BASES


function posterior_samples_from_p(chain::Chains, i::Int)
    for sym in (Symbol("p[$i]"), Symbol("p__$(i)"), Symbol("p_$(i)"), Symbol("p.$(i)"))
        if sym in names(chain)
            return vec(Array(chain[sym]))
        end
    end
    error("Could not find posterior samples for p[$i].")
end


function parameter_names_by_prefix(priors, prefix::AbstractString)
    names_ = filter(n -> startswith(n, prefix * "["), collect(keys(priors)))
    sort!(names_, by = n -> parse(Int, match(Regex("^" * prefix * "\\[(\\d+)\\]"), n).captures[1]))
    return names_
end


function region_update_flags(inference::Dict, prefix::AbstractString; alpha::Float64=UPDATE_ALPHA)
    priors = inference["priors"]
    chain = inference["chain"]
    regions = String.(inference["labels"])
    param_names = parameter_names_by_prefix(priors, prefix)
    prior_keys = collect(keys(priors))
    flags = Dict{String, Int}()
    for (region, pname) in zip(regions, param_names)
        idx = findfirst(==(pname), prior_keys)
        idx === nothing && error("Could not find $pname in priors.")
        samples = posterior_samples_from_p(chain, idx)
        p = pvalue(ApproximateOneSampleKSTest(samples, priors[pname]))
        flags[region] = Int(p < alpha)
    end
    return flags
end


function app_parameter_df()
    inf_app = load_inference(APP_PATH)
    labels = String.(inf_app["labels"])
    beta_app = PathoSpread.posterior_mean_vector_from_priors(inf_app["chain"], inf_app["priors"], "beta", length(labels))
    gamma_app = PathoSpread.posterior_mean_vector_from_priors(inf_app["chain"], inf_app["priors"], "gamma", length(labels))

    df = DataFrame(
        region = labels,
        beta_app = beta_app,
        gamma_app = gamma_app,
        beta_updated = [get(region_update_flags(inf_app, "beta"), r, 0) == 1 for r in labels],
        gamma_updated = [get(region_update_flags(inf_app, "gamma"), r, 0) == 1 for r in labels],
    )
    if EXCLUDE_SEED_SITES
        df = df[.!is_seed_region.(df.region), :]
    end
    return df
end


function join_abeta(df_params::DataFrame)
    ab40_mean, ab40_n = region_means_from_wide(ABETA40_PATH)
    ab42_mean, ab42_n = region_means_from_wide(ABETA42_PATH)

    df = copy(df_params)
    df.ab40 = [get(ab40_mean, r, NaN) for r in df.region]
    df.ab42 = [get(ab42_mean, r, NaN) for r in df.region]
    df.ab40_n = [get(ab40_n, r, 0) for r in df.region]
    df.ab42_n = [get(ab42_n, r, 0) for r in df.region]

    # Ab40 is extremely skewed in raw units, so analyze both peptides on log10(1+x) scale.
    df.ab40_log10p1 = log10.(1 .+ df.ab40)
    df.ab42_log10p1 = log10.(1 .+ df.ab42)

    valid40 = .!isnan.(df.ab40_log10p1)
    valid42 = .!isnan.(df.ab42_log10p1)
    μ40, σ40 = mean(df.ab40_log10p1[valid40]), std(df.ab40_log10p1[valid40])
    μ42, σ42 = mean(df.ab42_log10p1[valid42]), std(df.ab42_log10p1[valid42])
    z40 = fill(NaN, nrow(df))
    z42 = fill(NaN, nrow(df))
    z40[valid40] = (df.ab40_log10p1[valid40] .- μ40) ./ (σ40 + eps())
    z42[valid42] = (df.ab42_log10p1[valid42] .- μ42) ./ (σ42 + eps())
    df.abeta_mean_z = (z40 .+ z42) ./ 2
    df.abeta_mean_log10p1 = (df.ab40_log10p1 .+ df.ab42_log10p1) ./ 2
    return df
end


function linear_fit_xy(x::AbstractVector, y::AbstractVector)
    X = hcat(ones(length(x)), x)
    β = X \ y
    return β[1], β[2]
end


function association_stats(x::AbstractVector, y::AbstractVector)
    pearson_r = cor(x, y)
    pearson_p = pvalue(CorrelationTest(x, y))
    spearman_r = cor(tiedrank(x), tiedrank(y))
    spearman_p = pvalue(CorrelationTest(tiedrank(x), tiedrank(y)))
    return pearson_r, pearson_p, spearman_r, spearman_p
end


function scatter_panel(df::DataFrame, xcol::Symbol, ycol::Symbol; xlabel::AbstractString, ylabel::AbstractString, title::AbstractString)
    mask = .!isnan.(df[!, xcol]) .& .!isnan.(df[!, ycol])
    x = df[mask, xcol]
    y = df[mask, ycol]

    plt = scatter(
        x, y;
        xlabel = xlabel,
        ylabel = ylabel,
        title = title,
        markercolor = :black,
        markeralpha = 0.55,
        markersize = 6,
        markerstrokewidth = 0.0,
        label = false,
    )

    if length(x) >= 3
        a, b = linear_fit_xy(x, y)
        xs = collect(range(minimum(x), maximum(x), length=200))
        plot!(plt, xs, a .+ b .* xs; color=:firebrick3, lw=3, label=false)

        pearson_r, pearson_p, spearman_r, spearman_p = association_stats(x, y)
        ann = "Pearson r = $(round(pearson_r, sigdigits=3)) (p = $(round(pearson_p, sigdigits=3)))\n" *
              "Spearman ρ = $(round(spearman_r, sigdigits=3)) (p = $(round(spearman_p, sigdigits=3)))\n" *
              "n = $(length(x))"
        x0, x1 = extrema(x)
        y0, y1 = extrema(y)
        annotate!(plt, (x0 + 0.04 * (x1 - x0 + eps()), y1 - 0.04 * (y1 - y0 + eps()), text(ann, ANNOT_SIZE, :black, :left, :top)))
    end

    return plt
end


function association_table(df::DataFrame)
    combos = [
        ("log10(1+Ab40)", :ab40_log10p1, "APP syn β", :beta_app, :beta_updated),
        ("log10(1+Ab40)", :ab40_log10p1, "APP syn γ", :gamma_app, :gamma_updated),
        ("log10(1+Ab42)", :ab42_log10p1, "APP syn β", :beta_app, :beta_updated),
        ("log10(1+Ab42)", :ab42_log10p1, "APP syn γ", :gamma_app, :gamma_updated),
        ("Mean z(log Ab40, log Ab42)", :abeta_mean_z, "APP syn β", :beta_app, :beta_updated),
        ("Mean z(log Ab40, log Ab42)", :abeta_mean_z, "APP syn γ", :gamma_app, :gamma_updated),
    ]

    rows = DataFrame(
        abeta_measure = String[],
        parameter = String[],
        n = Int[],
        pearson_r = Float64[],
        pearson_p = Float64[],
        spearman_rho = Float64[],
        spearman_p = Float64[],
    )

    for (ameas, xcol, pname, ycol, updated_col) in combos
        updated_mask = FILTER_UPDATED ? df[!, updated_col] : trues(nrow(df))
        mask = .!isnan.(df[!, xcol]) .& .!isnan.(df[!, ycol]) .& updated_mask
        x = df[mask, xcol]
        y = df[mask, ycol]
        if length(x) < 3
            continue
        end
        pearson_r, pearson_p, spearman_r, spearman_p = association_stats(x, y)
        push!(rows, (ameas, pname, length(x), pearson_r, pearson_p, spearman_r, spearman_p))
    end

    rows.pearson_q_fdr = fdr_bh(rows.pearson_p)
    rows.spearman_q_fdr = fdr_bh(rows.spearman_p)
    sort!(rows, :pearson_p)
    return rows
end


function main()
    mkpath(OUTPUT_DIR)
    mkpath(CSV_DIR)
    setup_defaults()

    df = join_abeta(app_parameter_df())
    tag = FILTER_UPDATED ? "updated" : "all"
    CSV.write(joinpath(CSV_DIR, "regional_abeta_syn_app_parameter_table_$(tag).csv"), df)

    stats_df = association_table(df)
    CSV.write(joinpath(CSV_DIR, "abeta_syn_app_association_stats_$(tag).csv"), stats_df)

    ab40_panel = plot(
        scatter_panel(df[FILTER_UPDATED ? df.beta_updated : trues(nrow(df)), :], :ab40_log10p1, :beta_app; xlabel="log10(1 + Ab40 burden)", ylabel="APP syn rise parameter β", title=FILTER_UPDATED ? "Ab40 vs APP syn β (updated only)" : "Ab40 vs APP syn β"),
        scatter_panel(df[FILTER_UPDATED ? df.gamma_updated : trues(nrow(df)), :], :ab40_log10p1, :gamma_app; xlabel="log10(1 + Ab40 burden)", ylabel="APP syn fall parameter γ", title=FILTER_UPDATED ? "Ab40 vs APP syn γ (updated only)" : "Ab40 vs APP syn γ");
        layout = (1, 2),
        size = (1200, 600),
    )
    savefig(ab40_panel, joinpath(OUTPUT_DIR, "ab40_syn_app_associations_$(tag).pdf"))

    ab42_panel = plot(
        scatter_panel(df[FILTER_UPDATED ? df.beta_updated : trues(nrow(df)), :], :ab42_log10p1, :beta_app; xlabel="log10(1 + Ab42 burden)", ylabel="APP syn rise parameter β", title=FILTER_UPDATED ? "Ab42 vs APP syn β (updated only)" : "Ab42 vs APP syn β"),
        scatter_panel(df[FILTER_UPDATED ? df.gamma_updated : trues(nrow(df)), :], :ab42_log10p1, :gamma_app; xlabel="log10(1 + Ab42 burden)", ylabel="APP syn fall parameter γ", title=FILTER_UPDATED ? "Ab42 vs APP syn γ (updated only)" : "Ab42 vs APP syn γ");
        layout = (1, 2),
        size = (1200, 600),
    )
    savefig(ab42_panel, joinpath(OUTPUT_DIR, "ab42_syn_app_associations_$(tag).pdf"))

    combined_panel = plot(
        scatter_panel(df[FILTER_UPDATED ? df.beta_updated : trues(nrow(df)), :], :abeta_mean_z, :beta_app; xlabel="Mean z-score of log Aβ burden", ylabel="APP syn rise parameter β", title=FILTER_UPDATED ? "Mean Aβ vs APP syn β (updated only)" : "Mean Aβ vs APP syn β"),
        scatter_panel(df[FILTER_UPDATED ? df.gamma_updated : trues(nrow(df)), :], :abeta_mean_z, :gamma_app; xlabel="Mean z-score of log Aβ burden", ylabel="APP syn fall parameter γ", title=FILTER_UPDATED ? "Mean Aβ vs APP syn γ (updated only)" : "Mean Aβ vs APP syn γ");
        layout = (1, 2),
        size = (1200, 600),
    )
    savefig(combined_panel, joinpath(OUTPUT_DIR, "combined_abeta_syn_app_associations_$(tag).pdf"))

    println("Saved figures to $OUTPUT_DIR")
    println("Saved tables to $CSV_DIR")
end


main()
