#!/usr/bin/env julia

using PathoSpread
using CairoMakie
using Statistics
using DataFrames
using CSV
using Printf
using HypothesisTests
using MCMCChains

const MAPT_PATH = "simulations/syn_mapt_DIFFGA_RETRO.jls"
const APP_PATH = "simulations/syn_app_DIFFGA_RETRO.jls"
const OUTPUT_DIR = get(ENV, "COPATHOLOGY_OUTDIR", "figures/copathology_updated_growth")
const TOP_N = 20
const UPDATE_ALPHA = 0.001
const EXCLUDE_SEED_SITES = get(ENV, "COPATHOLOGY_EXCLUDE_SEEDS", "false") == "true"
const SEED_SITE_BASES = Set(["CA1", "CA3", "DG"])


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


function save_figure_pair(fig::Figure, stem::AbstractString)
    pdf_path = joinpath(OUTPUT_DIR, stem * ".pdf")
    png_path = joinpath(OUTPUT_DIR, stem * ".png")
    save(pdf_path, fig)
    save(png_path, fig)
    return (pdf_path=pdf_path, png_path=png_path)
end


strip_hemi(region::AbstractString) = (startswith(region, "i") || startswith(region, "c")) ? region[2:end] : String(region)
is_seed_region(region::AbstractString) = strip_hemi(region) in SEED_SITE_BASES


function posterior_summary(inf_a, inf_b, family::String; label_a::String="MAPT", label_b::String="APP")
    labels_a = String.(inf_a["labels"])
    labels_b = String.(inf_b["labels"])
    labels_a == labels_b || error("Region labels differ between $(label_a) and $(label_b) inferences.")

    n_regions = length(labels_a)
    beta_a = PathoSpread.posterior_mean_vector_from_priors(inf_a["chain"], inf_a["priors"], "beta", n_regions)
    beta_b = PathoSpread.posterior_mean_vector_from_priors(inf_b["chain"], inf_b["priors"], "beta", n_regions)
    mean_a = PathoSpread.posterior_mean_vector_from_priors(inf_a["chain"], inf_a["priors"], family, n_regions)
    mean_b = PathoSpread.posterior_mean_vector_from_priors(inf_b["chain"], inf_b["priors"], family, n_regions)
    sd_a = PathoSpread.posterior_sd_vector_from_priors(inf_a["chain"], inf_a["priors"], family, n_regions)
    sd_b = PathoSpread.posterior_sd_vector_from_priors(inf_b["chain"], inf_b["priors"], family, n_regions)

    delta = mean_b .- mean_a
    pooled_sd = sqrt.(sd_a .^ 2 .+ sd_b .^ 2 .+ eps())
    effect_size = delta ./ pooled_sd

    df = DataFrame(
        region = labels_a,
        family = fill(family, n_regions),
        beta_mapt = beta_a,
        beta_app = beta_b,
        mean_mapt = mean_a,
        sd_mapt = sd_a,
        mean_app = mean_b,
        sd_app = sd_b,
        delta_app_minus_mapt = delta,
        effect_size = effect_size,
        abs_delta = abs.(delta),
        abs_effect_size = abs.(effect_size),
    )

    flags_a = region_update_flags(inf_a, family)
    flags_b = region_update_flags(inf_b, family)
    df.updated_mapt = [get(flags_a, r, 0) == 1 for r in df.region]
    df.updated_app = [get(flags_b, r, 0) == 1 for r in df.region]
    df.updated_both = df.updated_mapt .& df.updated_app
    df.beta_positive_both = (df.beta_mapt .> 0) .& (df.beta_app .> 0)
    df = df[df.updated_both .& df.beta_positive_both, :]
    if EXCLUDE_SEED_SITES
        df = df[.!is_seed_region.(df.region), :]
    end
    return df
end


function add_identity_line!(ax::Axis, x::AbstractVector, y::AbstractVector)
    minval = min(minimum(x), minimum(y))
    maxval = max(maximum(x), maximum(y))
    pad = 0.05 * (maxval - minval + eps())
    limits = (minval - pad, maxval + pad)
    lines!(ax, [limits[1], limits[2]], [limits[1], limits[2]]; color=:gray55, linestyle=:dash, linewidth=2)
    xlims!(ax, limits...)
    ylims!(ax, limits...)
    return nothing
end


function scatter_figure(df::DataFrame, family::String)
    x = df.mean_mapt
    y = df.mean_app
    corr_xy = cor(x, y)

    fig = Figure(size=(1000, 900))
    ax = Axis(fig[1, 1], title="$(uppercase(family)) posterior means: APP vs MAPT (updated + β>0)",
        xlabel="MAPT posterior mean", ylabel="APP posterior mean",
        titlesize=28, xlabelsize=24, ylabelsize=24, xticklabelsize=18, yticklabelsize=18)

    scatter!(ax, x, y; color=(:black, 0.6), markersize=11)
    add_identity_line!(ax, x, y)
    text!(ax, minimum(x) + 0.06 * (maximum(x) - minimum(x) + eps()),
        maximum(y) - 0.05 * (maximum(y) - minimum(y) + eps());
        text=@sprintf("Pearson r = %.3f\nN = %d regions", corr_xy, nrow(df)),
        align=(:left, :top), fontsize=20)
    return fig
end


function lollipop_figure(df::DataFrame, family::String; top_n::Int=TOP_N)
    order = sortperm(df.abs_delta; rev=true)[1:min(top_n, nrow(df))]
    top_df = df[order, :]
    sort_idx = sortperm(top_df.delta_app_minus_mapt)
    top_df = top_df[sort_idx, :]

    values = top_df.delta_app_minus_mapt
    labels = top_df.region
    colors = ifelse.(values .>= 0, :firebrick3, :steelblue3)
    y = 1:length(values)
    max_abs = maximum(abs.(values))

    fig = Figure(size=(1100, 1000))
    ax = Axis(fig[1, 1], title="$(uppercase(family)) largest APP - MAPT shifts (updated + β>0)",
        xlabel="Posterior mean shift (APP - MAPT)", ylabel="",
        titlesize=28, xlabelsize=24, xticklabelsize=18, yticklabelsize=15)

    vlines!(ax, [0.0]; color=:gray60, linestyle=:dash, linewidth=2)
    for i in eachindex(values)
        lines!(ax, [0.0, values[i]], [y[i], y[i]]; color=colors[i], linewidth=6)
        scatter!(ax, [values[i]], [y[i]]; color=colors[i], markersize=17)
    end
    ax.yticks = (y, labels)
    xlims!(ax, -1.1 * max_abs, 1.1 * max_abs)
    return fig
end


function write_summary(df::DataFrame, stem::AbstractString)
    path = joinpath(OUTPUT_DIR, stem * ".csv")
    CSV.write(path, sort(df, :abs_delta, rev=true))
    return path
end


function main()
    mkpath(OUTPUT_DIR)
    PathoSpread.setup_plot_theme!(font="Arial", base=18, lw=3, markersize=12, dpi=300)

    inf_mapt = load_inference(MAPT_PATH)
    inf_app = load_inference(APP_PATH)

    beta_df = posterior_summary(inf_mapt, inf_app, "beta")
    gamma_df = posterior_summary(inf_mapt, inf_app, "gamma")

    beta_csv = write_summary(beta_df, "beta_app_minus_mapt_summary")
    gamma_csv = write_summary(gamma_df, "gamma_app_minus_mapt_summary")

    beta_scatter_paths = save_figure_pair(scatter_figure(beta_df, "beta"), "beta_app_vs_mapt_scatter")
    gamma_scatter_paths = save_figure_pair(scatter_figure(gamma_df, "gamma"), "gamma_app_vs_mapt_scatter")
    beta_shift_paths = save_figure_pair(lollipop_figure(beta_df, "beta"), "beta_top_app_minus_mapt_shifts")
    gamma_shift_paths = save_figure_pair(lollipop_figure(gamma_df, "gamma"), "gamma_top_app_minus_mapt_shifts")

    println("Saved CSV summaries:")
    println(beta_csv)
    println(gamma_csv)
    println()
    println("Saved figures:")
    for path in (
        beta_scatter_paths.pdf_path, beta_scatter_paths.png_path,
        gamma_scatter_paths.pdf_path, gamma_scatter_paths.png_path,
        beta_shift_paths.pdf_path, beta_shift_paths.png_path,
        gamma_shift_paths.pdf_path, gamma_shift_paths.png_path,
    )
        println(path)
    end
end


main()
