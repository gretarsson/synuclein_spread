#!/usr/bin/env julia

using Serialization
using Distributions
using OrderedCollections
using MCMCChains
using PathoSpread
using Statistics
using HypothesisTests
using Plots
using StatsPlots
using KernelDensity
using StatsBase
using Measures

# ============================================================
# USER SETTINGS
# ============================================================

# Single inference file to analyze
inference_file = "simulations/DIFFGA_RETRO.jls"

# Name used for outputs
model_name = "DIFFGA"

# Which parameter prefixes to process
prefixes_to_process = ["beta", "gamma"]

save_dir = "figures/posteriors_mean/DIFFGA_RETRO"
csv_dir  = save_dir
plot_dir = joinpath(save_dir, "update_checks")
scatter_dir = joinpath(save_dir, "parameter_scatter")
mkpath(csv_dir)
mkpath(plot_dir)
mkpath(scatter_dir)

# Threshold for calling a parameter "updated"
const ALPHA = 0.001

# Plot settings
const NGRID = 500
const PLOT_FORMAT = "pdf"   # "pdf" or "png"

# If true, only save plots for non-updated or borderline parameters
const SAVE_ONLY_FLAGGED_PLOTS = false

# Borderline window if SAVE_ONLY_FLAGGED_PLOTS = true
const BORDERLINE_LOW  = 0.5 * ALPHA
const BORDERLINE_HIGH = 2.0 * ALPHA

# Scatter settings
const SCATTER_MARKERSIZE = 6
const SCATTER_ALPHA = 0.8

# ============================================================
# HELPERS
# ============================================================

"""
Load posterior samples for p[i], assuming the chain stores parameters as Symbol("p[i]").
Returns a Vector{Float64} or nothing if missing.
"""
function posterior_samples_from_p(chain::Chains, i::Int)
    var = Symbol("p[$i]")
    if !(var in names(chain))
        return nothing
    end
    vals = chain[var]
    return vec(Array(vals))
end

"""
Return parameter names with the form prefix[1], prefix[2], ..., sorted by index.
"""
function parameter_names_by_prefix(priors, prefix::AbstractString)
    names_ = filter(n -> startswith(n, prefix * "["), collect(keys(priors)))
    sort!(names_, by = n -> parse(Int, match(Regex("^" * prefix * "\\[(\\d+)\\]"), n).captures[1]))
    return names_
end

"""
One-sample KS test:
    H0: posterior samples are distributed as the prior
Returns:
    pvalue::Float64
    updated::Int
"""
function posterior_vs_prior_update_flag(samples::AbstractVector, prior_dist; alpha::Float64=ALPHA)
    test = ApproximateOneSampleKSTest(samples, prior_dist)
    p = pvalue(test)
    updated = Int(p < alpha)
    return p, updated
end

"""
Try to get a reasonable plotting range for the prior/posterior overlay.
"""
function plotting_range(samples::AbstractVector, prior_dist)
    smin, smax = minimum(samples), maximum(samples)
    ss = std(samples)

    lo = smin - 0.5 * ss
    hi = smax + 0.5 * ss

    try
        qlo = quantile(prior_dist, 0.001)
        qhi = quantile(prior_dist, 0.999)
        lo = min(lo, qlo)
        hi = max(hi, qhi)
    catch
        try
            pm = mean(prior_dist)
            ps = std(prior_dist)
            lo = min(lo, pm - 5ps)
            hi = max(hi, pm + 5ps)
        catch
        end
    end

    if !isfinite(lo) || !isfinite(hi) || lo == hi
        pad = max(ss, 1e-3)
        lo = smin - pad
        hi = smax + pad
    end

    return lo, hi
end

"""
Make a filesystem-safe name from a region label.
"""
function safe_filename(s::AbstractString)
    s = replace(s, r"[ /\\\(\)\[\],:;]+" => "_")
    s = replace(s, r"_+" => "_")
    return strip(s, '_')
end

"""
Decide whether to save a plot for this parameter.
"""
function should_save_plot(updated::Int, ks_p::Float64)
    if !SAVE_ONLY_FLAGGED_PLOTS
        return true
    end
    return (updated == 0) || (BORDERLINE_LOW <= ks_p <= BORDERLINE_HIGH)
end

"""
Plot prior density and posterior KDE for one parameter.
"""
function save_prior_posterior_plot(outfile, samples, prior_dist, region, pname, ks_p, updated)
    lo, hi = plotting_range(samples, prior_dist)
    x = range(lo, hi; length=NGRID)

    prior_y = [pdf(prior_dist, xi) for xi in x]
    kd = kde(samples)

    ttl = "$(pname) | $(region)\nupdated=$(updated), KS p=$(round(ks_p, sigdigits=4))"

    plt = plot(
        x, prior_y;
        lw=3,
        ls=:dash,
        color=:black,
        label="prior",
        xlabel=pname,
        ylabel="Density",
        title=ttl,
        legend=:topright
    )

    plot!(plt, kd.x, kd.density; lw=3, color=:blue, label="posterior")

    savefig(plt, outfile)
end

"""
Write a vector of NamedTuples to CSV.
"""
function write_namedtuple_csv(outfile::AbstractString, rows)
    isempty(rows) && error("No rows to write for $outfile")
    headers = collect(keys(rows[1]))

    open(outfile, "w") do io
        println(io, join(string.(headers), ","))
        for row in rows
            vals = [getproperty(row, h) for h in headers]
            println(io, join(string.(vals), ","))
        end
    end
end

"""
Extract posterior summary rows for all parameters with a given prefix, match them to regions,
and optionally save prior/posterior comparison plots.

Output columns:
    region
    mean_post
    sd_post
    mean_prior
    sd_prior
    ks_pvalue
    updated
"""
function extract_parameter_table_and_plots(
    inf::AbstractDict,
    prefix::AbstractString,
    model::AbstractString;
    alpha::Float64=ALPHA,
    base_plot_dir::AbstractString=plot_dir
)
    priors = inf["priors"]
    chain  = inf["chain"]
    regions = String.(inf["labels"])

    param_names = parameter_names_by_prefix(priors, prefix)
    prior_keys = collect(keys(priors))

    if length(param_names) != length(regions)
        error("Length mismatch for $model/$prefix: $(length(param_names)) parameters vs $(length(regions)) labels.")
    end

    this_plot_dir = joinpath(base_plot_dir, model, prefix)
    mkpath(this_plot_dir)

    rows = Vector{NamedTuple}()

    for (region, pname) in zip(regions, param_names)
        idx = findfirst(==(pname), prior_keys)
        idx === nothing && error("Could not find $pname in prior keys.")

        samples = posterior_samples_from_p(chain, idx)
        samples === nothing && error("Missing posterior samples for $pname at p[$idx].")

        prior_dist = priors[pname]
        ks_p, updated = posterior_vs_prior_update_flag(samples, prior_dist; alpha=alpha)

        push!(rows, (
            region     = region,
            mean_post  = mean(samples),
            sd_post    = std(samples),
            mean_prior = mean(prior_dist),
            sd_prior   = std(prior_dist),
            ks_pvalue  = ks_p,
            updated    = updated
        ))

        if should_save_plot(updated, ks_p)
            outfile = joinpath(
                this_plot_dir,
                "$(safe_filename(region))_$(prefix).$(PLOT_FORMAT)"
            )
            save_prior_posterior_plot(outfile, samples, prior_dist, region, pname, ks_p, updated)
        end
    end

    return rows
end

"""
Compute summary statistics from rows.
"""
function summarize_rows(rows, model::AbstractString, prefix::AbstractString; alpha::Float64=ALPHA)
    n = length(rows)
    n == 0 && error("No rows to summarize for $model/$prefix")

    ks_pvalues = [r.ks_pvalue for r in rows]
    updated    = [r.updated for r in rows]
    mean_post  = [r.mean_post for r in rows]
    sd_post    = [r.sd_post for r in rows]
    mean_prior = [r.mean_prior for r in rows]
    sd_prior   = [r.sd_prior for r in rows]

    n_updated = sum(updated)

    return (
        model                  = model,
        parameter              = prefix,
        alpha                  = alpha,
        n_regions              = n,
        n_updated              = n_updated,
        frac_updated           = n_updated / n,

        n_p_lt_0_05            = sum(ks_pvalues .< 0.05),
        n_p_lt_0_01            = sum(ks_pvalues .< 0.01),
        n_p_lt_0_001           = sum(ks_pvalues .< 0.001),

        ks_pvalue_min          = minimum(ks_pvalues),
        ks_pvalue_q25          = quantile(ks_pvalues, 0.25),
        ks_pvalue_median       = median(ks_pvalues),
        ks_pvalue_mean         = mean(ks_pvalues),
        ks_pvalue_q75          = quantile(ks_pvalues, 0.75),
        ks_pvalue_max          = maximum(ks_pvalues),

        mean_post_mean         = mean(mean_post),
        mean_post_sd           = std(mean_post),
        mean_post_min          = minimum(mean_post),
        mean_post_median       = median(mean_post),
        mean_post_max          = maximum(mean_post),

        sd_post_mean           = mean(sd_post),
        sd_post_sd             = std(sd_post),
        sd_post_min            = minimum(sd_post),
        sd_post_median         = median(sd_post),
        sd_post_max            = maximum(sd_post),

        mean_prior_mean        = mean(mean_prior),
        mean_prior_sd          = std(mean_prior),
        mean_prior_min         = minimum(mean_prior),
        mean_prior_median      = median(mean_prior),
        mean_prior_max         = maximum(mean_prior),

        sd_prior_mean          = mean(sd_prior),
        sd_prior_sd            = std(sd_prior),
        sd_prior_min           = minimum(sd_prior),
        sd_prior_median        = median(sd_prior),
        sd_prior_max           = maximum(sd_prior),
    )
end

"""
Return borderline regions:
- updated rows with the largest p-values
- non-updated rows with the smallest p-values
"""
function borderline_regions(rows; nshow::Int=10)
    updated_rows    = filter(r -> r.updated == 1, rows)
    nonupdated_rows = filter(r -> r.updated == 0, rows)

    sort!(updated_rows, by = r -> r.ks_pvalue, rev=true)
    sort!(nonupdated_rows, by = r -> r.ks_pvalue)

    return updated_rows[1:min(nshow, length(updated_rows))],
           nonupdated_rows[1:min(nshow, length(nonupdated_rows))]
end

"""
Write a summary CSV.
"""
function write_summary_csv(outfile::AbstractString, rows)
    write_namedtuple_csv(outfile, rows)
end

"""
Write a human-readable summary TXT, including borderline regions.
"""
function write_summary_txt(outfile::AbstractString, summary_rows, row_lookup::Dict{Tuple{String,String},Any})
    open(outfile, "w") do io
        println(io, "Posterior-vs-prior update summary")
        println(io, "alpha = $(ALPHA)")
        println(io, "SAVE_ONLY_FLAGGED_PLOTS = $(SAVE_ONLY_FLAGGED_PLOTS)")
        println(io)

        for srow in summary_rows
            println(io, "==================================================")
            println(io, "Model:      ", srow.model)
            println(io, "Parameter:  ", srow.parameter)
            println(io, "Regions:    ", srow.n_regions)
            println(io, "Updated:    ", srow.n_updated, " (", round(100 * srow.frac_updated, digits=2), "%)")
            println(io)

            println(io, "KS p-value counts:")
            println(io, "  p < 0.05 :  ", srow.n_p_lt_0_05)
            println(io, "  p < 0.01 :  ", srow.n_p_lt_0_01)
            println(io, "  p < 0.001:  ", srow.n_p_lt_0_001)
            println(io)

            println(io, "KS p-values:")
            println(io, "  min    = ", srow.ks_pvalue_min)
            println(io, "  q25    = ", srow.ks_pvalue_q25)
            println(io, "  median = ", srow.ks_pvalue_median)
            println(io, "  mean   = ", srow.ks_pvalue_mean)
            println(io, "  q75    = ", srow.ks_pvalue_q75)
            println(io, "  max    = ", srow.ks_pvalue_max)
            println(io)

            println(io, "Posterior mean across regions:")
            println(io, "  mean   = ", srow.mean_post_mean)
            println(io, "  sd     = ", srow.mean_post_sd)
            println(io, "  min    = ", srow.mean_post_min)
            println(io, "  median = ", srow.mean_post_median)
            println(io, "  max    = ", srow.mean_post_max)
            println(io)

            println(io, "Posterior SD across regions:")
            println(io, "  mean   = ", srow.sd_post_mean)
            println(io, "  sd     = ", srow.sd_post_sd)
            println(io, "  min    = ", srow.sd_post_min)
            println(io, "  median = ", srow.sd_post_median)
            println(io, "  max    = ", srow.sd_post_max)
            println(io)

            println(io, "Prior mean across regions:")
            println(io, "  mean   = ", srow.mean_prior_mean)
            println(io, "  sd     = ", srow.mean_prior_sd)
            println(io, "  min    = ", srow.mean_prior_min)
            println(io, "  median = ", srow.mean_prior_median)
            println(io, "  max    = ", srow.mean_prior_max)
            println(io)

            println(io, "Prior SD across regions:")
            println(io, "  mean   = ", srow.sd_prior_mean)
            println(io, "  sd     = ", srow.sd_prior_sd)
            println(io, "  min    = ", srow.sd_prior_min)
            println(io, "  median = ", srow.sd_prior_median)
            println(io, "  max    = ", srow.sd_prior_max)
            println(io)

            rows = row_lookup[(srow.model, srow.parameter)]
            upd_border, nonupd_border = borderline_regions(rows; nshow=10)

            println(io, "Borderline updated regions (largest p among updated):")
            if isempty(upd_border)
                println(io, "  none")
            else
                for r in upd_border
                    println(io, "  ", r.region, "  | p = ", r.ks_pvalue, " | updated = ", r.updated)
                end
            end
            println(io)

            println(io, "Borderline non-updated regions (smallest p among non-updated):")
            if isempty(nonupd_border)
                println(io, "  none")
            else
                for r in nonupd_border
                    println(io, "  ", r.region, "  | p = ", r.ks_pvalue, " | updated = ", r.updated)
                end
            end
            println(io)
        end
    end
end

"""
Format p-values for annotation.
"""
function format_pvalue(p::Real)
    if isnan(p)
        return "NaN"
    elseif p < 1e-4
        return "< 1e-4"
    else
        return string(round(p, sigdigits=3))
    end
end

"""
Create beta-gamma scatter from row tables.
If updated_only=true, keep only rows with beta.updated==1 and gamma.updated==1.
Spearman rho is computed as Pearson correlation of tied ranks.
The p-value is taken from CorrelationTest on the tied ranks.
"""
function save_beta_gamma_scatter(
    beta_rows,
    gamma_rows,
    outfile::AbstractString;
    updated_only::Bool=false,
    title_prefix::AbstractString=model_name
)
    beta_map  = Dict(r.region => r for r in beta_rows)
    gamma_map = Dict(r.region => r for r in gamma_rows)

    common_regions = intersect(Set(keys(beta_map)), Set(keys(gamma_map)))
    regions = sort!(collect(common_regions))

    x = Float64[]
    y = Float64[]

    for region in regions
        br = beta_map[region]
        gr = gamma_map[region]

        if updated_only && !((br.updated == 1) && (gr.updated == 1))
            continue
        end

        push!(x, br.mean_post)
        push!(y, gr.mean_post)
    end

    ttl = updated_only ? "$title_prefix: beta vs gamma (updated only)" :
                         "$title_prefix: beta vs gamma (all regions)"

    plt = scatter(
        x, y;
        markersize=SCATTER_MARKERSIZE,
        alpha=SCATTER_ALPHA,
        xlabel="beta posterior mean",
        ylabel="gamma posterior mean",
        title="",
        label=false,
        guidefontsize=18,
        bottom_margin=5mm,
        top_margin=5mm
    )

    if length(x) >= 3
        rx = tiedrank(x)
        ry = tiedrank(y)
        ρ = cor(rx, ry)
        test = CorrelationTest(rx, ry)
        p = pvalue(test)

        ann = "Spearman ρ = $(round(ρ, sigdigits=3))\np = $(format_pvalue(p))\nn = $(length(x))"
    else
        @warn "Not enough points to compute Spearman correlation for $outfile"
        ann = "n = $(length(x))"
    end

    # Put annotation in top-left corner in DATA coordinates
    if !isempty(x) && !isempty(y)
        xmin, xmax = extrema(x)
        ymin, ymax = extrema(y)

        xpad = xmax == xmin ? 1.0 : 0.05 * (xmax - xmin)
        ypad = ymax == ymin ? 1.0 : 0.05 * (ymax - ymin)

        xloc = xmin + xpad
        yloc = ymax - ypad

        annotate!(plt, xloc, yloc, text(ann, 15, :black, :left))
    end

    savefig(plt, outfile)
end
# ============================================================
# MAIN
# ============================================================

println("Loading inference file...")
inf = load_inference(inference_file)

println("Loaded model: $model_name")
println("Inference file: $inference_file")
println("Using KS threshold alpha = $ALPHA")

summary_rows = NamedTuple[]
row_lookup = Dict{Tuple{String,String},Any}()

for prefix in prefixes_to_process
    rows = extract_parameter_table_and_plots(inf, prefix, model_name; alpha=ALPHA)

    row_lookup[(model_name, prefix)] = rows
    push!(summary_rows, summarize_rows(rows, model_name, prefix; alpha=ALPHA))

    outfile = joinpath(csv_dir, "$(model_name)_$(prefix)_optimal.csv")
    write_namedtuple_csv(outfile, rows)

    println("Saved → $outfile")
end

# ------------------------------------------------------------
# Save beta-gamma scatter plots
# ------------------------------------------------------------
if haskey(row_lookup, (model_name, "beta")) && haskey(row_lookup, (model_name, "gamma"))
    beta_rows = row_lookup[(model_name, "beta")]
    gamma_rows = row_lookup[(model_name, "gamma")]

    scatter_all_file = joinpath(scatter_dir, "$(model_name)_beta_vs_gamma_all.$(PLOT_FORMAT)")
    scatter_updated_file = joinpath(scatter_dir, "$(model_name)_beta_vs_gamma_UPDATED.$(PLOT_FORMAT)")

    save_beta_gamma_scatter(beta_rows, gamma_rows, scatter_all_file; updated_only=false, title_prefix=model_name)
    save_beta_gamma_scatter(beta_rows, gamma_rows, scatter_updated_file; updated_only=true, title_prefix=model_name)

    println("Saved → $scatter_all_file")
    println("Saved → $scatter_updated_file")
end

# ------------------------------------------------------------
# Save global summaries
# ------------------------------------------------------------
summary_csv = joinpath(save_dir, "update_summary.csv")
summary_txt = joinpath(save_dir, "update_summary.txt")

write_summary_csv(summary_csv, summary_rows)
write_summary_txt(summary_txt, summary_rows, row_lookup)

println("Saved → $summary_csv")
println("Saved → $summary_txt")
println("Done.")