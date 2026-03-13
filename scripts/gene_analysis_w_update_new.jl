using PathoSpread, CSV, DataFrames, Statistics, ProgressMeter, Random, HypothesisTests, Plots
using Serialization, Distributions, OrderedCollections, MCMCChains, StatsBase, Measures, LinearAlgebra

# Settings
# Choose how to summarise the posterior across samples:
#   :mean   -> posterior mean
#   :median -> posterior median
#   :map    -> joint MAP (mode) via max log-posterior (:lp)
const PARAM_SUMMARY = :mean

# to skip zero regions or not, and how to define zero regions (maximum over time below zero_threshold)
const skip_zero_regions = false
const zero_threshold = 0.01

# significance level
const α = 0.05

# updated-region filtering
const UPDATED_ONLY = true
const UPDATE_ALPHA = 0.001

# If true, z-score regression predictors before running multiple regression.
# For the interaction model, the interaction term is formed as z(beta) * z(gamma).
const STANDARDIZE_REGRESSION_PREDICTORS = true

# If true, additionally save filtered CSVs containing only genes with
# positive correlation with BOTH beta and gamma.
const SAVE_DOUBLE_POSITIVE_SUBSET = false
const DOUBLE_POSITIVE_SUFFIX = "_BETApos_GAMMApos"

# If true, save TXT files for the upper-right (beta>0,gamma>0) and
# lower-left (beta<0,gamma<0) quadrants from the beta-vs-gamma gene-alignment table.
const SAVE_QUADRANT_GENE_LISTS = true

# If true, also compute a joint score:
#   beta_scaled = beta / max(abs(beta))
#   gamma_scaled = (gamma - min(gamma)) / (max(gamma) - min(gamma))
#   joint_score = beta_scaled * gamma_scaled
# and save a gene-correlation CSV for that score.
const SAVE_BETA_GAMMA_SCALED_PRODUCT = false
const BETA_GAMMA_SCALED_PRODUCT_NAME = "beta_gamma_scaled_product"

# If true, also compute gamma and beta gene correlations only in regions with raw beta > threshold.
const SAVE_GAMMA_WHEN_BETA_POS = true
const SAVE_BETA_WHEN_BETA_POS  = true

const GAMMA_WHEN_BETA_POS_NAME = "gamma_when_beta_gt0"
const BETA_WHEN_BETA_POS_NAME  = "beta_when_beta_gt0"

const BETA_THRESHOLD_FOR_GAMMA = 0.0
const BETA_THRESHOLD_FOR_BETA  = 0.0

# Names for interaction-model outputs
const BETA_MULTIREG_INT_NAME       = "beta_multireg_interaction"
const GAMMA_MULTIREG_INT_NAME      = "gamma_multireg_interaction"
const BETAGAMMA_MULTIREG_INT_NAME  = "beta_gamma_interaction_multireg"

"""
    fdr_bh(pvals::AbstractVector{<:Real})

Compute Benjamini–Hochberg FDR-adjusted p-values.
Input and output vectors have the same ordering.
"""
function fdr_bh(pvals::AbstractVector{<:Real})
    N = length(pvals)
    p = Float64.(pvals)

    p_sorted, idx = sort(p), sortperm(p)
    q_raw = p_sorted .* (N ./ (1:N))

    q_sorted = similar(q_raw)
    q_sorted[end] = min(q_raw[end], 1.0)
    for i in (N-1):-1:1
        q_sorted[i] = min(q_raw[i], q_sorted[i+1])
    end

    q = similar(q_sorted)
    q[idx] = q_sorted
    return q
end

"""
Format p-values for printing/annotation.
"""
function format_p(p::Real)
    if isnan(p)
        return "NaN"
    elseif p < 1e-4
        return "< 1e-4"
    else
        return string(round(p, sigdigits=3))
    end
end

"""
Write one gene per line.
"""
function write_gene_list(outfile::AbstractString, genes)
    open(outfile, "w") do io
        for g in genes
            println(io, g)
        end
    end
end

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
function posterior_vs_prior_update_flag(samples::AbstractVector, prior_dist; alpha::Float64=UPDATE_ALPHA)
    test = ApproximateOneSampleKSTest(samples, prior_dist)
    p = pvalue(test)
    updated = Int(p < alpha)
    return p, updated
end

"""
Build Dict(region_name => updated flag) for a given parameter prefix, e.g. "beta" or "gamma".
Assumes inference["labels"] aligns with sorted prefix[i] parameters.
"""
function region_update_flags(inference::Dict, prefix::AbstractString; alpha::Float64=UPDATE_ALPHA)
    priors = inference["priors"]
    chain = inference["chain"]
    regions = String.(inference["labels"])

    param_names = parameter_names_by_prefix(priors, prefix)
    prior_keys = collect(keys(priors))

    if length(param_names) != length(regions)
        error("Length mismatch for $prefix: $(length(param_names)) parameters vs $(length(regions)) labels.")
    end

    flags = Dict{String,Int}()

    for (region, pname) in zip(regions, param_names)
        idx = findfirst(==(pname), prior_keys)
        idx === nothing && error("Could not find $pname in prior keys.")

        samples = posterior_samples_from_p(chain, idx)
        samples === nothing && error("Missing posterior samples for $pname at p[$idx].")

        prior_dist = priors[pname]
        _, updated = posterior_vs_prior_update_flag(samples, prior_dist; alpha=alpha)
        flags[String(region)] = updated
    end

    return flags
end

"""
Construct the common filename suffix used by outputs.
"""
function common_suffix(skip_zero_regions::Bool, updated_only::Bool)
    suffix = ""
    if skip_zero_regions
        suffix *= "_NONZERO"
    end
    if updated_only
        suffix *= "_UPDATED"
    end
    return suffix
end

"""
Additional suffix for regression outputs depending on whether predictors were z-scored.
"""
function regression_suffix()
    return STANDARDIZE_REGRESSION_PREDICTORS ? "_zscored" : ""
end

"""
Compute scaled joint score:
    beta_scaled = beta / max(abs(beta))
    gamma_scaled = (gamma - min(gamma)) / (max(gamma) - min(gamma))
    joint_score = beta_scaled * gamma_scaled
"""
function beta_gamma_scaled_product(beta_vec::AbstractVector, gamma_vec::AbstractVector)
    length(beta_vec) == length(gamma_vec) || error("beta_vec and gamma_vec must have same length")

    beta_den = maximum(abs.(beta_vec))
    if !isfinite(beta_den) || beta_den == 0
        error("Cannot scale beta: maximum(abs.(beta)) is zero or non-finite.")
    end
    beta_scaled = beta_vec ./ beta_den

    γmin = minimum(gamma_vec)
    γmax = maximum(gamma_vec)
    γrange = γmax - γmin
    if !isfinite(γrange) || γrange == 0
        error("Cannot scale gamma: range is zero or non-finite.")
    end
    gamma_scaled = (gamma_vec .- γmin) ./ γrange

    return beta_scaled .* gamma_scaled
end

"""
Build regional parameter vector aligned with gene_regions.
Optionally filter matched model regions using updated flags.
"""
function build_par_gene(
    par_vec::AbstractVector,
    gene_regions,
    region_map::Dict{String, Vector{String}},
    model_index::Dict{String,Int};
    updated_flags::Union{Nothing,Dict{String,Int}}=nothing
)
    R = length(gene_regions)
    par_gene = Array{Float64}(undef, R)
    n_model_matches = zeros(Int, R)
    n_model_matches_after_filter = zeros(Int, R)

    for (r, gr) in enumerate(gene_regions)
        regions = copy(region_map[String(gr)])
        n_model_matches[r] = length(regions)

        if updated_flags !== nothing
            regions = [mr for mr in regions if get(updated_flags, mr, 0) == 1]
        end
        n_model_matches_after_filter[r] = length(regions)

        if isempty(regions)
            par_gene[r] = NaN
            continue
        end

        idxs = [model_index[mr] for mr in regions]
        par_gene[r] = mean(par_vec[idxs])
    end

    return par_gene, n_model_matches, n_model_matches_after_filter
end

"""
Z-score a vector using mean/std computed on non-NaN entries.
NaNs are preserved.
If std == 0, returns zeros on valid entries.
"""
function zscore_with_nans(x::AbstractVector)
    out = fill(Float64(NaN), length(x))
    mask = .!isnan.(x)

    if sum(mask) == 0
        return out
    end

    μ = mean(x[mask])
    σ = std(x[mask])

    if !isfinite(σ) || σ == 0
        out[mask] .= 0.0
    else
        out[mask] = (x[mask] .- μ) ./ σ
    end

    return out
end

"""
Optionally standardize regression predictors.
Returns transformed beta, gamma, interaction.
If standardize=true, interaction is formed as z(beta) * z(gamma).
"""
function prepare_regression_predictors(beta_gene::AbstractVector, gamma_gene::AbstractVector;
    standardize::Bool = STANDARDIZE_REGRESSION_PREDICTORS)

    if standardize
        beta_use = zscore_with_nans(beta_gene)
        gamma_use = zscore_with_nans(gamma_gene)
        interaction_use = beta_use .* gamma_use
    else
        beta_use = Float64.(beta_gene)
        gamma_use = Float64.(gamma_gene)
        interaction_use = beta_use .* gamma_use
    end

    return beta_use, gamma_use, interaction_use
end

"""
OLS with intercept and two predictors:
    y = a + b1*x1 + b2*x2 + error
"""
function ols_two_predictors(y::AbstractVector, x1::AbstractVector, x2::AbstractVector)
    n = length(y)
    n == length(x1) == length(x2) || error("Inputs must have same length.")
    n < 4 && error("Need at least 4 observations for regression with intercept + 2 predictors.")

    X = hcat(ones(n), x1, x2)
    p = size(X, 2)

    βhat = X \ y
    yhat = X * βhat
    resid = y - yhat

    dof = n - p
    dof <= 0 && error("Non-positive residual degrees of freedom.")

    sse = sum(abs2, resid)
    sst = sum(abs2, y .- mean(y))
    σ2 = sse / dof

    XtX_inv = inv(X' * X)
    se = sqrt.(diag(σ2 .* XtX_inv))

    tstats = βhat ./ se
    pvals = [2 * (1 - cdf(TDist(dof), abs(t))) for t in tstats]

    r2 = sst > 0 ? 1 - sse / sst : NaN

    return (
        intercept = βhat[1],
        beta_coef = βhat[2],
        gamma_coef = βhat[3],
        intercept_p = pvals[1],
        beta_p = pvals[2],
        gamma_p = pvals[3],
        r2 = r2,
        n_used = n
    )
end

"""
OLS with intercept and three predictors:
    y = a + b1*x1 + b2*x2 + b3*x3 + error

Used here for:
    expr = a + b*beta + c*gamma + d*(beta*gamma) + error
"""
function ols_three_predictors(y::AbstractVector, x1::AbstractVector, x2::AbstractVector, x3::AbstractVector)
    n = length(y)
    n == length(x1) == length(x2) == length(x3) || error("Inputs must have same length.")
    n < 5 && error("Need at least 5 observations for regression with intercept + 3 predictors.")

    X = hcat(ones(n), x1, x2, x3)
    p = size(X, 2)

    βhat = X \ y
    yhat = X * βhat
    resid = y - yhat

    dof = n - p
    dof <= 0 && error("Non-positive residual degrees of freedom.")

    sse = sum(abs2, resid)
    sst = sum(abs2, y .- mean(y))
    σ2 = sse / dof

    XtX_inv = inv(X' * X)
    se = sqrt.(diag(σ2 .* XtX_inv))

    tstats = βhat ./ se
    pvals = [2 * (1 - cdf(TDist(dof), abs(t))) for t in tstats]

    r2 = sst > 0 ? 1 - sse / sst : NaN

    return (
        intercept = βhat[1],
        beta_coef = βhat[2],
        gamma_coef = βhat[3],
        interaction_coef = βhat[4],
        intercept_p = pvals[1],
        beta_p = pvals[2],
        gamma_p = pvals[3],
        interaction_p = pvals[4],
        r2 = r2,
        n_used = n
    )
end

"""
Post-hoc comparison of two gene-level association vectors.

Uses two per-gene result tables with schema:
    gene, r, p_un, p_fdr, p_bonf, n_used
"""
function save_gene_alignment(
    df1::DataFrame,
    df2::DataFrame;
    simulation::AbstractString,
    analysis_name::AbstractString,
    x_label::AbstractString,
    y_label::AbstractString,
    param_summary,
    skip_zero_regions::Bool,
    updated_only::Bool,
    α::Real = 0.05
)
    outdir = "results/gene_correlation"
    mkpath(outdir)

    suffix = common_suffix(skip_zero_regions, updated_only)

    base = joinpath(outdir, "$(simulation)_$(analysis_name)_$(String(param_summary))$(suffix)")

    d1 = select(df1, :gene, :r => :x, :p_un => :p_x, :p_fdr => :fdr_x, :p_bonf => :bonf_x, :n_used => :n_used_x)
    d2 = select(df2, :gene, :r => :y, :p_un => :p_y, :p_fdr => :fdr_y, :p_bonf => :bonf_y, :n_used => :n_used_y)

    df = innerjoin(d1, d2, on=:gene)

    valid_all = .!isnan.(df.x) .& .!isnan.(df.y)
    valid_either_un  = valid_all .& ((df.p_x .< α) .| (df.p_y .< α))
    valid_either_fdr = valid_all .& ((df.fdr_x .< α) .| (df.fdr_y .< α))

    summary_rows = NamedTuple[]

    subsets = [
        ("all_valid_genes", valid_all),
        ("either_uncorrected", valid_either_un),
        ("either_fdr", valid_either_fdr),
    ]

    for (label, mask) in subsets
        sub = df[mask, :]
        n = nrow(sub)

        if n < 3
            push!(summary_rows, (
                subset = label,
                n = n,
                pearson_r = NaN,
                pearson_p = NaN,
                spearman_rho = NaN,
                spearman_p = NaN,
                median_abs_x = NaN,
                median_abs_y = NaN
            ))
            continue
        end

        x = Float64.(sub.x)
        y = Float64.(sub.y)

        pearson_r = cor(x, y)
        pearson_p = pvalue(CorrelationTest(x, y))

        rx = tiedrank(x)
        ry = tiedrank(y)
        spearman_rho = cor(rx, ry)
        spearman_p = pvalue(CorrelationTest(rx, ry))

        push!(summary_rows, (
            subset = label,
            n = n,
            pearson_r = pearson_r,
            pearson_p = pearson_p,
            spearman_rho = spearman_rho,
            spearman_p = spearman_p,
            median_abs_x = median(abs.(x)),
            median_abs_y = median(abs.(y))
        ))

        plt = scatter(
            x, y;
            xlabel = x_label,
            ylabel = y_label,
            label = false,
            markersize = 4,
            alpha = 0.6,
            title = "",
            guidefontsize = 16,
            top_margin = 9mm
        )

        hline!(plt, [0.0]; color=:black, linestyle=:dash, lw=1, label=false)
        vline!(plt, [0.0]; color=:black, linestyle=:dash, lw=1, label=false)

        xmin, xmax = extrema(x)
        ymin, ymax = extrema(y)
        xann = xmin + 0.03 * (xmax - xmin + eps())
        yann = ymax - 0.03 * (ymax - ymin + eps())

        ann = "Pearson r = $(round(pearson_r, sigdigits=3))  (p = $(format_p(pearson_p)))\n" *
              "Spearman ρ = $(round(spearman_rho, sigdigits=3))  (p = $(format_p(spearman_p)))\n" *
              "n = $(n)"

        annotate!(plt, (xann, yann, text(ann, 10, :black, :left, :top)))
        savefig(plt, "$(base)_$(label).pdf")

        opp_idx  = (sub.x .* sub.y) .< 0
        same_idx = (sub.x .* sub.y) .> 0

        write_gene_list("$(base)_$(label)_opposite_sign_genes.txt", sub.gene[opp_idx])
        write_gene_list("$(base)_$(label)_same_sign_genes.txt", sub.gene[same_idx])
    end

    open("$(base)_summary.txt", "w") do io
        println(io, "Gene-level alignment analysis")
        println(io, "simulation = $simulation")
        println(io, "analysis_name = $analysis_name")
        println(io, "PARAM_SUMMARY = $(param_summary)")
        println(io, "skip_zero_regions = $(skip_zero_regions)")
        println(io, "UPDATED_ONLY = $(updated_only)")
        println(io, "alpha = $(α)")
        println(io, "x_label = $x_label")
        println(io, "y_label = $y_label")
        println(io)

        for row in summary_rows
            println(io, "========================================")
            println(io, "subset              = ", row.subset)
            println(io, "n                   = ", row.n)
            println(io, "Pearson r           = ", row.pearson_r)
            println(io, "Pearson p           = ", row.pearson_p)
            println(io, "Spearman rho        = ", row.spearman_rho)
            println(io, "Spearman p          = ", row.spearman_p)
            println(io, "median |x|          = ", row.median_abs_x)
            println(io, "median |y|          = ", row.median_abs_y)
            println(io)
        end
    end

    CSV.write("$(base)_summary.csv", DataFrame(summary_rows))
    return df, DataFrame(summary_rows)
end

"""
Like save_gene_alignment, but color the scatter by a third per-gene quantity.
"""
function save_gene_alignment_colored(
    df1::DataFrame,
    df2::DataFrame,
    dfc::DataFrame;
    simulation::AbstractString,
    analysis_name::AbstractString,
    x_label::AbstractString,
    y_label::AbstractString,
    color_label::AbstractString,
    param_summary,
    skip_zero_regions::Bool,
    updated_only::Bool,
    α::Real = 0.05
)
    outdir = "results/gene_correlation"
    mkpath(outdir)

    suffix = common_suffix(skip_zero_regions, updated_only)
    base = joinpath(outdir, "$(simulation)_$(analysis_name)_$(String(param_summary))$(suffix)")

    d1 = select(df1, :gene, :r => :x, :p_un => :p_x, :p_fdr => :fdr_x, :p_bonf => :bonf_x, :n_used => :n_used_x)
    d2 = select(df2, :gene, :r => :y, :p_un => :p_y, :p_fdr => :fdr_y, :p_bonf => :bonf_y, :n_used => :n_used_y)
    dc = select(dfc, :gene, :r => :z, :p_un => :p_z, :p_fdr => :fdr_z, :p_bonf => :bonf_z, :n_used => :n_used_z)

    df = innerjoin(innerjoin(d1, d2, on=:gene), dc, on=:gene)

    valid_all = .!isnan.(df.x) .& .!isnan.(df.y) .& .!isnan.(df.z)
    valid_either_un  = valid_all .& ((df.p_x .< α) .| (df.p_y .< α) .| (df.p_z .< α))
    valid_either_fdr = valid_all .& ((df.fdr_x .< α) .| (df.fdr_y .< α) .| (df.fdr_z .< α))

    summary_rows = NamedTuple[]
    subsets = [
        ("all_valid_genes", valid_all),
        ("either_uncorrected", valid_either_un),
        ("either_fdr", valid_either_fdr),
    ]

    for (label, mask) in subsets
        sub = df[mask, :]
        n = nrow(sub)

        if n < 3
            push!(summary_rows, (
                subset = label,
                n = n,
                pearson_r_xy = NaN,
                pearson_p_xy = NaN,
                spearman_rho_xy = NaN,
                spearman_p_xy = NaN,
                median_abs_x = NaN,
                median_abs_y = NaN,
                median_abs_z = NaN
            ))
            continue
        end

        x = Float64.(sub.x)
        y = Float64.(sub.y)
        z = Float64.(sub.z)

        pearson_r_xy = cor(x, y)
        pearson_p_xy = pvalue(CorrelationTest(x, y))
        rx = tiedrank(x)
        ry = tiedrank(y)
        spearman_rho_xy = cor(rx, ry)
        spearman_p_xy = pvalue(CorrelationTest(rx, ry))

        push!(summary_rows, (
            subset = label,
            n = n,
            pearson_r_xy = pearson_r_xy,
            pearson_p_xy = pearson_p_xy,
            spearman_rho_xy = spearman_rho_xy,
            spearman_p_xy = spearman_p_xy,
            median_abs_x = median(abs.(x)),
            median_abs_y = median(abs.(y)),
            median_abs_z = median(abs.(z))
        ))

        plt = scatter(
            x, y;
            marker_z = z,
            xlabel = x_label,
            ylabel = y_label,
            colorbar_title = color_label,
            label = false,
            markersize = 5,
            alpha = 0.75,
            title = "",
            guidefontsize = 16,
            top_margin = 9mm
        )

        hline!(plt, [0.0]; color=:black, linestyle=:dash, lw=1, label=false)
        vline!(plt, [0.0]; color=:black, linestyle=:dash, lw=1, label=false)

        xmin, xmax = extrema(x)
        ymin, ymax = extrema(y)
        xann = xmin + 0.03 * (xmax - xmin + eps())
        yann = ymax - 0.03 * (ymax - ymin + eps())

        ann = "Pearson r = $(round(pearson_r_xy, sigdigits=3))  (p = $(format_p(pearson_p_xy)))\n" *
              "Spearman ρ = $(round(spearman_rho_xy, sigdigits=3))  (p = $(format_p(spearman_p_xy)))\n" *
              "n = $(n)"

        annotate!(plt, (xann, yann, text(ann, 10, :black, :left, :top)))
        savefig(plt, "$(base)_$(label).pdf")

        opp_idx  = (sub.x .* sub.y) .< 0
        same_idx = (sub.x .* sub.y) .> 0

        write_gene_list("$(base)_$(label)_opposite_sign_genes.txt", sub.gene[opp_idx])
        write_gene_list("$(base)_$(label)_same_sign_genes.txt", sub.gene[same_idx])
    end

    open("$(base)_summary.txt", "w") do io
        println(io, "Gene-level colored alignment analysis")
        println(io, "simulation = $simulation")
        println(io, "analysis_name = $analysis_name")
        println(io, "PARAM_SUMMARY = $(param_summary)")
        println(io, "skip_zero_regions = $(skip_zero_regions)")
        println(io, "UPDATED_ONLY = $(updated_only)")
        println(io, "alpha = $(α)")
        println(io, "x_label = $x_label")
        println(io, "y_label = $y_label")
        println(io, "color_label = $color_label")
        println(io)

        for row in summary_rows
            println(io, "========================================")
            println(io, "subset              = ", row.subset)
            println(io, "n                   = ", row.n)
            println(io, "Pearson r (x,y)     = ", row.pearson_r_xy)
            println(io, "Pearson p (x,y)     = ", row.pearson_p_xy)
            println(io, "Spearman rho (x,y)  = ", row.spearman_rho_xy)
            println(io, "Spearman p (x,y)    = ", row.spearman_p_xy)
            println(io, "median |x|          = ", row.median_abs_x)
            println(io, "median |y|          = ", row.median_abs_y)
            println(io, "median |z|          = ", row.median_abs_z)
            println(io)
        end
    end

    CSV.write("$(base)_summary.csv", DataFrame(summary_rows))
    return df, DataFrame(summary_rows)
end

function save_double_positive_subset_csvs(
    gene_corr_results::Dict{String,DataFrame},
    alignment_df::DataFrame;
    simulation::AbstractString,
    param_summary,
    skip_zero_regions::Bool,
    updated_only::Bool,
    subset_suffix::AbstractString = DOUBLE_POSITIVE_SUFFIX
)
    if !haskey(gene_corr_results, "beta") || !haskey(gene_corr_results, "gamma")
        error("gene_corr_results must contain both 'beta' and 'gamma'.")
    end

    mask = (.!isnan.(alignment_df.x)) .&
           (.!isnan.(alignment_df.y)) .&
           (alignment_df.x .> 0) .&
           (alignment_df.y .> 0)

    selected_genes = Set(String.(alignment_df.gene[mask]))
    println("Double-positive gene subset size = $(length(selected_genes))")

    suffix = common_suffix(skip_zero_regions, updated_only)

    for par_name in ["beta", "gamma"]
        df_full = gene_corr_results[par_name]
        keep = [String(g) in selected_genes for g in df_full.gene]
        df_sub = df_full[keep, :]

        out_path = "results/gene_correlation/$(simulation)_gene_corr_$(par_name)_$(String(param_summary))$(suffix)$(subset_suffix).csv"
        CSV.write(out_path, df_sub)
        println("Saved → $out_path")
    end

    genes_path = "results/gene_correlation/$(simulation)_double_positive_genes_$(String(param_summary))$(suffix).txt"
    write_gene_list(genes_path, sort!(collect(selected_genes)))
    println("Saved → $genes_path")
end

function save_quadrant_gene_lists(
    alignment_df::DataFrame;
    simulation::AbstractString,
    param_summary,
    skip_zero_regions::Bool,
    updated_only::Bool
)
    suffix = common_suffix(skip_zero_regions, updated_only)

    valid = (.!isnan.(alignment_df.x)) .& (.!isnan.(alignment_df.y))

    upper_right = sort!(collect(Set(String.(alignment_df.gene[valid .& (alignment_df.x .> 0) .& (alignment_df.y .> 0)]))))
    lower_left  = sort!(collect(Set(String.(alignment_df.gene[valid .& (alignment_df.x .< 0) .& (alignment_df.y .< 0)]))))

    upper_right_path = "results/gene_correlation/$(simulation)_upper_right_genes_$(String(param_summary))$(suffix).txt"
    lower_left_path  = "results/gene_correlation/$(simulation)_lower_left_genes_$(String(param_summary))$(suffix).txt"

    write_gene_list(upper_right_path, upper_right)
    write_gene_list(lower_left_path, lower_left)

    println("Saved → $upper_right_path")
    println("Saved → $lower_left_path")
    println("Upper-right gene count = $(length(upper_right))")
    println("Lower-left gene count  = $(length(lower_left))")
end

# Read Gene expression dataset
gene_data = CSV.read("data/avg_Pangea_exp.csv", DataFrame)

# find region and gene IDs, and convert to matrix
gene_regions = String.(gene_data[:, 1])
gene_IDs = names(gene_data)[2:end]
gene_matrix = Matrix(gene_data[:, 2:end])

# Read Disease Spreading Model
simulation = "DIFFGA_RETRO"
inference = load_inference("simulations/" * simulation * ".jls")

# Extract raw data used in inference
nonzero_nodes = nonzero_regions(inference["data"], eps=zero_threshold)

# Extract parameters of interest
chain = inference["chain"]
priors = inference["priors"]
model_regions = String.(inference["labels"])
parameter_names = collect(keys(priors))

# find indices of beta and gamma parameters in chain
beta_ind = findall(k -> startswith(k, "beta["), parameter_names)
gamma_ind = findall(k -> startswith(k, "gamma["), parameter_names)

# precompute updated flags if requested
beta_updated_flags  = UPDATED_ONLY ? region_update_flags(inference, "beta";  alpha=UPDATE_ALPHA) : Dict{String,Int}()
gamma_updated_flags = UPDATED_ONLY ? region_update_flags(inference, "gamma"; alpha=UPDATE_ALPHA) : Dict{String,Int}()

# find regions common to both gene data and model
region_map = Dict{String, Vector{String}}()
for gr in gene_regions
    gr_str = String(gr)
    L = length(gr_str)
    matches = String[]

    for (i, mr) in enumerate(model_regions)
        if skip_zero_regions && !(i in nonzero_nodes)
            continue
        end

        mr_str = String(mr)

        if !(startswith(mr_str, "i") || startswith(mr_str, "c"))
            continue
        end

        if length(mr_str) < 1 + L
            continue
        end

        mr_sub = mr_str[2:1+L]

        if mr_sub == gr_str
            if length(mr_str) == 1 + L
                push!(matches, mr_str)
                continue
            end

            next_char = mr_str[2+L]
            if !isletter(next_char)
                push!(matches, mr_str)
            end
        end
    end

    region_map[gr_str] = matches
end

# look-up for model regions
model_index = Dict(model_regions[i] => i for i in eachindex(model_regions))

#######################################################################
# === FREQUENTIST ANALYSIS USING SUMMARY PARAMETER FIELD ==============
#######################################################################
println("\n=== Running frequentist analysis using summary parameter fields ===")
println("UPDATED_ONLY = $(UPDATED_ONLY)")
println("STANDARDIZE_REGRESSION_PREDICTORS = $(STANDARDIZE_REGRESSION_PREDICTORS)")
if UPDATED_ONLY
    println("UPDATE_ALPHA = $(UPDATE_ALPHA)")
end
println("SAVE_DOUBLE_POSITIVE_SUBSET = $(SAVE_DOUBLE_POSITIVE_SUBSET)")
if SAVE_DOUBLE_POSITIVE_SUBSET
    println("DOUBLE_POSITIVE_SUFFIX = $(DOUBLE_POSITIVE_SUFFIX)")
end
println("SAVE_QUADRANT_GENE_LISTS = $(SAVE_QUADRANT_GENE_LISTS)")
println("SAVE_BETA_GAMMA_SCALED_PRODUCT = $(SAVE_BETA_GAMMA_SCALED_PRODUCT)")
if SAVE_BETA_GAMMA_SCALED_PRODUCT
    println("BETA_GAMMA_SCALED_PRODUCT_NAME = $(BETA_GAMMA_SCALED_PRODUCT_NAME)")
end
println("SAVE_GAMMA_WHEN_BETA_POS = $(SAVE_GAMMA_WHEN_BETA_POS)")
if SAVE_GAMMA_WHEN_BETA_POS
    println("GAMMA_WHEN_BETA_POS_NAME = $(GAMMA_WHEN_BETA_POS_NAME)")
    println("BETA_THRESHOLD_FOR_GAMMA = $(BETA_THRESHOLD_FOR_GAMMA)")
end
println("SAVE_BETA_WHEN_BETA_POS = $(SAVE_BETA_WHEN_BETA_POS)")
if SAVE_BETA_WHEN_BETA_POS
    println("BETA_WHEN_BETA_POS_NAME = $(BETA_WHEN_BETA_POS_NAME)")
    println("BETA_THRESHOLD_FOR_BETA = $(BETA_THRESHOLD_FOR_BETA)")
end

# Extract posterior arrays: iter × param × chain
beta_post  = Array(chain[:, beta_ind, :])
gamma_post = Array(chain[:, gamma_ind, :])

# Posterior mean across iterations and chains
beta_mean   = vec(mean(beta_post,  dims=1))
gamma_mean  = vec(mean(gamma_post, dims=1))

# Posterior median across iterations and chains
beta_median  = vec(median(beta_post,  dims=1))
gamma_median = vec(median(gamma_post, dims=1))

# log-posterior for each sample × chain
lp_mat = Array(chain[:lp])

# Find MAP index (linear)
imax = argmax(lp_mat)

# Convert linear index to sample + chain indices
idx = CartesianIndices(lp_mat)[imax]
i_idx = idx[1]
c_idx = idx[2]

println("MAP sample = $(i_idx), chain = $(c_idx), lp = $(lp_mat[i_idx, c_idx])")
beta_map  = vec(Array(chain[i_idx, beta_ind,  c_idx]))
gamma_map = vec(Array(chain[i_idx, gamma_ind, c_idx]))

# Pick which summary to use
function pick_summary(par_name::String)
    if PARAM_SUMMARY == :mean
        return par_name == "beta" ? beta_mean : gamma_mean
    elseif PARAM_SUMMARY == :median
        return par_name == "beta" ? beta_median : gamma_median
    elseif PARAM_SUMMARY == :map
        return par_name == "beta" ? beta_map : gamma_map
    else
        error("Unknown PARAM_SUMMARY = $PARAM_SUMMARY")
    end
end

function pick_updated_flags(par_name::String)
    if par_name == "beta"
        return beta_updated_flags
    elseif par_name == "gamma"
        return gamma_updated_flags
    else
        error("Unknown parameter name: $par_name")
    end
end

R = length(gene_regions)
G = size(gene_matrix, 2)

mkpath("results/gene_correlation")

# store outputs so we can compare beta vs gamma afterward
gene_corr_results = Dict{String,DataFrame}()

###############################################################
# Univariate correlations
###############################################################
for par_name in ["beta", "gamma"]

    println("  Using $(PARAM_SUMMARY) summary for $par_name")

    par_vec = pick_summary(par_name)
    updated_flags = pick_updated_flags(par_name)

    par_gene = Array{Float64}(undef, R)
    n_model_matches = zeros(Int, R)
    n_model_matches_after_filter = zeros(Int, R)

    for (r, gr) in enumerate(gene_regions)
        regions = region_map[gr]
        n_model_matches[r] = length(regions)

        if UPDATED_ONLY
            regions = [mr for mr in regions if get(updated_flags, mr, 0) == 1]
        end
        n_model_matches_after_filter[r] = length(regions)

        if isempty(regions)
            par_gene[r] = NaN
            continue
        end

        idxs = [model_index[mr] for mr in regions]
        par_gene[r] = mean(par_vec[idxs])
    end

    println("  Regions with at least one matched model region before filter: ",
            count(>(0), n_model_matches))
    if UPDATED_ONLY
        println("  Regions with at least one matched model region after UPDATED filter: ",
                count(>(0), n_model_matches_after_filter))
    end

    println("  Computing correlations for $par_name ...")

    r_vec = Array{Float64}(undef, G)
    p_vec = Array{Float64}(undef, G)
    n_used_vec = Array{Int}(undef, G)

    for g in 1:G
        expr = gene_matrix[:, g]
        mask = .!isnan.(par_gene)

        n_used_vec[g] = sum(mask)

        if n_used_vec[g] < 3
            r_vec[g] = NaN
            p_vec[g] = NaN
            continue
        end

        r_vec[g] = cor(expr[mask], par_gene[mask])
        p_vec[g] = pvalue(CorrelationTest(expr[mask], par_gene[mask]))
    end

    p_bonf = [isnan(p) ? NaN : min(p * G, 1.0) for p in p_vec]
    p_fdr  = fdr_bh(p_vec)

    out_df = DataFrame(
        gene   = gene_IDs,
        r      = Float32.(r_vec),
        p_un   = Float32.(p_vec),
        p_bonf = Float32.(p_bonf),
        p_fdr  = Float32.(p_fdr),
        n_used = n_used_vec
    )

    gene_corr_results[par_name] = out_df

    suffix = common_suffix(skip_zero_regions, UPDATED_ONLY)
    out_path = "results/gene_correlation/$(simulation)_gene_corr_$(par_name)_$(String(PARAM_SUMMARY))$(suffix).csv"
    CSV.write(out_path, out_df)
    println("  Saved frequentist summary to $out_path")

    valid = .!isnan.(p_vec)

    println("\nSignificant genes for $par_name (summary = $(PARAM_SUMMARY)):")
    println("  Uncorrected p < $(α)      : ", count(i -> valid[i] && p_vec[i]   < α, eachindex(p_vec)))
    println("  Bonferroni p < $(α)       : ", count(i -> valid[i] && p_bonf[i] < α, eachindex(p_bonf)))
    println("  FDR BH p < $(α)           : ", count(i -> valid[i] && p_fdr[i]  < α, eachindex(p_fdr)))

    idx_un   = [i for i in eachindex(p_vec)   if valid[i] && p_vec[i]   < α]
    idx_fdr  = [i for i in eachindex(p_fdr)   if valid[i] && p_fdr[i]   < α]
    idx_bonf = [i for i in eachindex(p_bonf)  if valid[i] && p_bonf[i]  < α]

    genes_un   = gene_IDs[idx_un]
    genes_fdr  = gene_IDs[idx_fdr]
    genes_bonf = gene_IDs[idx_bonf]

    base = "results/gene_correlation/$(simulation)_siggenes_$(par_name)_$(String(PARAM_SUMMARY))$(suffix)"

    open(base * "_uncorrected.txt", "w") do io
        for g in genes_un
            println(io, g)
        end
    end
    open(base * "_fdr.txt", "w") do io
        for g in genes_fdr
            println(io, g)
        end
    end
    open(base * "_bonferroni.txt", "w") do io
        for g in genes_bonf
            println(io, g)
        end
    end
    println("  Wrote significant-gene TXT files for $par_name")
end

###############################################################
# Joint multiple regression: expr ~ beta + gamma
###############################################################
println("\nRunning joint multiple regression: expression ~ beta + gamma ...")

beta_vec_joint  = pick_summary("beta")
gamma_vec_joint = pick_summary("gamma")

beta_flags_joint  = UPDATED_ONLY ? beta_updated_flags  : nothing
gamma_flags_joint = UPDATED_ONLY ? gamma_updated_flags : nothing

beta_gene, beta_n_match, beta_n_after = build_par_gene(
    beta_vec_joint, gene_regions, region_map, model_index;
    updated_flags = beta_flags_joint
)

gamma_gene, gamma_n_match, gamma_n_after = build_par_gene(
    gamma_vec_joint, gene_regions, region_map, model_index;
    updated_flags = gamma_flags_joint
)

beta_reg, gamma_reg, _ = prepare_regression_predictors(beta_gene, gamma_gene)

println("  Regions with beta matches before filter: ", count(>(0), beta_n_match))
println("  Regions with gamma matches before filter: ", count(>(0), gamma_n_match))
if UPDATED_ONLY
    println("  Regions with beta matches after UPDATED filter: ", count(>(0), beta_n_after))
    println("  Regions with gamma matches after UPDATED filter: ", count(>(0), gamma_n_after))
end

beta_coef_vec      = Array{Float64}(undef, G)
gamma_coef_vec     = Array{Float64}(undef, G)
intercept_vec      = Array{Float64}(undef, G)
beta_p_vec         = Array{Float64}(undef, G)
gamma_p_vec        = Array{Float64}(undef, G)
intercept_p_vec    = Array{Float64}(undef, G)
r2_vec             = Array{Float64}(undef, G)
n_used_vec         = Array{Int}(undef, G)

for g in 1:G
    expr = gene_matrix[:, g]
    mask = .!isnan.(beta_reg) .& .!isnan.(gamma_reg) .& .!isnan.(expr)
    n_used_vec[g] = sum(mask)

    if n_used_vec[g] < 4
        intercept_vec[g]   = NaN
        beta_coef_vec[g]   = NaN
        gamma_coef_vec[g]  = NaN
        intercept_p_vec[g] = NaN
        beta_p_vec[g]      = NaN
        gamma_p_vec[g]     = NaN
        r2_vec[g]          = NaN
        continue
    end

    fit = ols_two_predictors(expr[mask], beta_reg[mask], gamma_reg[mask])

    intercept_vec[g]   = fit.intercept
    beta_coef_vec[g]   = fit.beta_coef
    gamma_coef_vec[g]  = fit.gamma_coef
    intercept_p_vec[g] = fit.intercept_p
    beta_p_vec[g]      = fit.beta_p
    gamma_p_vec[g]     = fit.gamma_p
    r2_vec[g]          = fit.r2
end

beta_p_bonf  = [isnan(p) ? NaN : min(p * G, 1.0) for p in beta_p_vec]
beta_p_fdr   = fdr_bh(beta_p_vec)
gamma_p_bonf = [isnan(p) ? NaN : min(p * G, 1.0) for p in gamma_p_vec]
gamma_p_fdr  = fdr_bh(gamma_p_vec)

out_df_beta_joint = DataFrame(
    gene   = gene_IDs,
    r      = Float32.(beta_coef_vec),
    p_un   = Float32.(beta_p_vec),
    p_bonf = Float32.(beta_p_bonf),
    p_fdr  = Float32.(beta_p_fdr),
    n_used = n_used_vec
)

out_df_gamma_joint = DataFrame(
    gene   = gene_IDs,
    r      = Float32.(gamma_coef_vec),
    p_un   = Float32.(gamma_p_vec),
    p_bonf = Float32.(gamma_p_bonf),
    p_fdr  = Float32.(gamma_p_fdr),
    n_used = n_used_vec
)

gene_corr_results["beta_multireg"] = out_df_beta_joint
gene_corr_results["gamma_multireg"] = out_df_gamma_joint

suffix = common_suffix(skip_zero_regions, UPDATED_ONLY)
reg_sfx = regression_suffix()

out_path_beta  = "results/gene_correlation/$(simulation)_gene_corr_beta_multireg$(reg_sfx)_$(String(PARAM_SUMMARY))$(suffix).csv"
out_path_gamma = "results/gene_correlation/$(simulation)_gene_corr_gamma_multireg$(reg_sfx)_$(String(PARAM_SUMMARY))$(suffix).csv"

CSV.write(out_path_beta, out_df_beta_joint)
CSV.write(out_path_gamma, out_df_gamma_joint)

println("  Saved joint-regression beta summary to $out_path_beta")
println("  Saved joint-regression gamma summary to $out_path_gamma")

valid_beta  = .!isnan.(beta_p_vec)
valid_gamma = .!isnan.(gamma_p_vec)

println("\nSignificant genes for beta coefficient in joint model:")
println("  Uncorrected p < $(α)      : ", count(i -> valid_beta[i]  && beta_p_vec[i]   < α, eachindex(beta_p_vec)))
println("  Bonferroni p < $(α)       : ", count(i -> valid_beta[i]  && beta_p_bonf[i]  < α, eachindex(beta_p_bonf)))
println("  FDR BH p < $(α)           : ", count(i -> valid_beta[i]  && beta_p_fdr[i]   < α, eachindex(beta_p_fdr)))

println("\nSignificant genes for gamma coefficient in joint model:")
println("  Uncorrected p < $(α)      : ", count(i -> valid_gamma[i] && gamma_p_vec[i]   < α, eachindex(gamma_p_vec)))
println("  Bonferroni p < $(α)       : ", count(i -> valid_gamma[i] && gamma_p_bonf[i]  < α, eachindex(gamma_p_bonf)))
println("  FDR BH p < $(α)           : ", count(i -> valid_gamma[i] && gamma_p_fdr[i]   < α, eachindex(gamma_p_fdr)))

base = "results/gene_correlation/$(simulation)_siggenes_beta_gamma_multiple_regression$(reg_sfx)_$(String(PARAM_SUMMARY))$(suffix)"

open(base * "_beta_uncorrected.txt", "w") do io
    for g in gene_IDs[[i for i in eachindex(beta_p_vec) if valid_beta[i] && beta_p_vec[i] < α]]
        println(io, g)
    end
end

open(base * "_beta_fdr.txt", "w") do io
    for g in gene_IDs[[i for i in eachindex(beta_p_fdr) if valid_beta[i] && beta_p_fdr[i] < α]]
        println(io, g)
    end
end

open(base * "_beta_bonferroni.txt", "w") do io
    for g in gene_IDs[[i for i in eachindex(beta_p_bonf) if valid_beta[i] && beta_p_bonf[i] < α]]
        println(io, g)
    end
end

open(base * "_gamma_uncorrected.txt", "w") do io
    for g in gene_IDs[[i for i in eachindex(gamma_p_vec) if valid_gamma[i] && gamma_p_vec[i] < α]]
        println(io, g)
    end
end

open(base * "_gamma_fdr.txt", "w") do io
    for g in gene_IDs[[i for i in eachindex(gamma_p_fdr) if valid_gamma[i] && gamma_p_fdr[i] < α]]
        println(io, g)
    end
end

open(base * "_gamma_bonferroni.txt", "w") do io
    for g in gene_IDs[[i for i in eachindex(gamma_p_bonf) if valid_gamma[i] && gamma_p_bonf[i] < α]]
        println(io, g)
    end
end

println("  Wrote significant-gene TXT files for joint beta+gamma regression")

###############################################################
# Interaction multiple regression: expr ~ beta + gamma + beta*gamma
###############################################################
println("\nRunning interaction multiple regression: expression ~ beta + gamma + beta*gamma ...")

beta_reg_int, gamma_reg_int, interaction_gene = prepare_regression_predictors(beta_gene, gamma_gene)

beta_coef_int_vec        = Array{Float64}(undef, G)
gamma_coef_int_vec       = Array{Float64}(undef, G)
interaction_coef_vec     = Array{Float64}(undef, G)
intercept_int_vec        = Array{Float64}(undef, G)
beta_p_int_vec           = Array{Float64}(undef, G)
gamma_p_int_vec          = Array{Float64}(undef, G)
interaction_p_vec        = Array{Float64}(undef, G)
intercept_p_int_vec      = Array{Float64}(undef, G)
r2_int_vec               = Array{Float64}(undef, G)
n_used_int_vec           = Array{Int}(undef, G)

for g in 1:G
    expr = gene_matrix[:, g]
    mask = .!isnan.(beta_reg_int) .& .!isnan.(gamma_reg_int) .& .!isnan.(interaction_gene) .& .!isnan.(expr)
    n_used_int_vec[g] = sum(mask)

    if n_used_int_vec[g] < 5
        intercept_int_vec[g]    = NaN
        beta_coef_int_vec[g]    = NaN
        gamma_coef_int_vec[g]   = NaN
        interaction_coef_vec[g] = NaN
        intercept_p_int_vec[g]  = NaN
        beta_p_int_vec[g]       = NaN
        gamma_p_int_vec[g]      = NaN
        interaction_p_vec[g]    = NaN
        r2_int_vec[g]           = NaN
        continue
    end

    fit = ols_three_predictors(
        expr[mask],
        beta_reg_int[mask],
        gamma_reg_int[mask],
        interaction_gene[mask]
    )

    intercept_int_vec[g]    = fit.intercept
    beta_coef_int_vec[g]    = fit.beta_coef
    gamma_coef_int_vec[g]   = fit.gamma_coef
    interaction_coef_vec[g] = fit.interaction_coef
    intercept_p_int_vec[g]  = fit.intercept_p
    beta_p_int_vec[g]       = fit.beta_p
    gamma_p_int_vec[g]      = fit.gamma_p
    interaction_p_vec[g]    = fit.interaction_p
    r2_int_vec[g]           = fit.r2
end

beta_p_int_bonf        = [isnan(p) ? NaN : min(p * G, 1.0) for p in beta_p_int_vec]
beta_p_int_fdr         = fdr_bh(beta_p_int_vec)
gamma_p_int_bonf       = [isnan(p) ? NaN : min(p * G, 1.0) for p in gamma_p_int_vec]
gamma_p_int_fdr        = fdr_bh(gamma_p_int_vec)
interaction_p_bonf     = [isnan(p) ? NaN : min(p * G, 1.0) for p in interaction_p_vec]
interaction_p_fdr      = fdr_bh(interaction_p_vec)

out_df_beta_int = DataFrame(
    gene   = gene_IDs,
    r      = Float32.(beta_coef_int_vec),
    p_un   = Float32.(beta_p_int_vec),
    p_bonf = Float32.(beta_p_int_bonf),
    p_fdr  = Float32.(beta_p_int_fdr),
    n_used = n_used_int_vec
)

out_df_gamma_int = DataFrame(
    gene   = gene_IDs,
    r      = Float32.(gamma_coef_int_vec),
    p_un   = Float32.(gamma_p_int_vec),
    p_bonf = Float32.(gamma_p_int_bonf),
    p_fdr  = Float32.(gamma_p_int_fdr),
    n_used = n_used_int_vec
)

out_df_interaction_int = DataFrame(
    gene   = gene_IDs,
    r      = Float32.(interaction_coef_vec),
    p_un   = Float32.(interaction_p_vec),
    p_bonf = Float32.(interaction_p_bonf),
    p_fdr  = Float32.(interaction_p_fdr),
    n_used = n_used_int_vec
)

gene_corr_results[BETA_MULTIREG_INT_NAME]      = out_df_beta_int
gene_corr_results[GAMMA_MULTIREG_INT_NAME]     = out_df_gamma_int
gene_corr_results[BETAGAMMA_MULTIREG_INT_NAME] = out_df_interaction_int

out_path_beta_int = "results/gene_correlation/$(simulation)_gene_corr_$(BETA_MULTIREG_INT_NAME)$(reg_sfx)_$(String(PARAM_SUMMARY))$(suffix).csv"
out_path_gamma_int = "results/gene_correlation/$(simulation)_gene_corr_$(GAMMA_MULTIREG_INT_NAME)$(reg_sfx)_$(String(PARAM_SUMMARY))$(suffix).csv"
out_path_interaction_int = "results/gene_correlation/$(simulation)_gene_corr_$(BETAGAMMA_MULTIREG_INT_NAME)$(reg_sfx)_$(String(PARAM_SUMMARY))$(suffix).csv"

CSV.write(out_path_beta_int, out_df_beta_int)
CSV.write(out_path_gamma_int, out_df_gamma_int)
CSV.write(out_path_interaction_int, out_df_interaction_int)

println("  Saved interaction-regression beta summary to $out_path_beta_int")
println("  Saved interaction-regression gamma summary to $out_path_gamma_int")
println("  Saved interaction-regression beta*gamma summary to $out_path_interaction_int")

valid_beta_int        = .!isnan.(beta_p_int_vec)
valid_gamma_int       = .!isnan.(gamma_p_int_vec)
valid_interaction_int = .!isnan.(interaction_p_vec)

println("\nSignificant genes for beta coefficient in interaction model:")
println("  Uncorrected p < $(α)      : ", count(i -> valid_beta_int[i] && beta_p_int_vec[i] < α, eachindex(beta_p_int_vec)))
println("  Bonferroni p < $(α)       : ", count(i -> valid_beta_int[i] && beta_p_int_bonf[i] < α, eachindex(beta_p_int_bonf)))
println("  FDR BH p < $(α)           : ", count(i -> valid_beta_int[i] && beta_p_int_fdr[i] < α, eachindex(beta_p_int_fdr)))

println("\nSignificant genes for gamma coefficient in interaction model:")
println("  Uncorrected p < $(α)      : ", count(i -> valid_gamma_int[i] && gamma_p_int_vec[i] < α, eachindex(gamma_p_int_vec)))
println("  Bonferroni p < $(α)       : ", count(i -> valid_gamma_int[i] && gamma_p_int_bonf[i] < α, eachindex(gamma_p_int_bonf)))
println("  FDR BH p < $(α)           : ", count(i -> valid_gamma_int[i] && gamma_p_int_fdr[i] < α, eachindex(gamma_p_int_fdr)))

println("\nSignificant genes for beta*gamma interaction coefficient in interaction model:")
println("  Uncorrected p < $(α)      : ", count(i -> valid_interaction_int[i] && interaction_p_vec[i] < α, eachindex(interaction_p_vec)))
println("  Bonferroni p < $(α)       : ", count(i -> valid_interaction_int[i] && interaction_p_bonf[i] < α, eachindex(interaction_p_bonf)))
println("  FDR BH p < $(α)           : ", count(i -> valid_interaction_int[i] && interaction_p_fdr[i] < α, eachindex(interaction_p_fdr)))

base_int = "results/gene_correlation/$(simulation)_siggenes_interaction_multiple_regression$(reg_sfx)_$(String(PARAM_SUMMARY))$(suffix)"

open(base_int * "_beta_uncorrected.txt", "w") do io
    for g in gene_IDs[[i for i in eachindex(beta_p_int_vec) if valid_beta_int[i] && beta_p_int_vec[i] < α]]
        println(io, g)
    end
end

open(base_int * "_beta_fdr.txt", "w") do io
    for g in gene_IDs[[i for i in eachindex(beta_p_int_fdr) if valid_beta_int[i] && beta_p_int_fdr[i] < α]]
        println(io, g)
    end
end

open(base_int * "_beta_bonferroni.txt", "w") do io
    for g in gene_IDs[[i for i in eachindex(beta_p_int_bonf) if valid_beta_int[i] && beta_p_int_bonf[i] < α]]
        println(io, g)
    end
end

open(base_int * "_gamma_uncorrected.txt", "w") do io
    for g in gene_IDs[[i for i in eachindex(gamma_p_int_vec) if valid_gamma_int[i] && gamma_p_int_vec[i] < α]]
        println(io, g)
    end
end

open(base_int * "_gamma_fdr.txt", "w") do io
    for g in gene_IDs[[i for i in eachindex(gamma_p_int_fdr) if valid_gamma_int[i] && gamma_p_int_fdr[i] < α]]
        println(io, g)
    end
end

open(base_int * "_gamma_bonferroni.txt", "w") do io
    for g in gene_IDs[[i for i in eachindex(gamma_p_int_bonf) if valid_gamma_int[i] && gamma_p_int_bonf[i] < α]]
        println(io, g)
    end
end

open(base_int * "_betagamma_uncorrected.txt", "w") do io
    for g in gene_IDs[[i for i in eachindex(interaction_p_vec) if valid_interaction_int[i] && interaction_p_vec[i] < α]]
        println(io, g)
    end
end

open(base_int * "_betagamma_fdr.txt", "w") do io
    for g in gene_IDs[[i for i in eachindex(interaction_p_fdr) if valid_interaction_int[i] && interaction_p_fdr[i] < α]]
        println(io, g)
    end
end

open(base_int * "_betagamma_bonferroni.txt", "w") do io
    for g in gene_IDs[[i for i in eachindex(interaction_p_bonf) if valid_interaction_int[i] && interaction_p_bonf[i] < α]]
        println(io, g)
    end
end

println("  Wrote significant-gene TXT files for interaction multiple regression")

###############################################################
# Optional: gamma correlation restricted to raw beta > threshold
###############################################################
if SAVE_GAMMA_WHEN_BETA_POS
    println("\nRunning $(GAMMA_WHEN_BETA_POS_NAME) gene-correlation analysis ...")

    beta_vec_for_filter = beta_mean
    gamma_vec_for_filter = gamma_mean
    if PARAM_SUMMARY == :median
        beta_vec_for_filter = beta_median
        gamma_vec_for_filter = gamma_median
    elseif PARAM_SUMMARY == :map
        beta_vec_for_filter = beta_map
        gamma_vec_for_filter = gamma_map
    end

    par_gene = Array{Float64}(undef, R)
    n_model_matches = zeros(Int, R)
    n_model_matches_after_filter = zeros(Int, R)

    for (r, gr) in enumerate(gene_regions)
        regions = region_map[gr]
        n_model_matches[r] = length(regions)

        if UPDATED_ONLY
            regions = [mr for mr in regions if (get(beta_updated_flags, mr, 0) == 1) && (get(gamma_updated_flags, mr, 0) == 1)]
        end

        regions = [mr for mr in regions if beta_vec_for_filter[model_index[mr]] > BETA_THRESHOLD_FOR_GAMMA]
        n_model_matches_after_filter[r] = length(regions)

        if isempty(regions)
            par_gene[r] = NaN
            continue
        end

        idxs = [model_index[mr] for mr in regions]
        par_gene[r] = mean(gamma_vec_for_filter[idxs])
    end

    println("  Regions with at least one matched model region before filter: ",
            count(>(0), n_model_matches))
    println("  Regions with at least one matched model region after beta > $(BETA_THRESHOLD_FOR_GAMMA) filter: ",
            count(>(0), n_model_matches_after_filter))

    println("  Computing correlations for $(GAMMA_WHEN_BETA_POS_NAME) ...")

    r_vec = Array{Float64}(undef, G)
    p_vec = Array{Float64}(undef, G)
    n_used_vec = Array{Int}(undef, G)

    for g in 1:G
        expr = gene_matrix[:, g]
        mask = .!isnan.(par_gene)

        n_used_vec[g] = sum(mask)

        if n_used_vec[g] < 3
            r_vec[g] = NaN
            p_vec[g] = NaN
            continue
        end

        r_vec[g] = cor(expr[mask], par_gene[mask])
        p_vec[g] = pvalue(CorrelationTest(expr[mask], par_gene[mask]))
    end

    p_bonf = [isnan(p) ? NaN : min(p * G, 1.0) for p in p_vec]
    p_fdr  = fdr_bh(p_vec)

    out_df = DataFrame(
        gene   = gene_IDs,
        r      = Float32.(r_vec),
        p_un   = Float32.(p_vec),
        p_bonf = Float32.(p_bonf),
        p_fdr  = Float32.(p_fdr),
        n_used = n_used_vec
    )

    metric_key = GAMMA_WHEN_BETA_POS_NAME
    gene_corr_results[metric_key] = out_df

    out_path = "results/gene_correlation/$(simulation)_gene_corr_$(GAMMA_WHEN_BETA_POS_NAME)_$(String(PARAM_SUMMARY))$(suffix).csv"
    CSV.write(out_path, out_df)
    println("  Saved frequentist summary to $out_path")

    valid = .!isnan.(p_vec)
    println("\nSignificant genes for $(GAMMA_WHEN_BETA_POS_NAME) (summary = $(PARAM_SUMMARY)):")
    println("  Uncorrected p < $(α)      : ", count(i -> valid[i] && p_vec[i]   < α, eachindex(p_vec)))
    println("  Bonferroni p < $(α)       : ", count(i -> valid[i] && p_bonf[i] < α, eachindex(p_bonf)))
    println("  FDR BH p < $(α)           : ", count(i -> valid[i] && p_fdr[i]  < α, eachindex(p_fdr)))

    idx_un   = [i for i in eachindex(p_vec)   if valid[i] && p_vec[i]   < α]
    idx_fdr  = [i for i in eachindex(p_fdr)   if valid[i] && p_fdr[i]   < α]
    idx_bonf = [i for i in eachindex(p_bonf)  if valid[i] && p_bonf[i]  < α]

    genes_un   = gene_IDs[idx_un]
    genes_fdr  = gene_IDs[idx_fdr]
    genes_bonf = gene_IDs[idx_bonf]

    base = "results/gene_correlation/$(simulation)_siggenes_$(GAMMA_WHEN_BETA_POS_NAME)_$(String(PARAM_SUMMARY))$(suffix)"

    open(base * "_uncorrected.txt", "w") do io
        for g in genes_un
            println(io, g)
        end
    end
    open(base * "_fdr.txt", "w") do io
        for g in genes_fdr
            println(io, g)
        end
    end
    open(base * "_bonferroni.txt", "w") do io
        for g in genes_bonf
            println(io, g)
        end
    end
    println("  Wrote significant-gene TXT files for $(GAMMA_WHEN_BETA_POS_NAME)")
end

###############################################################
# Optional: beta correlation restricted to raw beta > threshold
###############################################################
if SAVE_BETA_WHEN_BETA_POS
    println("\nRunning $(BETA_WHEN_BETA_POS_NAME) gene-correlation analysis ...")

    beta_vec_for_filter = beta_mean
    if PARAM_SUMMARY == :median
        beta_vec_for_filter = beta_median
    elseif PARAM_SUMMARY == :map
        beta_vec_for_filter = beta_map
    end

    par_gene = Array{Float64}(undef, R)
    n_model_matches = zeros(Int, R)
    n_model_matches_after_filter = zeros(Int, R)

    for (r, gr) in enumerate(gene_regions)
        regions = region_map[gr]
        n_model_matches[r] = length(regions)

        if UPDATED_ONLY
            regions = [mr for mr in regions if get(beta_updated_flags, mr, 0) == 1]
        end

        regions = [mr for mr in regions if beta_vec_for_filter[model_index[mr]] > BETA_THRESHOLD_FOR_BETA]
        n_model_matches_after_filter[r] = length(regions)

        if isempty(regions)
            par_gene[r] = NaN
            continue
        end

        idxs = [model_index[mr] for mr in regions]
        par_gene[r] = mean(beta_vec_for_filter[idxs])
    end

    println("  Regions with at least one matched model region before filter: ",
            count(>(0), n_model_matches))
    println("  Regions with at least one matched model region after beta > $(BETA_THRESHOLD_FOR_BETA) filter: ",
            count(>(0), n_model_matches_after_filter))

    println("  Computing correlations for $(BETA_WHEN_BETA_POS_NAME) ...")

    r_vec = Array{Float64}(undef, G)
    p_vec = Array{Float64}(undef, G)
    n_used_vec = Array{Int}(undef, G)

    for g in 1:G
        expr = gene_matrix[:, g]
        mask = .!isnan.(par_gene)

        n_used_vec[g] = sum(mask)

        if n_used_vec[g] < 3
            r_vec[g] = NaN
            p_vec[g] = NaN
            continue
        end

        r_vec[g] = cor(expr[mask], par_gene[mask])
        p_vec[g] = pvalue(CorrelationTest(expr[mask], par_gene[mask]))
    end

    p_bonf = [isnan(p) ? NaN : min(p * G, 1.0) for p in p_vec]
    p_fdr  = fdr_bh(p_vec)

    out_df = DataFrame(
        gene   = gene_IDs,
        r      = Float32.(r_vec),
        p_un   = Float32.(p_vec),
        p_bonf = Float32.(p_bonf),
        p_fdr  = Float32.(p_fdr),
        n_used = n_used_vec
    )

    metric_key = BETA_WHEN_BETA_POS_NAME
    gene_corr_results[metric_key] = out_df

    out_path = "results/gene_correlation/$(simulation)_gene_corr_$(BETA_WHEN_BETA_POS_NAME)_$(String(PARAM_SUMMARY))$(suffix).csv"
    CSV.write(out_path, out_df)
    println("  Saved frequentist summary to $out_path")

    valid = .!isnan.(p_vec)
    println("\nSignificant genes for $(BETA_WHEN_BETA_POS_NAME) (summary = $(PARAM_SUMMARY)):")
    println("  Uncorrected p < $(α)      : ", count(i -> valid[i] && p_vec[i]   < α, eachindex(p_vec)))
    println("  Bonferroni p < $(α)       : ", count(i -> valid[i] && p_bonf[i] < α, eachindex(p_bonf)))
    println("  FDR BH p < $(α)           : ", count(i -> valid[i] && p_fdr[i]  < α, eachindex(p_fdr)))

    idx_un   = [i for i in eachindex(p_vec)   if valid[i] && p_vec[i]   < α]
    idx_fdr  = [i for i in eachindex(p_fdr)   if valid[i] && p_fdr[i]   < α]
    idx_bonf = [i for i in eachindex(p_bonf)  if valid[i] && p_bonf[i]  < α]

    genes_un   = gene_IDs[idx_un]
    genes_fdr  = gene_IDs[idx_fdr]
    genes_bonf = gene_IDs[idx_bonf]

    base = "results/gene_correlation/$(simulation)_siggenes_$(BETA_WHEN_BETA_POS_NAME)_$(String(PARAM_SUMMARY))$(suffix)"

    open(base * "_uncorrected.txt", "w") do io
        for g in genes_un
            println(io, g)
        end
    end
    open(base * "_fdr.txt", "w") do io
        for g in genes_fdr
            println(io, g)
        end
    end
    open(base * "_bonferroni.txt", "w") do io
        for g in genes_bonf
            println(io, g)
        end
    end
    println("  Wrote significant-gene TXT files for $(BETA_WHEN_BETA_POS_NAME)")
end

###############################################################
# Optional: beta-gamma scaled product gene correlation analysis
###############################################################
if SAVE_BETA_GAMMA_SCALED_PRODUCT
    println("\nRunning $(BETA_GAMMA_SCALED_PRODUCT_NAME) gene-correlation analysis ...")

    beta_vec_for_joint = beta_mean
    gamma_vec_for_joint = gamma_mean
    if PARAM_SUMMARY == :median
        beta_vec_for_joint = beta_median
        gamma_vec_for_joint = gamma_median
    elseif PARAM_SUMMARY == :map
        beta_vec_for_joint = beta_map
        gamma_vec_for_joint = gamma_map
    end

    joint_vec = beta_gamma_scaled_product(beta_vec_for_joint, gamma_vec_for_joint)

    par_gene = Array{Float64}(undef, R)
    n_model_matches = zeros(Int, R)
    n_model_matches_after_filter = zeros(Int, R)

    for (r, gr) in enumerate(gene_regions)
        beta_regions = region_map[gr]
        n_model_matches[r] = length(beta_regions)

        regions = beta_regions
        if UPDATED_ONLY
            regions = [mr for mr in regions if (get(beta_updated_flags, mr, 0) == 1) && (get(gamma_updated_flags, mr, 0) == 1)]
        end
        n_model_matches_after_filter[r] = length(regions)

        if isempty(regions)
            par_gene[r] = NaN
            continue
        end

        idxs = [model_index[mr] for mr in regions]
        par_gene[r] = mean(joint_vec[idxs])
    end

    println("  Regions with at least one matched model region before filter: ",
            count(>(0), n_model_matches))
    if UPDATED_ONLY
        println("  Regions with at least one matched model region after joint UPDATED filter: ",
                count(>(0), n_model_matches_after_filter))
    end

    println("  Computing correlations for $(BETA_GAMMA_SCALED_PRODUCT_NAME) ...")

    r_vec = Array{Float64}(undef, G)
    p_vec = Array{Float64}(undef, G)
    n_used_vec = Array{Int}(undef, G)

    for g in 1:G
        expr = gene_matrix[:, g]
        mask = .!isnan.(par_gene)

        n_used_vec[g] = sum(mask)

        if n_used_vec[g] < 3
            r_vec[g] = NaN
            p_vec[g] = NaN
            continue
        end

        r_vec[g] = cor(expr[mask], par_gene[mask])
        p_vec[g] = pvalue(CorrelationTest(expr[mask], par_gene[mask]))
    end

    p_bonf = [isnan(p) ? NaN : min(p * G, 1.0) for p in p_vec]
    p_fdr  = fdr_bh(p_vec)

    out_df = DataFrame(
        gene   = gene_IDs,
        r      = Float32.(r_vec),
        p_un   = Float32.(p_vec),
        p_bonf = Float32.(p_bonf),
        p_fdr  = Float32.(p_fdr),
        n_used = n_used_vec
    )

    gene_corr_results[BETA_GAMMA_SCALED_PRODUCT_NAME] = out_df

    out_path = "results/gene_correlation/$(simulation)_gene_corr_$(BETA_GAMMA_SCALED_PRODUCT_NAME)_$(String(PARAM_SUMMARY))$(suffix).csv"
    CSV.write(out_path, out_df)
    println("  Saved frequentist summary to $out_path")

    valid = .!isnan.(p_vec)
    println("\nSignificant genes for $(BETA_GAMMA_SCALED_PRODUCT_NAME) (summary = $(PARAM_SUMMARY)):")
    println("  Uncorrected p < $(α)      : ", count(i -> valid[i] && p_vec[i]   < α, eachindex(p_vec)))
    println("  Bonferroni p < $(α)       : ", count(i -> valid[i] && p_bonf[i] < α, eachindex(p_bonf)))
    println("  FDR BH p < $(α)           : ", count(i -> valid[i] && p_fdr[i]  < α, eachindex(p_fdr)))

    idx_un   = [i for i in eachindex(p_vec)   if valid[i] && p_vec[i]   < α]
    idx_fdr  = [i for i in eachindex(p_fdr)   if valid[i] && p_fdr[i]   < α]
    idx_bonf = [i for i in eachindex(p_bonf)  if valid[i] && p_bonf[i]  < α]

    genes_un   = gene_IDs[idx_un]
    genes_fdr  = gene_IDs[idx_fdr]
    genes_bonf = gene_IDs[idx_bonf]

    base = "results/gene_correlation/$(simulation)_siggenes_$(BETA_GAMMA_SCALED_PRODUCT_NAME)_$(String(PARAM_SUMMARY))$(suffix)"

    open(base * "_uncorrected.txt", "w") do io
        for g in genes_un
            println(io, g)
        end
    end
    open(base * "_fdr.txt", "w") do io
        for g in genes_fdr
            println(io, g)
        end
    end
    open(base * "_bonferroni.txt", "w") do io
        for g in genes_bonf
            println(io, g)
        end
    end
    println("  Wrote significant-gene TXT files for $(BETA_GAMMA_SCALED_PRODUCT_NAME)")
end

###############################################################
# Alignment analyses
###############################################################
if haskey(gene_corr_results, "beta") && haskey(gene_corr_results, "gamma")
    println("\nRunning beta-gamma gene alignment analysis (correlations) ...")
    alignment_df, alignment_summary = save_gene_alignment(
        gene_corr_results["beta"],
        gene_corr_results["gamma"];
        simulation = simulation,
        analysis_name = "beta_gamma_gene_alignment",
        x_label = "gene correlation with beta",
        y_label = "gene correlation with gamma",
        param_summary = PARAM_SUMMARY,
        skip_zero_regions = skip_zero_regions,
        updated_only = UPDATED_ONLY,
        α = α
    )
    println("Saved beta-gamma correlation alignment plots and summaries.")

    if SAVE_DOUBLE_POSITIVE_SUBSET
        println("\nSaving double-positive subset CSVs ...")
        save_double_positive_subset_csvs(
            gene_corr_results,
            alignment_df;
            simulation = simulation,
            param_summary = PARAM_SUMMARY,
            skip_zero_regions = skip_zero_regions,
            updated_only = UPDATED_ONLY,
            subset_suffix = DOUBLE_POSITIVE_SUFFIX
        )
    end

    if SAVE_QUADRANT_GENE_LISTS
        println("\nSaving upper-right and lower-left gene lists ...")
        save_quadrant_gene_lists(
            alignment_df;
            simulation = simulation,
            param_summary = PARAM_SUMMARY,
            skip_zero_regions = skip_zero_regions,
            updated_only = UPDATED_ONLY
        )
    end
end

if haskey(gene_corr_results, "beta_multireg") && haskey(gene_corr_results, "gamma_multireg")
    println("\nRunning beta-gamma gene alignment analysis (multiple regression coefficients) ...")
    multireg_alignment_df, multireg_alignment_summary = save_gene_alignment(
        gene_corr_results["beta_multireg"],
        gene_corr_results["gamma_multireg"];
        simulation = simulation,
        analysis_name = "beta_gamma_multireg_alignment$(reg_sfx)",
        x_label = STANDARDIZE_REGRESSION_PREDICTORS ? "standardized gene coefficient for beta" : "gene coefficient for beta",
        y_label = STANDARDIZE_REGRESSION_PREDICTORS ? "standardized gene coefficient for gamma" : "gene coefficient for gamma",
        param_summary = PARAM_SUMMARY,
        skip_zero_regions = skip_zero_regions,
        updated_only = UPDATED_ONLY,
        α = α
    )
    println("Saved beta-gamma multireg alignment plots and summaries.")

    if SAVE_QUADRANT_GENE_LISTS
        println("\nSaving upper-right and lower-left multireg gene lists ...")
        save_quadrant_gene_lists(
            multireg_alignment_df;
            simulation = simulation * "_multireg" * reg_sfx,
            param_summary = PARAM_SUMMARY,
            skip_zero_regions = skip_zero_regions,
            updated_only = UPDATED_ONLY
        )
    end
end

if haskey(gene_corr_results, BETA_MULTIREG_INT_NAME) &&
   haskey(gene_corr_results, GAMMA_MULTIREG_INT_NAME) &&
   haskey(gene_corr_results, BETAGAMMA_MULTIREG_INT_NAME)

    println("\nRunning beta-gamma interaction-model alignment analysis (beta vs gamma, colored by beta*gamma) ...")
    interaction_alignment_df, interaction_alignment_summary = save_gene_alignment_colored(
        gene_corr_results[BETA_MULTIREG_INT_NAME],
        gene_corr_results[GAMMA_MULTIREG_INT_NAME],
        gene_corr_results[BETAGAMMA_MULTIREG_INT_NAME];
        simulation = simulation,
        analysis_name = "beta_gamma_interaction_multireg_alignment_colored$(reg_sfx)",
        x_label = STANDARDIZE_REGRESSION_PREDICTORS ? "standardized gene coefficient for beta" : "gene coefficient for beta",
        y_label = STANDARDIZE_REGRESSION_PREDICTORS ? "standardized gene coefficient for gamma" : "gene coefficient for gamma",
        color_label = STANDARDIZE_REGRESSION_PREDICTORS ? "standardized gene coefficient for beta*gamma" : "gene coefficient for beta*gamma",
        param_summary = PARAM_SUMMARY,
        skip_zero_regions = skip_zero_regions,
        updated_only = UPDATED_ONLY,
        α = α
    )
    println("Saved beta-gamma interaction-model colored alignment plots and summaries.")

    if SAVE_QUADRANT_GENE_LISTS
        println("\nSaving upper-right and lower-left interaction-model gene lists ...")
        save_quadrant_gene_lists(
            interaction_alignment_df;
            simulation = simulation * "_interaction_multireg" * reg_sfx,
            param_summary = PARAM_SUMMARY,
            skip_zero_regions = skip_zero_regions,
            updated_only = UPDATED_ONLY
        )
    end
end

println("\n=== Frequentist analysis with $(PARAM_SUMMARY) summary complete ===\n")

#################################
# === Save all gene names ======
#################################
open("results/gene_correlation/PANGEA_all_genes.txt", "w") do io
    for g in gene_IDs
        println(io, g)
    end
end

println("\nSaved list of ALL gene names to results/gene_correlation/PANGEA_all_genes.txt")