#!/usr/bin/env julia

using PathoSpread, CSV, DataFrames, Statistics, HypothesisTests, Plots
using Serialization, Distributions, OrderedCollections, MCMCChains, StatsBase, Measures, LinearAlgebra
using LaTeXStrings

# ============================================================
# SETTINGS
# ============================================================

const SIMULATION = "DIFFGA_RETRO"
const GENE_DATA_CSV = "data/avg_Pangea_exp.csv"

# posterior summary for beta/gamma parameters
# :mean, :median, :map
const PARAM_SUMMARY = :mean

# region filtering
const SKIP_ZERO_REGIONS = false
const ZERO_THRESHOLD = 0.01

# updated-region filtering
const UPDATED_ONLY = true
const UPDATE_ALPHA = 0.001

# OPTIONAL: keep only regions with beta > threshold
const BETA_POS_ONLY = true
const BETA_POS_THRESHOLD = 0.0

# use z-scored beta/gamma predictors in the multiple regression
const STANDARDIZE_REGRESSION_PREDICTORS = true

# significance level
const α = 0.05

# orient PC1 so that increasing eta means increasing gamma
const FORCE_PC1_POSITIVE_GAMMA = true

# orient PC2 so positive eta2 corresponds to higher beta + gamma
const FORCE_PC2_POSITIVE_BETA_PLUS_GAMMA = true

# annotate selected regions on regional plots
const ANNOTATE_REGIONS = true
const REGIONS_TO_ANNOTATE = ["SNc", "VTA"]
const REGION_ANNOTATION_FONTSIZE = 22

# output
const OUTROOT_BASE = "results/gene_correlation_pca_axis"
const OUTROOT = BETA_POS_ONLY ? "$(OUTROOT_BASE)_beta_gt_0" : OUTROOT_BASE
const OUTCSV  = joinpath(OUTROOT, "csv")
const OUTFIG  = joinpath(OUTROOT, "figures")
const OUTTXT  = joinpath(OUTROOT, "txt")

mkpath(OUTROOT)
mkpath(OUTCSV)
mkpath(OUTFIG)
mkpath(OUTTXT)

# ============================================================
# HELPERS
# ============================================================

function fdr_bh(pvals::AbstractVector{<:Real})
    N = length(pvals)
    p = Float64.(pvals)

    valid = .!isnan.(p)
    q = fill(Float64(NaN), N)

    if !any(valid)
        return q
    end

    pv = p[valid]
    ord = sortperm(pv)
    p_sorted = pv[ord]

    q_raw = p_sorted .* (length(p_sorted) ./ (1:length(p_sorted)))

    q_sorted = similar(q_raw)
    q_sorted[end] = min(q_raw[end], 1.0)
    for i in (length(q_raw)-1):-1:1
        q_sorted[i] = min(q_raw[i], q_sorted[i+1])
    end

    q_valid = similar(q_sorted)
    q_valid[ord] = q_sorted
    q[valid] = q_valid

    return q
end

function format_p(p::Real)
    if isnan(p)
        return "NaN"
    elseif p < 1e-4
        return "< 1e-4"
    else
        return string(round(p, sigdigits=3))
    end
end

function write_gene_list(outfile::AbstractString, genes)
    open(outfile, "w") do io
        for g in genes
            println(io, g)
        end
    end
end

function posterior_samples_from_p(chain::Chains, i::Int)
    var = Symbol("p[$i]")
    if !(var in names(chain))
        return nothing
    end
    return vec(Array(chain[var]))
end

function parameter_names_by_prefix(priors, prefix::AbstractString)
    names_ = filter(n -> startswith(n, prefix * "["), collect(keys(priors)))
    sort!(names_, by = n -> parse(Int, match(Regex("^" * prefix * "\\[(\\d+)\\]"), n).captures[1]))
    return names_
end

function posterior_vs_prior_update_flag(samples::AbstractVector, prior_dist; alpha::Float64=UPDATE_ALPHA)
    test = ApproximateOneSampleKSTest(samples, prior_dist)
    p = pvalue(test)
    updated = Int(p < alpha)
    return p, updated
end

function region_update_flags(inference::Dict, prefix::AbstractString; alpha::Float64=UPDATE_ALPHA)
    priors  = inference["priors"]
    chain   = inference["chain"]
    regions = String.(inference["labels"])

    param_names = parameter_names_by_prefix(priors, prefix)
    prior_keys  = collect(keys(priors))

    length(param_names) == length(regions) ||
        error("Length mismatch for $prefix: $(length(param_names)) parameters vs $(length(regions)) labels.")

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

function common_suffix(skip_zero_regions::Bool, updated_only::Bool, beta_pos_only::Bool)
    suffix = ""
    if skip_zero_regions
        suffix *= "_NONZERO"
    end
    if updated_only
        suffix *= "_UPDATED"
    end
    if beta_pos_only
        suffix *= "_BETAGT0"
    end
    return suffix
end

function regression_suffix()
    return STANDARDIZE_REGRESSION_PREDICTORS ? "_zscored" : ""
end

function zscore_with_nans(x::AbstractVector)
    out = fill(Float64(NaN), length(x))
    mask = .!isnan.(x)

    if !any(mask)
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
        intercept   = βhat[1],
        beta_coef   = βhat[2],
        gamma_coef  = βhat[3],
        intercept_p = pvals[1],
        beta_p      = pvals[2],
        gamma_p     = pvals[3],
        r2          = r2,
        n_used      = n
    )
end

"""
Manual 2D PCA on rows of X (n x 2), after centering.
Returns:
- mean_vec
- eigvecs: columns are PC1, PC2
- eigvals: descending
- explained_var_ratio
- scores
"""
function pca_2d(X::AbstractMatrix)
    size(X, 2) == 2 || error("X must have exactly 2 columns.")
    n = size(X, 1)
    n >= 2 || error("Need at least 2 rows for PCA.")

    μ = vec(mean(X, dims=1))
    Xc = X .- reshape(μ, 1, :)

    Σ = (Xc' * Xc) / (n - 1)
    F = eigen(Symmetric(Σ))

    ord = sortperm(F.values, rev=true)
    eigvals = F.values[ord]
    eigvecs = F.vectors[:, ord]

    expl = eigvals ./ sum(eigvals)
    scores = Xc * eigvecs

    return (
        mean_vec = μ,
        eigvecs = eigvecs,
        eigvals = eigvals,
        explained_var_ratio = expl,
        scores = scores
    )
end

function plot_pca_lines!(
    plt,
    center::AbstractVector,
    v1::AbstractVector,
    v2::AbstractVector,
    xrange::Tuple,
    yrange::Tuple;
    scale1=0.45,
    scale2=0.35,
    label1="PC1",
    label2="PC2",
    lw=8
)
    xspan = xrange[2] - xrange[1]
    yspan = yrange[2] - yrange[1]

    L1 = scale1 * min(xspan / max(abs(v1[1]), eps()), yspan / max(abs(v1[2]), eps()))
    L2 = scale2 * min(xspan / max(abs(v2[1]), eps()), yspan / max(abs(v2[2]), eps()))

    p1a = center .- L1 .* v1
    p1b = center .+ L1 .* v1
    p2a = center .- L2 .* v2
    p2b = center .+ L2 .* v2

    plot!(plt, [p2a[1], p2b[1]], [p2a[2], p2b[2]];
        lw=lw, color=:gray35, linestyle=:solid, label=label2)
    plot!(plt, [p1a[1], p1b[1]], [p1a[2], p1b[2]];
        lw=lw, color=:black, label=label1)

    return plt
end

function save_sig_gene_lists(base::AbstractString, gene_IDs, p_vec, p_bonf, p_fdr; alpha=α)
    valid = .!isnan.(p_vec)

    idx_un   = [i for i in eachindex(p_vec)  if valid[i] && p_vec[i]  < alpha]
    idx_bonf = [i for i in eachindex(p_bonf) if valid[i] && p_bonf[i] < alpha]
    idx_fdr  = [i for i in eachindex(p_fdr)  if valid[i] && p_fdr[i]  < alpha]

    write_gene_list(base * "_uncorrected.txt", gene_IDs[idx_un])
    write_gene_list(base * "_bonferroni.txt", gene_IDs[idx_bonf])
    write_gene_list(base * "_fdr.txt", gene_IDs[idx_fdr])
end

function gene_correlations_with_score(score_region::AbstractVector, gene_matrix, gene_IDs; alpha=α)
    G = size(gene_matrix, 2)

    r_vec      = fill(Float64(NaN), G)
    p_vec      = fill(Float64(NaN), G)
    n_used_vec = zeros(Int, G)

    for g in 1:G
        expr = Float64.(gene_matrix[:, g])
        mask = .!isnan.(score_region) .& .!isnan.(expr)
        n_used_vec[g] = sum(mask)

        if n_used_vec[g] < 3
            continue
        end

        r_vec[g] = cor(expr[mask], score_region[mask])
        p_vec[g] = pvalue(CorrelationTest(expr[mask], score_region[mask]))
    end

    p_bonf = [isnan(p) ? NaN : min(p * G, 1.0) for p in p_vec]
    p_fdr  = fdr_bh(p_vec)

    df = DataFrame(
        gene   = gene_IDs,
        r      = Float32.(r_vec),
        p_un   = Float32.(p_vec),
        p_bonf = Float32.(p_bonf),
        p_fdr  = Float32.(p_fdr),
        n_used = n_used_vec
    )

    return df, r_vec, p_vec, p_bonf, p_fdr, n_used_vec
end

# affine form in raw beta/gamma coordinates
function affine_score_raw(beta_vals, gamma_vals, beta_coef_raw, gamma_coef_raw, intercept_raw)
    return beta_coef_raw .* beta_vals .+ gamma_coef_raw .* gamma_vals .+ intercept_raw
end

function annotate_selected_regions!(
    plt,
    xvals::AbstractVector,
    yvals::AbstractVector,
    labels::AbstractVector{<:AbstractString};
    selected_regions::AbstractVector{<:AbstractString}=REGIONS_TO_ANNOTATE,
    fontsize::Int=REGION_ANNOTATION_FONTSIZE,
    color=RGB(0.7, 0.7, 0.7),
    xoffset_frac::Float64=0.045,
    yoffset_frac::Float64=0.035
)
    isempty(xvals) && return plt
    isempty(yvals) && return plt

    xmin, xmax = extrema(xvals)
    ymin, ymax = extrema(yvals)

    dx = xoffset_frac * (xmax - xmin + eps())
    dy = yoffset_frac * (ymax - ymin + eps())

    selected_set = Set(String.(selected_regions))

    for i in eachindex(labels)
        lab = String(labels[i])
        if lab in selected_set && isfinite(xvals[i]) && isfinite(yvals[i])
            annotate!(plt, (xvals[i] + dx, yvals[i] + dy, text(lab, fontsize, color, :left)))
        end
    end

    return plt
end

# ============================================================
# LOAD DATA
# ============================================================

println("\n=== PCA-on-coefficients script ===")
println("SIMULATION = $SIMULATION")
println("PARAM_SUMMARY = $PARAM_SUMMARY")
println("UPDATED_ONLY = $UPDATED_ONLY")
println("BETA_POS_ONLY = $BETA_POS_ONLY")
println("BETA_POS_THRESHOLD = $BETA_POS_THRESHOLD")
println("STANDARDIZE_REGRESSION_PREDICTORS = $STANDARDIZE_REGRESSION_PREDICTORS")
println("ANNOTATE_REGIONS = $ANNOTATE_REGIONS")
println("REGIONS_TO_ANNOTATE = $(REGIONS_TO_ANNOTATE)")

gene_data    = CSV.read(GENE_DATA_CSV, DataFrame)
gene_regions = String.(gene_data[:, 1])
gene_IDs     = names(gene_data)[2:end]
gene_matrix  = Matrix(gene_data[:, 2:end])

inference = load_inference("simulations/" * SIMULATION * ".jls")

nonzero_nodes   = nonzero_regions(inference["data"], eps=ZERO_THRESHOLD)
chain           = inference["chain"]
priors          = inference["priors"]
model_regions   = String.(inference["labels"])
parameter_names = collect(keys(priors))

beta_ind  = findall(k -> startswith(k, "beta["), parameter_names)
gamma_ind = findall(k -> startswith(k, "gamma["), parameter_names)

beta_updated_flags  = UPDATED_ONLY ? region_update_flags(inference, "beta";  alpha=UPDATE_ALPHA) : Dict{String,Int}()
gamma_updated_flags = UPDATED_ONLY ? region_update_flags(inference, "gamma"; alpha=UPDATE_ALPHA) : Dict{String,Int}()

region_map = Dict{String, Vector{String}}()
for gr in gene_regions
    gr_str = String(gr)
    L = length(gr_str)
    matches = String[]

    for (i, mr) in enumerate(model_regions)
        if SKIP_ZERO_REGIONS && !(i in nonzero_nodes)
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

model_index = Dict(model_regions[i] => i for i in eachindex(model_regions))

# ============================================================
# PARAMETER SUMMARIES
# ============================================================

beta_post  = Array(chain[:, beta_ind, :])
gamma_post = Array(chain[:, gamma_ind, :])

beta_mean   = vec(mean(beta_post,  dims=1))
gamma_mean  = vec(mean(gamma_post, dims=1))

beta_median  = vec(median(beta_post,  dims=1))
gamma_median = vec(median(gamma_post, dims=1))

lp_mat = Array(chain[:lp])
imax   = argmax(lp_mat)
idx    = CartesianIndices(lp_mat)[imax]
i_idx  = idx[1]
c_idx  = idx[2]

println("MAP sample = $(i_idx), chain = $(c_idx), lp = $(lp_mat[i_idx, c_idx])")

beta_map  = vec(Array(chain[i_idx, beta_ind,  c_idx]))
gamma_map = vec(Array(chain[i_idx, gamma_ind, c_idx]))

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

# ============================================================
# BUILD REGIONAL beta/gamma VECTORS ALIGNED TO GENE REGIONS
# ============================================================

beta_vec_joint  = pick_summary("beta")
gamma_vec_joint = pick_summary("gamma")

beta_gene, beta_n_match, beta_n_after = build_par_gene(
    beta_vec_joint, gene_regions, region_map, model_index;
    updated_flags = UPDATED_ONLY ? beta_updated_flags : nothing
)

gamma_gene, gamma_n_match, gamma_n_after = build_par_gene(
    gamma_vec_joint, gene_regions, region_map, model_index;
    updated_flags = UPDATED_ONLY ? gamma_updated_flags : nothing
)

println("Regions with beta matches before filter:  ", count(>(0), beta_n_match))
println("Regions with gamma matches before filter: ", count(>(0), gamma_n_match))
if UPDATED_ONLY
    println("Regions with beta matches after UPDATED filter:  ", count(>(0), beta_n_after))
    println("Regions with gamma matches after UPDATED filter: ", count(>(0), gamma_n_after))
end

# ============================================================
# OPTIONAL FILTER: beta > threshold
# ============================================================

beta_pos_keep_mask = trues(length(beta_gene))
if BETA_POS_ONLY
    beta_pos_keep_mask .= .!isnan.(beta_gene) .& (beta_gene .> BETA_POS_THRESHOLD)
    println("Applying beta-positive filter: beta > $BETA_POS_THRESHOLD")
    println("Regions kept after beta-positive filter: ", count(beta_pos_keep_mask), " / ", length(beta_pos_keep_mask))

    beta_gene  = [beta_pos_keep_mask[i] ? beta_gene[i]  : NaN for i in eachindex(beta_gene)]
    gamma_gene = [beta_pos_keep_mask[i] ? gamma_gene[i] : NaN for i in eachindex(gamma_gene)]
else
    println("No beta-positive filter applied.")
end

# standardized predictors for regression
beta_reg  = STANDARDIZE_REGRESSION_PREDICTORS ? zscore_with_nans(beta_gene)  : Float64.(beta_gene)
gamma_reg = STANDARDIZE_REGRESSION_PREDICTORS ? zscore_with_nans(gamma_gene) : Float64.(gamma_gene)

# means/sds of ORIGINAL regional parameter vectors
valid_region_mask = .!isnan.(beta_gene) .& .!isnan.(gamma_gene)
count(valid_region_mask) > 0 || error("No valid regions remain after filtering.")

μβ = mean(beta_gene[valid_region_mask])
μγ = mean(gamma_gene[valid_region_mask])
σβ = std(beta_gene[valid_region_mask])
σγ = std(gamma_gene[valid_region_mask])

println("Regional parameter stats used for plotting transform:")
println("  mean(beta)  = $μβ")
println("  mean(gamma) = $μγ")
println("  std(beta)   = $σβ")
println("  std(gamma)  = $σγ")

valid_region_labels = gene_regions[valid_region_mask]

# ============================================================
# GENE-WISE MULTIPLE REGRESSION: gene ~ beta + gamma
# ============================================================

R = length(gene_regions)
G = size(gene_matrix, 2)

beta_coef_vec      = fill(Float64(NaN), G)
gamma_coef_vec     = fill(Float64(NaN), G)
intercept_vec      = fill(Float64(NaN), G)
beta_p_vec         = fill(Float64(NaN), G)
gamma_p_vec        = fill(Float64(NaN), G)
intercept_p_vec    = fill(Float64(NaN), G)
r2_vec             = fill(Float64(NaN), G)
n_used_vec         = zeros(Int, G)

println("\nRunning gene-wise multiple regression: gene ~ beta + gamma ...")

for g in 1:G
    expr = Float64.(gene_matrix[:, g])
    mask = .!isnan.(beta_reg) .& .!isnan.(gamma_reg) .& .!isnan.(expr)
    n_used_vec[g] = sum(mask)

    if n_used_vec[g] < 4
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

suffix  = common_suffix(SKIP_ZERO_REGIONS, UPDATED_ONLY, BETA_POS_ONLY)
reg_sfx = regression_suffix()

df_beta = DataFrame(
    gene   = gene_IDs,
    r      = Float32.(beta_coef_vec),
    p_un   = Float32.(beta_p_vec),
    p_bonf = Float32.(beta_p_bonf),
    p_fdr  = Float32.(beta_p_fdr),
    n_used = n_used_vec
)

df_gamma = DataFrame(
    gene   = gene_IDs,
    r      = Float32.(gamma_coef_vec),
    p_un   = Float32.(gamma_p_vec),
    p_bonf = Float32.(gamma_p_bonf),
    p_fdr  = Float32.(gamma_p_fdr),
    n_used = n_used_vec
)

beta_csv  = joinpath(OUTCSV, "$(SIMULATION)_gene_corr_beta_multireg$(reg_sfx)_$(String(PARAM_SUMMARY))$(suffix).csv")
gamma_csv = joinpath(OUTCSV, "$(SIMULATION)_gene_corr_gamma_multireg$(reg_sfx)_$(String(PARAM_SUMMARY))$(suffix).csv")

CSV.write(beta_csv, df_beta)
CSV.write(gamma_csv, df_gamma)

println("Saved beta coefficient CSV  → $beta_csv")
println("Saved gamma coefficient CSV → $gamma_csv")

save_sig_gene_lists(
    joinpath(OUTTXT, "$(SIMULATION)_siggenes_beta_multireg$(reg_sfx)_$(String(PARAM_SUMMARY))$(suffix)"),
    gene_IDs, beta_p_vec, beta_p_bonf, beta_p_fdr
)

save_sig_gene_lists(
    joinpath(OUTTXT, "$(SIMULATION)_siggenes_gamma_multireg$(reg_sfx)_$(String(PARAM_SUMMARY))$(suffix)"),
    gene_IDs, gamma_p_vec, gamma_p_bonf, gamma_p_fdr
)

# ============================================================
# PCA ON (coef_beta, coef_gamma)
# ============================================================

coef_mask = .!isnan.(beta_coef_vec) .& .!isnan.(gamma_coef_vec)
coef_mat  = hcat(beta_coef_vec[coef_mask], gamma_coef_vec[coef_mask])

pca = pca_2d(coef_mat)

pc1 = copy(pca.eigvecs[:, 1])
pc2 = copy(pca.eigvecs[:, 2])

if FORCE_PC1_POSITIVE_GAMMA && pc1[2] < 0
    pc1 .*= -1
    pc2 .*= -1
end

if FORCE_PC2_POSITIVE_BETA_PLUS_GAMMA && (pc2[1] + pc2[2] < 0)
    pc2 .*= -1
end

println("\nPCA on coefficient cloud:")
println("  PC1 = ($(pc1[1]), $(pc1[2]))")
println("  PC2 = ($(pc2[1]), $(pc2[2]))")
println("  explained variance ratio = ", pca.explained_var_ratio)

# ============================================================
# RAW-SPACE AFFINE FORMS FOR PC1 / PC2
# If eta = a*z(beta) + b*z(gamma), then
# eta = (a/σβ)*beta + (b/σγ)*gamma + intercept
# ============================================================

pc1_beta_raw  = pc1[1] / σβ
pc1_gamma_raw = pc1[2] / σγ
pc2_beta_raw  = pc2[1] / σβ
pc2_gamma_raw = pc2[2] / σγ

pc1_intercept_raw = -(pc1[1] * μβ / σβ + pc1[2] * μγ / σγ)
pc2_intercept_raw = -(pc2[1] * μβ / σβ + pc2[2] * μγ / σγ)

println("\nEquivalent raw-space axes:")
println("  eta  = $(pc1_beta_raw) * beta + $(pc1_gamma_raw) * gamma + $(pc1_intercept_raw)")
println("  eta2 = $(pc2_beta_raw) * beta + $(pc2_gamma_raw) * gamma + $(pc2_intercept_raw)")

println("\nRaw-space directions only (ignoring intercept):")
println("  eta  direction  ∝ ($(pc1_beta_raw), $(pc1_gamma_raw))")
println("  eta2 direction ∝ ($(pc2_beta_raw), $(pc2_gamma_raw))")

pca_summary = DataFrame(
    component = ["PC1", "PC2"],
    beta_loading = Float64[pc1[1], pc2[1]],
    gamma_loading = Float64[pc1[2], pc2[2]],
    beta_loading_raw = Float64[pc1_beta_raw, pc2_beta_raw],
    gamma_loading_raw = Float64[pc1_gamma_raw, pc2_gamma_raw],
    intercept_raw = Float64[pc1_intercept_raw, pc2_intercept_raw],
    eigenvalue = pca.eigvals,
    explained_variance_ratio = pca.explained_var_ratio
)

pca_csv = joinpath(OUTCSV, "$(SIMULATION)_pca_on_beta_gamma_coefficients$(reg_sfx)_$(String(PARAM_SUMMARY))$(suffix).csv")
CSV.write(pca_csv, pca_summary)
println("Saved PCA summary → $pca_csv")

open(joinpath(OUTTXT, "$(SIMULATION)_pca_axis_summary$(reg_sfx)_$(String(PARAM_SUMMARY))$(suffix).txt"), "w") do io
    println(io, "PCA on coefficient cloud")
    println(io, "simulation = $SIMULATION")
    println(io, "PARAM_SUMMARY = $PARAM_SUMMARY")
    println(io, "UPDATED_ONLY = $UPDATED_ONLY")
    println(io, "BETA_POS_ONLY = $BETA_POS_ONLY")
    println(io, "BETA_POS_THRESHOLD = $BETA_POS_THRESHOLD")
    println(io, "STANDARDIZE_REGRESSION_PREDICTORS = $STANDARDIZE_REGRESSION_PREDICTORS")
    println(io)
    println(io, "PC1 beta loading  = ", pc1[1])
    println(io, "PC1 gamma loading = ", pc1[2])
    println(io, "PC2 beta loading  = ", pc2[1])
    println(io, "PC2 gamma loading = ", pc2[2])
    println(io)
    println(io, "PC1 raw beta coefficient  = ", pc1_beta_raw)
    println(io, "PC1 raw gamma coefficient = ", pc1_gamma_raw)
    println(io, "PC1 raw intercept         = ", pc1_intercept_raw)
    println(io)
    println(io, "PC2 raw beta coefficient  = ", pc2_beta_raw)
    println(io, "PC2 raw gamma coefficient = ", pc2_gamma_raw)
    println(io, "PC2 raw intercept         = ", pc2_intercept_raw)
    println(io)
    println(io, "PC1 explained variance ratio = ", pca.explained_var_ratio[1])
    println(io, "PC2 explained variance ratio = ", pca.explained_var_ratio[2])
    println(io)
    println(io, "Chosen eta axis:")
    println(io, "eta = $(pc1[1]) * z(beta) + $(pc1[2]) * z(gamma)")
    println(io, "eta = $(pc1_beta_raw) * beta + $(pc1_gamma_raw) * gamma + $(pc1_intercept_raw)")
    println(io, "(sign chosen so that increasing eta gives increasing gamma)")
    println(io)
    println(io, "Chosen eta2 axis:")
    println(io, "eta2 = $(pc2[1]) * z(beta) + $(pc2[2]) * z(gamma)")
    println(io, "eta2 = $(pc2_beta_raw) * beta + $(pc2_gamma_raw) * gamma + $(pc2_intercept_raw)")
end

# ============================================================
# PLOT 1: COEFFICIENT SCATTER WITH PC1 AND PC2
# ============================================================

pearson_coef_r = cor(coef_mat[:, 1], coef_mat[:, 2])
pearson_coef_p = pvalue(CorrelationTest(coef_mat[:, 1], coef_mat[:, 2]))
spear_coef_r   = cor(tiedrank(coef_mat[:, 1]), tiedrank(coef_mat[:, 2]))
spear_coef_p   = pvalue(CorrelationTest(tiedrank(coef_mat[:, 1]), tiedrank(coef_mat[:, 2])))

xcoef = coef_mat[:, 1]
ycoef = coef_mat[:, 2]

coef_plt = scatter(
    xcoef, ycoef;
    xlabel = STANDARDIZE_REGRESSION_PREDICTORS ? L"\mathrm{gene\ coefficient\ for}\ z(\beta)" : L"\mathrm{gene\ coefficient\ for}\ \beta",
    ylabel = STANDARDIZE_REGRESSION_PREDICTORS ? L"\mathrm{gene\ coefficient\ for}\ z(\gamma)" : L"\mathrm{gene\ coefficient\ for}\ \gamma",
    markersize = 12,
    markercolor = RGB(0.25,0.35,0.55),
    markeralpha = 0.5,
    markerstrokewidth = 0.5,
    label = false,
    title = "",
    size = (700, 700),
    guidefontsize = 30,
    tickfontsize = 15,
    legendfontsize = 18,
    legend = :topright,
    left_margin = 4mm,
    right_margin = 4mm,
    top_margin = 4mm,
    bottom_margin = 4mm,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
)

hline!(coef_plt, [0.0]; color=:gray, linestyle=:dash, lw=2, label=false)
vline!(coef_plt, [0.0]; color=:gray, linestyle=:dash, lw=2, label=false)

plot_pca_lines!(
    coef_plt,
    pca.mean_vec,
    pc1,
    pc2,
    extrema(xcoef),
    extrema(ycoef);
    scale1 = 0.34,
    scale2 = 0.26,
    label1 = "PC1",
    label2 = "PC2"
)

ann = "Pearson r = $(round(pearson_coef_r, sigdigits=3))  (p = $(format_p(pearson_coef_p)))\n" *
      "Spearman ρ = $(round(spear_coef_r, sigdigits=3))  (p = $(format_p(spear_coef_p)))\n" *
      "PC1 var = $(round(100 * pca.explained_var_ratio[1], digits=1))%\n" *
      "n = $(size(coef_mat,1))"

xmin, xmax = extrema(xcoef)
ymin, ymax = extrema(ycoef)
xann = xmin + 0.03 * (xmax - xmin + eps())
yann = ymax - 0.03 * (ymax - ymin + eps())

#annotate!(coef_plt, (xann, yann, text(ann, 9, :black, :left, :top)))

coef_plot_pdf = joinpath(
    OUTFIG,
    "$(SIMULATION)_beta_gamma_coeff_scatter_with_pca$(reg_sfx)_$(String(PARAM_SUMMARY))$(suffix).pdf"
)
savefig(coef_plt, coef_plot_pdf)
println("Saved coefficient PCA plot → $coef_plot_pdf")

# ============================================================
# ETA AXES
# eta  = pc1[1] * z(beta) + pc1[2] * z(gamma)
# eta2 = pc2[1] * z(beta) + pc2[2] * z(gamma)
# also compute equivalent raw-space eta explicitly
# ============================================================

eta_region  = fill(Float64(NaN), R)
eta2_region = fill(Float64(NaN), R)

eta_valid_mask = .!isnan.(beta_reg) .& .!isnan.(gamma_reg)

eta_region[eta_valid_mask]  = pc1[1] .* beta_reg[eta_valid_mask] .+ pc1[2] .* gamma_reg[eta_valid_mask]
eta2_region[eta_valid_mask] = pc2[1] .* beta_reg[eta_valid_mask] .+ pc2[2] .* gamma_reg[eta_valid_mask]

eta_region_raw  = fill(Float64(NaN), R)
eta2_region_raw = fill(Float64(NaN), R)

eta_region_raw[valid_region_mask] = affine_score_raw(
    beta_gene[valid_region_mask],
    gamma_gene[valid_region_mask],
    pc1_beta_raw,
    pc1_gamma_raw,
    pc1_intercept_raw
)

eta2_region_raw[valid_region_mask] = affine_score_raw(
    beta_gene[valid_region_mask],
    gamma_gene[valid_region_mask],
    pc2_beta_raw,
    pc2_gamma_raw,
    pc2_intercept_raw
)

ηmaxdiff  = maximum(abs.(eta_region[valid_region_mask] .- eta_region_raw[valid_region_mask]))
η2maxdiff = maximum(abs.(eta2_region[valid_region_mask] .- eta2_region_raw[valid_region_mask]))

println("\nConsistency checks:")
println("  max |eta(z-space)  - eta(raw-space)|  = $ηmaxdiff")
println("  max |eta2(z-space) - eta2(raw-space)| = $η2maxdiff")

region_eta_df = DataFrame(
    gene_region = gene_regions,
    beta = beta_gene,
    gamma = gamma_gene,
    z_beta = beta_reg,
    z_gamma = gamma_reg,
    eta = eta_region,
    eta_raw = eta_region_raw,
    eta2 = eta2_region,
    eta2_raw = eta2_region_raw,
    beta_pos_keep = Int.(beta_pos_keep_mask),
    n_beta_matches_before = beta_n_match,
    n_beta_matches_after = beta_n_after,
    n_gamma_matches_before = gamma_n_match,
    n_gamma_matches_after = gamma_n_after
)

region_eta_csv = joinpath(OUTCSV, "$(SIMULATION)_regional_beta_gamma_eta$(reg_sfx)_$(String(PARAM_SUMMARY))$(suffix).csv")
CSV.write(region_eta_csv, region_eta_df)
println("Saved region-level eta table → $region_eta_csv")

# ============================================================
# GENE CORRELATION WITH ETA FOR GSEA
# ============================================================

println("\nComputing gene correlations with eta ...")

eta_df, eta_r_vec, eta_p_vec, eta_p_bonf, eta_p_fdr, eta_n_used_vec =
    gene_correlations_with_score(eta_region, gene_matrix, gene_IDs)

eta_csv = joinpath(OUTCSV, "$(SIMULATION)_gene_corr_eta_from_pca_beta_gamma$(reg_sfx)_$(String(PARAM_SUMMARY))$(suffix).csv")
CSV.write(eta_csv, eta_df)
println("Saved eta gene-correlation CSV → $eta_csv")

save_sig_gene_lists(
    joinpath(OUTTXT, "$(SIMULATION)_siggenes_eta_from_pca_beta_gamma$(reg_sfx)_$(String(PARAM_SUMMARY))$(suffix)"),
    gene_IDs, eta_p_vec, eta_p_bonf, eta_p_fdr
)

# ============================================================
# GENE CORRELATION WITH ETA2 FOR GSEA
# ============================================================

println("\nComputing gene correlations with eta2 ...")

eta2_df, eta2_r_vec, eta2_p_vec, eta2_p_bonf, eta2_p_fdr, eta2_n_used_vec =
    gene_correlations_with_score(eta2_region, gene_matrix, gene_IDs)

eta2_csv = joinpath(OUTCSV, "$(SIMULATION)_gene_corr_eta2_from_pca_beta_gamma$(reg_sfx)_$(String(PARAM_SUMMARY))$(suffix).csv")
CSV.write(eta2_csv, eta2_df)
println("Saved eta2 gene-correlation CSV → $eta2_csv")

save_sig_gene_lists(
    joinpath(OUTTXT, "$(SIMULATION)_siggenes_eta2_from_pca_beta_gamma$(reg_sfx)_$(String(PARAM_SUMMARY))$(suffix)"),
    gene_IDs, eta2_p_vec, eta2_p_bonf, eta2_p_fdr
)

# ============================================================
# PLOT 2: ORIGINAL beta-gamma REGIONAL SCATTER WITH PC1/PC2
# ============================================================

pc1_orig = [pc1[1] * σβ, pc1[2] * σγ]
pc2_orig = [pc2[1] * σβ, pc2[2] * σγ]

pc1_orig ./= norm(pc1_orig)
pc2_orig ./= norm(pc2_orig)

βv = beta_gene[valid_region_mask]
γv = gamma_gene[valid_region_mask]
ηv_raw = eta_region_raw[valid_region_mask]

eta_clims = extrema(ηv_raw)

region_plt = scatter(
    βv, γv;
    marker_z = ηv_raw,
    clims = eta_clims,
    xlabel = "beta",
    ylabel = "gamma",
    colorbar_title = "eta",
    markersize = 5,
    alpha = 0.85,
    label = false,
    title = "",
    guidefontsize = 15,
    top_margin = 9mm
)

pearson_bg_r = cor(βv, γv)
pearson_bg_p = pvalue(CorrelationTest(βv, γv))
spear_bg_r   = cor(tiedrank(βv), tiedrank(γv))
spear_bg_p   = pvalue(CorrelationTest(tiedrank(βv), tiedrank(γv)))

plot_pca_lines!(
    region_plt,
    [μβ, μγ],
    pc1_orig,
    pc2_orig,
    extrema(βv),
    extrema(γv);
    label1 = "PC1 direction",
    label2 = "PC2 direction"
)

ann2 = "Pearson r = $(round(pearson_bg_r, sigdigits=3))  (p = $(format_p(pearson_bg_p)))\n" *
       "Spearman ρ = $(round(spear_bg_r, sigdigits=3))  (p = $(format_p(spear_bg_p)))\n" *
       "PC1 raw direction ∝ ($(round(pc1_beta_raw, sigdigits=3)), $(round(pc1_gamma_raw, sigdigits=3)))\n" *
       "PC2 raw direction ∝ ($(round(pc2_beta_raw, sigdigits=3)), $(round(pc2_gamma_raw, sigdigits=3)))\n" *
       "n = $(length(βv))"

xmin2, xmax2 = extrema(βv)
ymin2, ymax2 = extrema(γv)
xann2 = xmin2 + 0.03 * (xmax2 - xmin2 + eps())
yann2 = ymax2 - 0.03 * (ymax2 - ymin2 + eps())
annotate!(region_plt, (xann2, yann2, text(ann2, 9, :black, :left, :top)))

if ANNOTATE_REGIONS
    annotate_selected_regions!(
        region_plt,
        βv,
        γv,
        valid_region_labels;
        selected_regions = REGIONS_TO_ANNOTATE,
        fontsize = REGION_ANNOTATION_FONTSIZE
    )
end

region_plot_pdf = joinpath(OUTFIG, "$(SIMULATION)_regional_beta_gamma_scatter_with_eta_pca_direction$(reg_sfx)_$(String(PARAM_SUMMARY))$(suffix).pdf")
savefig(region_plt, region_plot_pdf)
println("Saved regional beta-gamma PCA-direction plot → $region_plot_pdf")

# ============================================================
# PLOT 3: ORIGINAL beta-gamma REGIONAL SCATTER WITH ETA HEATMAP
# ============================================================

βmin, βmax = extrema(βv)
γmin, γmax = extrema(γv)

βpad = 0.05 * (βmax - βmin + eps())
γpad = 0.05 * (γmax - γmin + eps())

βgrid = range(βmin - βpad, βmax + βpad, length=300)
γgrid = range(γmin - γpad, γmax + γpad, length=300)

ηgrid = [affine_score_raw(β, γ, pc1_beta_raw, pc1_gamma_raw, pc1_intercept_raw)
         for γ in γgrid, β in βgrid]

ηmin_grid, ηmax_grid = extrema(ηgrid)
eta_clims_plot = (ηmin_grid, ηmax_grid)
contour_levels = collect(range(ηmin_grid, ηmax_grid; length=7))

region_heatmap_plt = heatmap(
    βgrid,
    γgrid,
    ηgrid;
    xlabel = L"\beta",
    ylabel = L"\gamma",
    colorbar_title = L"\eta",
    colorbar_titlefontsize = 22,
    clims = eta_clims_plot,
    c = :balance,
    alpha = 0.78,
    aspect_ratio = :auto,
    label = false,
    title = "",
    guidefontsize = 22,
    top_margin = 9mm,
    right_margin = 9mm,
)

scatter!(
    region_heatmap_plt,
    βv, γv;
    markersize = 10,
    markercolor = :white,
    markeralpha = 0.90,
    markerstrokecolor = :black,
    markerstrokewidth = 2.5,
    label = false
)

contour!(
    region_heatmap_plt,
    βgrid,
    γgrid,
    ηgrid;
    levels = contour_levels,
    color = :black,
    linewidth = 2.0,
    alpha = 0.18,
    label = false
)

if ANNOTATE_REGIONS
    annotate_selected_regions!(
        region_heatmap_plt,
        βv,
        γv,
        valid_region_labels;
        selected_regions = REGIONS_TO_ANNOTATE,
        fontsize = REGION_ANNOTATION_FONTSIZE
    )
end

region_heatmap_pdf = joinpath(
    OUTFIG,
    "$(SIMULATION)_regional_beta_gamma_scatter_with_eta_heatmap$(reg_sfx)_$(String(PARAM_SUMMARY))$(suffix).pdf"
)
savefig(region_heatmap_plt, region_heatmap_pdf)
println("Saved regional beta-gamma heatmap plot → $region_heatmap_pdf")

# ============================================================
# PLOT 4: STANDARDIZED z(beta)-z(gamma) SCATTER WITH ETA HEATMAP
# ============================================================

βzv = beta_reg[valid_region_mask]
γzv = gamma_reg[valid_region_mask]
ηvz = eta_region[valid_region_mask]

βzmin, βzmax = extrema(βzv)
γzmin, γzmax = extrema(γzv)

βzpad = 0.05 * (βzmax - βzmin + eps())
γzpad = 0.05 * (γzmax - γzmin + eps())

βzgrid = range(βzmin - βzpad, βzmax + βzpad, length=300)
γzgrid = range(γzmin - γzpad, γzmax + γzpad, length=300)

ηzgrid = [pc1[1] * βz + pc1[2] * γz for γz in γzgrid, βz in βzgrid]

ηzmin_grid, ηzmax_grid = extrema(ηzgrid)
eta_z_clims_plot = (ηzmin_grid, ηzmax_grid)
contour_levels_z = collect(range(ηzmin_grid, ηzmax_grid; length=7))

region_heatmap_z_plt = heatmap(
    βzgrid,
    γzgrid,
    ηzgrid;
    xlabel = L"z(\beta)",
    ylabel = L"z(\gamma)",
    legendfontsize = 18,
    colorbar_title = L"\eta",
    colorbar_titlefontsize = 30,
    clims = eta_z_clims_plot,
    c = :balance,
    alpha = 0.78,
    aspect_ratio = :equal,
    size = (700, 700),
    guidefontsize = 30,
    tickfontsize = 20,
    left_margin = 4mm,
    right_margin = 8mm,
    top_margin = 4mm,
    bottom_margin = -8mm,
    label = false,
    title = "",
)

scatter!(
    region_heatmap_z_plt,
    βzv, γzv;
    markersize = 20,
    markercolor = :white,
    markeralpha = 0.90,
    markerstrokecolor = :black,
    markerstrokewidth = 1.5,
    label = false
)

contour!(
    region_heatmap_z_plt,
    βzgrid,
    γzgrid,
    ηzgrid;
    levels = contour_levels_z,
    color = :black,
    linewidth = 2.0,
    alpha = 0.18,
    label = false
)

plot_pca_lines!(
    region_heatmap_z_plt,
    [0.0, 0.0],
    pc1 ./ norm(pc1),
    pc2 ./ norm(pc2),
    extrema(βzgrid),
    extrema(γzgrid);
    scale1 = 0.28,
    scale2 = 0.22,
    label1 = "PC1",
    label2 = "PC2"
)

xlims!(region_heatmap_z_plt, first(βzgrid), last(βzgrid))
ylims!(region_heatmap_z_plt, first(γzgrid), last(γzgrid))

if ANNOTATE_REGIONS
    annotate_selected_regions!(
        region_heatmap_z_plt,
        βzv,
        γzv,
        valid_region_labels;
        selected_regions = REGIONS_TO_ANNOTATE,
        fontsize = REGION_ANNOTATION_FONTSIZE
    )
end

region_heatmap_z_pdf = joinpath(
    OUTFIG,
    "$(SIMULATION)_regional_zbeta_zgamma_scatter_with_eta_heatmap$(reg_sfx)_$(String(PARAM_SUMMARY))$(suffix).pdf"
)
savefig(region_heatmap_z_plt, region_heatmap_z_pdf)
println("Saved regional zbeta-zgamma heatmap plot → $region_heatmap_z_pdf")

# ============================================================
# FINAL SUMMARY FILE
# ============================================================

open(joinpath(OUTTXT, "$(SIMULATION)_run_summary$(reg_sfx)_$(String(PARAM_SUMMARY))$(suffix).txt"), "w") do io
    println(io, "Run summary")
    println(io, "simulation = $SIMULATION")
    println(io, "PARAM_SUMMARY = $PARAM_SUMMARY")
    println(io, "UPDATED_ONLY = $UPDATED_ONLY")
    println(io, "BETA_POS_ONLY = $BETA_POS_ONLY")
    println(io, "BETA_POS_THRESHOLD = $BETA_POS_THRESHOLD")
    println(io, "STANDARDIZE_REGRESSION_PREDICTORS = $STANDARDIZE_REGRESSION_PREDICTORS")
    println(io, "ANNOTATE_REGIONS = $ANNOTATE_REGIONS")
    println(io, "REGIONS_TO_ANNOTATE = $(REGIONS_TO_ANNOTATE)")
    println(io)
    println(io, "Chosen eta:")
    println(io, "eta = $(pc1[1]) * z(beta) + $(pc1[2]) * z(gamma)")
    println(io, "eta = $(pc1_beta_raw) * beta + $(pc1_gamma_raw) * gamma + $(pc1_intercept_raw)")
    println(io, "(sign chosen so increasing eta gives increasing gamma)")
    println(io)
    println(io, "Chosen eta2:")
    println(io, "eta2 = $(pc2[1]) * z(beta) + $(pc2[2]) * z(gamma)")
    println(io, "eta2 = $(pc2_beta_raw) * beta + $(pc2_gamma_raw) * gamma + $(pc2_intercept_raw)")
    println(io)
    println(io, "Explained variance:")
    println(io, "  PC1 = ", pca.explained_var_ratio[1])
    println(io, "  PC2 = ", pca.explained_var_ratio[2])
    println(io)
    println(io, "Consistency checks:")
    println(io, "  max |eta(z-space)  - eta(raw-space)|  = $ηmaxdiff")
    println(io, "  max |eta2(z-space) - eta2(raw-space)| = $η2maxdiff")
    println(io)
    println(io, "Main output files:")
    println(io, "  beta coeff CSV   = $beta_csv")
    println(io, "  gamma coeff CSV  = $gamma_csv")
    println(io, "  eta corr CSV     = $eta_csv")
    println(io, "  eta2 corr CSV    = $eta2_csv")
    println(io, "  PCA summary CSV  = $pca_csv")
    println(io, "  region eta CSV   = $region_eta_csv")
    println(io, "  coeff plot       = $coef_plot_pdf")
    println(io, "  region plot      = $region_plot_pdf")
    println(io, "  region heatmap   = $region_heatmap_pdf")
    println(io, "  z-space heatmap  = $region_heatmap_z_pdf")
end

println("\n=== Done ===")
println("Outputs saved under: $OUTROOT")

println("pc1 standardized = ", pc1)
println("raw eta coefficients (A,B) = ", (pc1_beta_raw, pc1_gamma_raw))

println("PCA displacement slope in raw space = ", (pc1[2] * σγ) / (pc1[1] * σβ))
println("eta gradient slope in raw space     = ", pc1_gamma_raw / pc1_beta_raw)
println("eta contour slope in raw space      = ", -pc1_beta_raw / pc1_gamma_raw)

threshold_gamma = 0.45
threshold_beta  = 0.25   # near zero

mask = (abs.(beta_coef_vec) .< threshold_beta) .& (gamma_coef_vec .> threshold_gamma)
idx = findall(mask)

println("Genes with beta ≈ 0 and large gamma:\n")

for i in sort(idx, by = i -> -gamma_coef_vec[i])
    println(gene_IDs[i])
end