#!/usr/bin/env julia

using PathoSpread
using CSV
using DataFrames
using Statistics
using LinearAlgebra
using HypothesisTests
using MCMCChains
using Plots
using Measures
using LaTeXStrings
using Random
using Distributions

const MAPT_NAME = "syn_mapt_DIFFGA_RETRO"
const APP_NAME = "syn_app_DIFFGA_RETRO"
const GENE_DATA_CSV = "data/avg_Pangea_exp.csv"
const OUTPUT_ROOT = get(ENV, "COPATHOLOGY_GENE_AXIS_OUTROOT", "figures/copathology_updated/gene_axis")
const FIG_DIR = OUTPUT_ROOT
const CSV_DIR = joinpath(OUTPUT_ROOT, "csv")

const PARAM_SUMMARY = :mean
const SKIP_ZERO_REGIONS = false
const ZERO_THRESHOLD = 0.01
const UPDATED_ONLY = get(ENV, "COPATHOLOGY_GENE_AXIS_UPDATED_ONLY", "true") == "true"
const UPDATE_ALPHA = 0.001
const BETA_POS_ONLY = get(ENV, "COPATHOLOGY_GENE_AXIS_BETA_POS_ONLY", "true") == "true"
const BETA_POS_THRESHOLD = 0.0
const EXCLUDE_SEED_SITES = get(ENV, "COPATHOLOGY_GENE_AXIS_EXCLUDE_SEEDS", "false") == "true"
const SEED_SITE_BASES = Set(["CA1", "CA3", "DG"])
const USE_HEMISPHERE_DUPLICATION = true
const STANDARDIZE_REGRESSION_PREDICTORS = true
const FORCE_PC1_POSITIVE_GAMMA = true
const FORCE_PC2_POSITIVE_BETA_PLUS_GAMMA = true

const PLOT_SIZE = (900, 700)
const PCA_LINEWIDTH = 10
const DIAG_LINEWIDTH = 3
const SCATTERSIZE_COEF = 8
const SCATTERSIZE_REGION = 10
const MARKERSTROKEWIDTH = 0.3
const CONTRAST_DRAWS = 5000


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


function format_p(p::Real)
    if isnan(p)
        return "NaN"
    elseif p < 1e-4
        return "<1e-4"
    else
        return string(round(p, sigdigits=3))
    end
end


function zscore_with_nans(x::AbstractVector)
    out = fill(Float64(NaN), length(x))
    mask = .!isnan.(x)
    any(mask) || return out

    μ = mean(x[mask])
    σ = std(x[mask])
    if !isfinite(σ) || σ == 0
        out[mask] .= 0.0
    else
        out[mask] = (x[mask] .- μ) ./ σ
    end
    return out
end


function posterior_samples_from_p(chain::Chains, i::Int)
    for sym in (Symbol("p[$i]"), Symbol("p__$(i)"), Symbol("p_$(i)"), Symbol("p.$(i)"))
        if sym in names(chain)
            return vec(Array(chain[sym]))
        end
    end
    error("Could not find samples for p[$i].")
end


function parameter_names_by_prefix(priors, prefix::AbstractString)
    names_ = filter(n -> startswith(n, prefix * "["), collect(keys(priors)))
    sort!(names_, by = n -> parse(Int, match(Regex("^" * prefix * "\\[(\\d+)\\]"), n).captures[1]))
    return names_
end


function posterior_vs_prior_update_flag(samples::AbstractVector, prior_dist; alpha::Float64=UPDATE_ALPHA)
    p = pvalue(ApproximateOneSampleKSTest(samples, prior_dist))
    return p, Int(p < alpha)
end


function region_update_flags(inference::Dict, prefix::AbstractString; alpha::Float64=UPDATE_ALPHA)
    priors = inference["priors"]
    chain = inference["chain"]
    regions = String.(inference["labels"])
    param_names = parameter_names_by_prefix(priors, prefix)
    prior_keys = collect(keys(priors))
    length(param_names) == length(regions) || error("Mismatch between regions and $prefix parameters.")

    flags = Dict{String, Int}()
    for (region, pname) in zip(regions, param_names)
        idx = findfirst(==(pname), prior_keys)
        idx === nothing && error("Missing $pname in priors.")
        samples = posterior_samples_from_p(chain, idx)
        prior_dist = priors[pname]
        _, updated = posterior_vs_prior_update_flag(samples, prior_dist; alpha=alpha)
        flags[region] = updated
    end
    return flags
end


function build_region_map(gene_regions, model_regions, nonzero_nodes)
    region_map = Dict{String, Vector{String}}()
    for gr in String.(gene_regions)
        L = length(gr)
        matches = String[]
        if EXCLUDE_SEED_SITES && (gr in SEED_SITE_BASES)
            region_map[gr] = matches
            continue
        end
        for (i, mr) in enumerate(String.(model_regions))
            if SKIP_ZERO_REGIONS && !(i in nonzero_nodes)
                continue
            end
            if EXCLUDE_SEED_SITES
                base = (startswith(mr, "i") || startswith(mr, "c")) ? mr[2:end] : mr
                if base in SEED_SITE_BASES
                    continue
                end
            end
            if !(startswith(mr, "i") || startswith(mr, "c"))
                continue
            end
            if length(mr) < 1 + L
                continue
            end
            mr_sub = mr[2:1+L]
            if mr_sub == gr
                if length(mr) == 1 + L
                    push!(matches, mr)
                else
                    next_char = mr[2 + L]
                    if !isletter(next_char)
                        push!(matches, mr)
                    end
                end
            end
        end
        region_map[gr] = matches
    end
    return region_map
end


function build_par_gene(
    par_vec::AbstractVector,
    gene_regions,
    region_map::Dict{String, Vector{String}},
    model_index::Dict{String, Int};
    updated_flags::Union{Nothing, Dict{String, Int}}=nothing,
)
    R = length(gene_regions)
    par_gene = Array{Float64}(undef, R)
    n_model_matches = zeros(Int, R)
    n_model_matches_after_filter = zeros(Int, R)

    for (r, gr) in enumerate(String.(gene_regions))
        regions = copy(region_map[gr])
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


function build_hemi_observations(
    beta_vec::AbstractVector,
    gamma_vec::AbstractVector,
    gene_regions,
    region_map::Dict{String, Vector{String}},
    model_index::Dict{String, Int};
    beta_updated_flags::Union{Nothing, Dict{String, Int}}=nothing,
    gamma_updated_flags::Union{Nothing, Dict{String, Int}}=nothing,
)
    obs_gene_region_idx = Int[]
    obs_gene_region_labels = String[]
    obs_model_region_labels = String[]
    obs_beta = Float64[]
    obs_gamma = Float64[]

    n_model_matches = zeros(Int, length(gene_regions))
    n_beta_after = zeros(Int, length(gene_regions))
    n_gamma_after = zeros(Int, length(gene_regions))
    n_joint_after = zeros(Int, length(gene_regions))

    for (r, gr) in enumerate(String.(gene_regions))
        regions0 = copy(region_map[gr])
        n_model_matches[r] = length(regions0)

        regions_beta = beta_updated_flags === nothing ? copy(regions0) : [mr for mr in regions0 if get(beta_updated_flags, mr, 0) == 1]
        regions_gamma = gamma_updated_flags === nothing ? copy(regions0) : [mr for mr in regions0 if get(gamma_updated_flags, mr, 0) == 1]

        n_beta_after[r] = length(regions_beta)
        n_gamma_after[r] = length(regions_gamma)

        gamma_set = Set(regions_gamma)
        joint_regions = [mr for mr in regions_beta if mr in gamma_set]
        n_joint_after[r] = length(joint_regions)

        for mr in joint_regions
            idx = model_index[mr]
            push!(obs_gene_region_idx, r)
            push!(obs_gene_region_labels, gr)
            push!(obs_model_region_labels, mr)
            push!(obs_beta, beta_vec[idx])
            push!(obs_gamma, gamma_vec[idx])
        end
    end

    return (
        gene_region_idx = obs_gene_region_idx,
        gene_region_labels = obs_gene_region_labels,
        model_region_labels = obs_model_region_labels,
        beta = obs_beta,
        gamma = obs_gamma,
        n_model_matches = n_model_matches,
        n_beta_after = n_beta_after,
        n_gamma_after = n_gamma_after,
        n_joint_after = n_joint_after,
    )
end


function ols_two_predictors(y::AbstractVector, x1::AbstractVector, x2::AbstractVector)
    X = hcat(ones(length(y)), x1, x2)
    βhat = X \ y
    yhat = X * βhat
    resid = y - yhat
    dof = length(y) - size(X, 2)
    σ2 = sum(abs2, resid) / dof
    XtX_inv = inv(X' * X)
    se = sqrt.(diag(σ2 .* XtX_inv))
    tstats = βhat ./ se
    pvals = [2 * (1 - cdf(TDist(dof), abs(t))) for t in tstats]
    sst = sum(abs2, y .- mean(y))
    sse = sum(abs2, resid)
    r2 = sst > 0 ? 1 - sse / sst : NaN
    return (
        intercept = βhat[1],
        beta_coef = βhat[2],
        gamma_coef = βhat[3],
        intercept_p = pvals[1],
        beta_p = pvals[2],
        gamma_p = pvals[3],
        r2 = r2,
        n_used = length(y),
    )
end


function pca_2d(X::AbstractMatrix)
    μ = vec(mean(X, dims=1))
    Xc = X .- reshape(μ, 1, :)
    Σ = (Xc' * Xc) / (size(X, 1) - 1)
    F = eigen(Symmetric(Σ))
    ord = sortperm(F.values, rev=true)
    eigvals = F.values[ord]
    eigvecs = F.vectors[:, ord]
    explained = eigvals ./ sum(eigvals)
    scores = Xc * eigvecs
    return (mean_vec=μ, eigvecs=eigvecs, eigvals=eigvals, explained_var_ratio=explained, scores=scores)
end


function affine_score_raw(beta_vals, gamma_vals, beta_coef_raw, gamma_coef_raw, intercept_raw)
    return beta_coef_raw .* beta_vals .+ gamma_coef_raw .* gamma_vals .+ intercept_raw
end


function gene_correlations_with_score(score_vec::AbstractVector, gene_matrix, gene_IDs, row_idx_vec::AbstractVector{<:Integer})
    G = size(gene_matrix, 2)
    r_vec = fill(Float64(NaN), G)
    p_vec = fill(Float64(NaN), G)
    n_used_vec = zeros(Int, G)

    for g in 1:G
        expr = Float64.(gene_matrix[row_idx_vec, g])
        mask = .!isnan.(score_vec) .& .!isnan.(expr)
        n_used_vec[g] = sum(mask)
        if n_used_vec[g] < 3
            continue
        end
        r_vec[g] = cor(expr[mask], score_vec[mask])
        p_vec[g] = pvalue(CorrelationTest(expr[mask], score_vec[mask]))
    end

    p_bonf = [isnan(p) ? NaN : min(p * G, 1.0) for p in p_vec]
    p_fdr = fdr_bh(p_vec)
    df = DataFrame(gene=gene_IDs, r=Float32.(r_vec), p_un=Float32.(p_vec), p_bonf=Float32.(p_bonf), p_fdr=Float32.(p_fdr), n_used=n_used_vec)
    return df, r_vec, p_vec, p_fdr
end


function setup_defaults()
    PathoSpread.setup_plot_theme!(font="Arial", base=18, lw=3, markersize=10, dpi=300)
    default(
        size = PLOT_SIZE,
        guidefontsize = 20,
        tickfontsize = 15,
        legendfontsize = 15,
        titlefontsize = 18,
        left_margin = 4mm,
        right_margin = 6mm,
        top_margin = 4mm,
        bottom_margin = 4mm,
    )
end


function coefficient_pca_plot(result, save_path::AbstractString)
    x = result.beta_coef_vec[result.coef_mask]
    y = result.gamma_coef_vec[result.coef_mask]
    pearson_r = cor(x, y)
    pearson_p = pvalue(CorrelationTest(x, y))

    plt = scatter(
        x, y;
        xlabel = STANDARDIZE_REGRESSION_PREDICTORS ? L"\mathrm{gene\ coefficient\ for}\ z(\beta)" : L"\mathrm{gene\ coefficient\ for}\ \beta",
        ylabel = STANDARDIZE_REGRESSION_PREDICTORS ? L"\mathrm{gene\ coefficient\ for}\ z(\gamma)" : L"\mathrm{gene\ coefficient\ for}\ \gamma",
        markercolor = :gray20,
        markeralpha = 0.18,
        markersize = SCATTERSIZE_COEF,
        markerstrokewidth = MARKERSTROKEWIDTH,
        label = false,
        legend = :bottomleft,
        title = result.name,
    )

    hline!(plt, [0.0]; color=:gray65, linestyle=:dash, lw=2, label=false)
    vline!(plt, [0.0]; color=:gray65, linestyle=:dash, lw=2, label=false)

    center = result.pca.mean_vec
    v1 = result.pc1
    v2 = result.pc2
    xr = extrema(x)
    yr = extrema(y)
    xspan = xr[2] - xr[1]
    yspan = yr[2] - yr[1]
    L1 = 0.36 * min(xspan / max(abs(v1[1]), eps()), yspan / max(abs(v1[2]), eps()))
    L2 = 0.28 * min(xspan / max(abs(v2[1]), eps()), yspan / max(abs(v2[2]), eps()))

    p1a = center .- L1 .* v1
    p1b = center .+ L1 .* v1
    p2a = center .- L2 .* v2
    p2b = center .+ L2 .* v2

    plot!(plt, [p2a[1], p2b[1]], [p2a[2], p2b[2]]; lw=PCA_LINEWIDTH, color=:gray45, label="PC2")
    plot!(plt, [p1a[1], p1b[1]], [p1a[2], p1b[2]]; lw=PCA_LINEWIDTH, color=:black, label="PC1")

    ann = "Pearson r = $(round(pearson_r, sigdigits=3)) (p = $(format_p(pearson_p)))\n" *
          "PC1 = ($(round(result.pc1[1], sigdigits=3)), $(round(result.pc1[2], sigdigits=3)))\n" *
          "PC1 var = $(round(100 * result.pca.explained_var_ratio[1], digits=1))%\n" *
          "n = $(length(x)) genes"

    annotate!(plt, (xr[1] + 0.04 * (xspan + eps()), yr[2] - 0.04 * (yspan + eps()), text(ann, 10, :black, :left, :top)))
    savefig(plt, save_path)
end


function eta_heatmap_plot(result, save_path::AbstractString)
    β = result.beta_analysis[result.valid_analysis_mask]
    γ = result.gamma_analysis[result.valid_analysis_mask]
    η = result.eta_analysis_raw[result.valid_analysis_mask]

    βmin, βmax = extrema(β)
    γmin, γmax = extrema(γ)
    βpad = 0.05 * (βmax - βmin + eps())
    γpad = 0.05 * (γmax - γmin + eps())
    βgrid = range(βmin - βpad, βmax + βpad, length=220)
    γgrid = range(γmin - γpad, γmax + γpad, length=220)
    ηgrid = [affine_score_raw(b, g, result.pc1_beta_raw, result.pc1_gamma_raw, result.pc1_intercept_raw) for g in γgrid, b in βgrid]

    plt = heatmap(
        βgrid,
        γgrid,
        ηgrid;
        xlabel = "rise parameter " * string(L"\beta"),
        ylabel = "fall parameter " * string(L"\gamma"),
        colorbar_title = "\nvulnerability axis " * string(L"\eta"),
        c = :balance,
        alpha = 0.78,
        aspect_ratio = :auto,
        title = result.name,
        label = false,
        right_margin = 16mm,
    )

    scatter!(
        plt,
        β,
        γ;
        markercolor = :white,
        markerstrokecolor = :black,
        markerstrokewidth = 1.0,
        markersize = SCATTERSIZE_REGION,
        label = false,
        alpha = 0.9,
    )

    savefig(plt, save_path)
end


function posterior_contrast_df(inf_mapt, inf_app, family::String; draws::Int=CONTRAST_DRAWS, rng::AbstractRNG=MersenneTwister(42))
    priors_a = inf_mapt["priors"]
    priors_b = inf_app["priors"]
    labels = String.(inf_mapt["labels"])
    prior_keys_a = collect(keys(priors_a))
    prior_keys_b = collect(keys(priors_b))
    names_a = parameter_names_by_prefix(priors_a, family)
    names_b = parameter_names_by_prefix(priors_b, family)
    names_a == names_b || error("Parameter families differ for $family.")

    mean_a = PathoSpread.posterior_mean_vector_from_priors(inf_mapt["chain"], priors_a, family, length(labels))
    mean_b = PathoSpread.posterior_mean_vector_from_priors(inf_app["chain"], priors_b, family, length(labels))

    rows = DataFrame(
        region = labels,
        family = fill(family, length(labels)),
        mean_mapt = mean_a,
        mean_app = mean_b,
        delta_app_minus_mapt = mean_b .- mean_a,
        ci_low = zeros(Float64, length(labels)),
        ci_high = zeros(Float64, length(labels)),
        prob_app_gt_mapt = zeros(Float64, length(labels)),
        prob_app_lt_mapt = zeros(Float64, length(labels)),
        p_two_sided = zeros(Float64, length(labels)),
    )

    for (i, pname) in enumerate(names_a)
        idx_a = findfirst(==(pname), prior_keys_a)
        idx_b = findfirst(==(pname), prior_keys_b)
        sa = posterior_samples_from_p(inf_mapt["chain"], idx_a)
        sb = posterior_samples_from_p(inf_app["chain"], idx_b)
        da = rand(rng, sa, draws)
        db = rand(rng, sb, draws)
        diff = db .- da
        prob_gt = mean(diff .> 0)
        prob_lt = mean(diff .< 0)
        rows.ci_low[i] = quantile(diff, 0.025)
        rows.ci_high[i] = quantile(diff, 0.975)
        rows.prob_app_gt_mapt[i] = prob_gt
        rows.prob_app_lt_mapt[i] = prob_lt
        rows.p_two_sided[i] = 2 * min(prob_gt, prob_lt)
    end

    rows.q_fdr = fdr_bh(rows.p_two_sided)
    rows.significant_fdr_05 = rows.q_fdr .< 0.05
    rows.abs_delta = abs.(rows.delta_app_minus_mapt)
    sort!(rows, [:significant_fdr_05, :abs_delta], rev=[true, true])
    return rows
end


function contrast_volcano_plot(df::DataFrame, family::String, save_path::AbstractString)
    x = df.delta_app_minus_mapt
    y = .-log10.(df.q_fdr .+ eps())
    colors = ifelse.(df.significant_fdr_05, :firebrick3, :gray40)

    plt = scatter(
        x, y;
        xlabel = "APP - MAPT posterior mean shift",
        ylabel = L"-\log_{10}(\mathrm{FDR})",
        markercolor = colors,
        markeralpha = 0.8,
        markersize = 7,
        markerstrokewidth = 0.0,
        label = false,
        title = uppercase(family),
    )

    hline!(plt, [-log10(0.05)]; color=:black, linestyle=:dash, lw=2, label="FDR = 0.05")
    vline!(plt, [0.0]; color=:gray65, linestyle=:dash, lw=2, label=false)
    savefig(plt, save_path)
end


function fit_gene_axis(name::String, inference_path::String, gene_data::DataFrame)
    inference = load_inference(inference_path)
    gene_regions = String.(gene_data[:, 1])
    gene_IDs = String.(names(gene_data)[2:end])
    gene_matrix = Matrix(gene_data[:, 2:end])
    model_regions = String.(inference["labels"])
    nonzero_nodes = nonzero_regions(inference["data"], eps=ZERO_THRESHOLD)
    region_map = build_region_map(gene_regions, model_regions, nonzero_nodes)
    model_index = Dict(model_regions[i] => i for i in eachindex(model_regions))

    beta_updated_flags = UPDATED_ONLY ? region_update_flags(inference, "beta"; alpha=UPDATE_ALPHA) : Dict{String, Int}()
    gamma_updated_flags = UPDATED_ONLY ? region_update_flags(inference, "gamma"; alpha=UPDATE_ALPHA) : Dict{String, Int}()

    beta_mean = PathoSpread.posterior_mean_vector_from_priors(inference["chain"], inference["priors"], "beta", length(model_regions))
    gamma_mean = PathoSpread.posterior_mean_vector_from_priors(inference["chain"], inference["priors"], "gamma", length(model_regions))
    beta_vec_joint = beta_mean
    gamma_vec_joint = gamma_mean

    beta_gene, beta_n_match, beta_n_after = build_par_gene(beta_vec_joint, gene_regions, region_map, model_index; updated_flags=UPDATED_ONLY ? beta_updated_flags : nothing)
    gamma_gene, gamma_n_match, gamma_n_after = build_par_gene(gamma_vec_joint, gene_regions, region_map, model_index; updated_flags=UPDATED_ONLY ? gamma_updated_flags : nothing)

    analysis_row_idx = Int[]
    analysis_gene_label = String[]
    analysis_model_label = String[]
    beta_analysis = Float64[]
    gamma_analysis = Float64[]
    n_joint_after = copy(beta_n_after)

    if USE_HEMISPHERE_DUPLICATION
        hemi_obs = build_hemi_observations(
            beta_vec_joint,
            gamma_vec_joint,
            gene_regions,
            region_map,
            model_index;
            beta_updated_flags = UPDATED_ONLY ? beta_updated_flags : nothing,
            gamma_updated_flags = UPDATED_ONLY ? gamma_updated_flags : nothing,
        )
        analysis_row_idx = hemi_obs.gene_region_idx
        analysis_gene_label = hemi_obs.gene_region_labels
        analysis_model_label = hemi_obs.model_region_labels
        beta_analysis = copy(hemi_obs.beta)
        gamma_analysis = copy(hemi_obs.gamma)
        n_joint_after = hemi_obs.n_joint_after
    else
        analysis_row_idx = collect(1:length(gene_regions))
        analysis_gene_label = copy(gene_regions)
        analysis_model_label = copy(gene_regions)
        beta_analysis = copy(beta_gene)
        gamma_analysis = copy(gamma_gene)
    end

    if BETA_POS_ONLY
        keep = .!isnan.(beta_analysis) .& (beta_analysis .> BETA_POS_THRESHOLD)
        beta_analysis = [keep[i] ? beta_analysis[i] : NaN for i in eachindex(beta_analysis)]
        gamma_analysis = [keep[i] ? gamma_analysis[i] : NaN for i in eachindex(gamma_analysis)]
    end

    beta_reg = STANDARDIZE_REGRESSION_PREDICTORS ? zscore_with_nans(beta_analysis) : Float64.(beta_analysis)
    gamma_reg = STANDARDIZE_REGRESSION_PREDICTORS ? zscore_with_nans(gamma_analysis) : Float64.(gamma_analysis)
    valid_analysis_mask = .!isnan.(beta_analysis) .& .!isnan.(gamma_analysis)
    count(valid_analysis_mask) > 0 || error("No valid observations remain for $name.")

    μβ = mean(beta_analysis[valid_analysis_mask])
    μγ = mean(gamma_analysis[valid_analysis_mask])
    σβ = std(beta_analysis[valid_analysis_mask])
    σγ = std(gamma_analysis[valid_analysis_mask])

    G = size(gene_matrix, 2)
    beta_coef_vec = fill(Float64(NaN), G)
    gamma_coef_vec = fill(Float64(NaN), G)
    beta_p_vec = fill(Float64(NaN), G)
    gamma_p_vec = fill(Float64(NaN), G)
    r2_vec = fill(Float64(NaN), G)
    n_used_vec = zeros(Int, G)

    for g in 1:G
        expr = Float64.(gene_matrix[analysis_row_idx, g])
        mask = .!isnan.(beta_reg) .& .!isnan.(gamma_reg) .& .!isnan.(expr)
        n_used_vec[g] = sum(mask)
        if n_used_vec[g] < 4
            continue
        end
        fit = ols_two_predictors(expr[mask], beta_reg[mask], gamma_reg[mask])
        beta_coef_vec[g] = fit.beta_coef
        gamma_coef_vec[g] = fit.gamma_coef
        beta_p_vec[g] = fit.beta_p
        gamma_p_vec[g] = fit.gamma_p
        r2_vec[g] = fit.r2
    end

    coef_mask = .!isnan.(beta_coef_vec) .& .!isnan.(gamma_coef_vec)
    coef_mat = hcat(beta_coef_vec[coef_mask], gamma_coef_vec[coef_mask])
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

    pc1_beta_raw = pc1[1] / σβ
    pc1_gamma_raw = pc1[2] / σγ
    pc1_intercept_raw = -(pc1[1] * μβ / σβ + pc1[2] * μγ / σγ)
    pc2_beta_raw = pc2[1] / σβ
    pc2_gamma_raw = pc2[2] / σγ
    pc2_intercept_raw = -(pc2[1] * μβ / σβ + pc2[2] * μγ / σγ)

    eta_analysis = fill(Float64(NaN), length(analysis_row_idx))
    eta_analysis[valid_analysis_mask] = affine_score_raw(
        beta_analysis[valid_analysis_mask],
        gamma_analysis[valid_analysis_mask],
        pc1_beta_raw,
        pc1_gamma_raw,
        pc1_intercept_raw,
    )

    gene_eta_df, eta_r_vec, _, eta_q_fdr = gene_correlations_with_score(eta_analysis, gene_matrix, gene_IDs, analysis_row_idx)
    regional_eta = affine_score_raw(beta_vec_joint, gamma_vec_joint, pc1_beta_raw, pc1_gamma_raw, pc1_intercept_raw)

    result = (
        name = name,
        inference = inference,
        gene_IDs = gene_IDs,
        gene_regions = gene_regions,
        model_regions = model_regions,
        beta_coef_vec = beta_coef_vec,
        gamma_coef_vec = gamma_coef_vec,
        beta_p_vec = beta_p_vec,
        gamma_p_vec = gamma_p_vec,
        r2_vec = r2_vec,
        coef_mask = coef_mask,
        pca = pca,
        pc1 = pc1,
        pc2 = pc2,
        pc1_beta_raw = pc1_beta_raw,
        pc1_gamma_raw = pc1_gamma_raw,
        pc1_intercept_raw = pc1_intercept_raw,
        pc2_beta_raw = pc2_beta_raw,
        pc2_gamma_raw = pc2_gamma_raw,
        pc2_intercept_raw = pc2_intercept_raw,
        beta_analysis = beta_analysis,
        gamma_analysis = gamma_analysis,
        valid_analysis_mask = valid_analysis_mask,
        eta_analysis_raw = eta_analysis,
        gene_eta_df = gene_eta_df,
        gene_eta_r = eta_r_vec,
        gene_eta_q_fdr = eta_q_fdr,
        regional_eta = regional_eta,
        beta_mean = beta_mean,
        gamma_mean = gamma_mean,
        region_summary_df = DataFrame(
            gene_region = gene_regions,
            beta_region_mean = beta_gene,
            gamma_region_mean = gamma_gene,
            n_beta_matches_before = beta_n_match,
            n_beta_matches_after = beta_n_after,
            n_gamma_matches_before = gamma_n_match,
            n_gamma_matches_after = gamma_n_after,
            n_joint_matches_after = n_joint_after,
        ),
    )

    CSV.write(joinpath(CSV_DIR, "$(name)_gene_corr_eta_from_pca.csv"), gene_eta_df)
    CSV.write(joinpath(CSV_DIR, "$(name)_regional_eta.csv"), DataFrame(region=model_regions, eta=regional_eta, beta=beta_mean, gamma=gamma_mean))
    CSV.write(joinpath(CSV_DIR, "$(name)_pca_summary.csv"), DataFrame(
        component = ["PC1", "PC2"],
        beta_loading = [pc1[1], pc2[1]],
        gamma_loading = [pc1[2], pc2[2]],
        beta_loading_raw = [pc1_beta_raw, pc2_beta_raw],
        gamma_loading_raw = [pc1_gamma_raw, pc2_gamma_raw],
        intercept_raw = [pc1_intercept_raw, pc2_intercept_raw],
        eigenvalue = pca.eigvals,
        explained_variance_ratio = pca.explained_var_ratio,
    ))
    CSV.write(joinpath(CSV_DIR, "$(name)_regional_gene_mapping_summary.csv"), result.region_summary_df)

    coefficient_pca_plot(result, joinpath(FIG_DIR, "$(name)_beta_gamma_coeff_scatter_with_pca.pdf"))
    eta_heatmap_plot(result, joinpath(FIG_DIR, "$(name)_beta_gamma_scatter_with_eta_heatmap.pdf"))

    return result
end


function axis_comparison_plots(mapt, app)
    common_gene_df = innerjoin(
        rename(mapt.gene_eta_df[:, [:gene, :r, :p_fdr]], :r => :r_mapt, :p_fdr => :q_mapt),
        rename(app.gene_eta_df[:, [:gene, :r, :p_fdr]], :r => :r_app, :p_fdr => :q_app),
        on = :gene,
    )
    common_gene_df.abs_delta = abs.(common_gene_df.r_app .- common_gene_df.r_mapt)
    sort!(common_gene_df, :abs_delta, rev=true)
    CSV.write(joinpath(CSV_DIR, "gene_eta_correlation_app_vs_mapt.csv"), common_gene_df)

    regional_eta_df = DataFrame(
        region = mapt.model_regions,
        eta_mapt = mapt.regional_eta,
        eta_app = app.regional_eta,
        delta_app_minus_mapt = app.regional_eta .- mapt.regional_eta,
        beta_mapt = mapt.beta_mean,
        beta_app = app.beta_mean,
        gamma_mapt = mapt.gamma_mean,
        gamma_app = app.gamma_mean,
    )
    regional_eta_df.abs_delta = abs.(regional_eta_df.delta_app_minus_mapt)
    sort!(regional_eta_df, :abs_delta, rev=true)
    CSV.write(joinpath(CSV_DIR, "regional_eta_app_vs_mapt.csv"), regional_eta_df)

    angle_deg = acos(clamp(dot(mapt.pc1, app.pc1), -1.0, 1.0)) * 180 / pi
    axis_summary = DataFrame(
        comparison = ["MAPT", "APP"],
        pc1_beta_loading = [mapt.pc1[1], app.pc1[1]],
        pc1_gamma_loading = [mapt.pc1[2], app.pc1[2]],
        pc1_beta_raw = [mapt.pc1_beta_raw, app.pc1_beta_raw],
        pc1_gamma_raw = [mapt.pc1_gamma_raw, app.pc1_gamma_raw],
        pc1_explained_variance = [mapt.pca.explained_var_ratio[1], app.pca.explained_var_ratio[1]],
    )
    CSV.write(joinpath(CSV_DIR, "pc1_axis_summary_app_vs_mapt.csv"), axis_summary)

    coeff_x = vcat(mapt.beta_coef_vec[mapt.coef_mask], app.beta_coef_vec[app.coef_mask])
    coeff_y = vcat(mapt.gamma_coef_vec[mapt.coef_mask], app.gamma_coef_vec[app.coef_mask])
    xr = extrema(coeff_x)
    yr = extrema(coeff_y)

    overlay = scatter(
        mapt.beta_coef_vec[mapt.coef_mask],
        mapt.gamma_coef_vec[mapt.coef_mask];
        markercolor = :steelblue3,
        markeralpha = 0.12,
        markersize = 7,
        label = "MAPT genes",
        xlabel = STANDARDIZE_REGRESSION_PREDICTORS ? L"\mathrm{gene\ coefficient\ for}\ z(\beta)" : L"\mathrm{gene\ coefficient\ for}\ \beta",
        ylabel = STANDARDIZE_REGRESSION_PREDICTORS ? L"\mathrm{gene\ coefficient\ for}\ z(\gamma)" : L"\mathrm{gene\ coefficient\ for}\ \gamma",
        title = "Gene coefficient clouds with PC1 axes",
        legend = :bottomleft,
    )
    scatter!(overlay, app.beta_coef_vec[app.coef_mask], app.gamma_coef_vec[app.coef_mask]; markercolor=:firebrick3, markeralpha=0.12, markersize=7, label="APP genes")

    center = [0.0, 0.0]
    L = 0.42 * min((xr[2] - xr[1]) / maximum(abs.([mapt.pc1[1], app.pc1[1], eps()])), (yr[2] - yr[1]) / maximum(abs.([mapt.pc1[2], app.pc1[2], eps()])))
    for (vec, color, label) in ((mapt.pc1, :steelblue4, "MAPT PC1"), (app.pc1, :firebrick4, "APP PC1"))
        p1a = center .- L .* vec
        p1b = center .+ L .* vec
        plot!(overlay, [p1a[1], p1b[1]], [p1a[2], p1b[2]]; color=color, lw=PCA_LINEWIDTH, label=label)
    end
    ann = "PC1 angle = $(round(angle_deg, digits=2))°\n" *
          "MAPT PC1 = ($(round(mapt.pc1[1], sigdigits=3)), $(round(mapt.pc1[2], sigdigits=3)))\n" *
          "APP PC1 = ($(round(app.pc1[1], sigdigits=3)), $(round(app.pc1[2], sigdigits=3)))"
    annotate!(overlay, (xr[1] + 0.03 * (xr[2] - xr[1] + eps()), yr[2] - 0.04 * (yr[2] - yr[1] + eps()), text(ann, 10, :black, :left, :top)))
    savefig(overlay, joinpath(FIG_DIR, "app_vs_mapt_pc1_axis_comparison.pdf"))

    gene_scatter = scatter(
        common_gene_df.r_mapt,
        common_gene_df.r_app;
        xlabel = "corr(gene, η) in MAPT",
        ylabel = "corr(gene, η) in APP",
        markercolor = :black,
        markeralpha = 0.18,
        markersize = 7,
        label = false,
        title = "Gene-level vulnerability association shift",
    )
    mn = min(minimum(common_gene_df.r_mapt), minimum(common_gene_df.r_app))
    mx = max(maximum(common_gene_df.r_mapt), maximum(common_gene_df.r_app))
    plot!(gene_scatter, [mn, mx], [mn, mx]; color=:gray55, linestyle=:dash, lw=DIAG_LINEWIDTH, label=false)
    gene_r = cor(common_gene_df.r_mapt, common_gene_df.r_app)
    annotate!(gene_scatter, (mn + 0.04 * (mx - mn + eps()), mx - 0.04 * (mx - mn + eps()), text("Pearson r = $(round(gene_r, sigdigits=3))\nN = $(nrow(common_gene_df)) genes", 10, :black, :left, :top)))
    savefig(gene_scatter, joinpath(FIG_DIR, "gene_eta_correlation_app_vs_mapt.pdf"))

    eta_scatter = scatter(
        regional_eta_df.eta_mapt,
        regional_eta_df.eta_app;
        xlabel = "Regional η in MAPT",
        ylabel = "Regional η in APP",
        markercolor = :black,
        markeralpha = 0.35,
        markersize = 7,
        label = false,
        title = "Regional vulnerability axis shift",
    )
    mn_eta = min(minimum(regional_eta_df.eta_mapt), minimum(regional_eta_df.eta_app))
    mx_eta = max(maximum(regional_eta_df.eta_mapt), maximum(regional_eta_df.eta_app))
    plot!(eta_scatter, [mn_eta, mx_eta], [mn_eta, mx_eta]; color=:gray55, linestyle=:dash, lw=DIAG_LINEWIDTH, label=false)
    eta_r = cor(regional_eta_df.eta_mapt, regional_eta_df.eta_app)
    annotate!(eta_scatter, (mn_eta + 0.04 * (mx_eta - mn_eta + eps()), mx_eta - 0.04 * (mx_eta - mn_eta + eps()), text("Pearson r = $(round(eta_r, sigdigits=3))\nN = $(nrow(regional_eta_df)) regions", 10, :black, :left, :top)))
    savefig(eta_scatter, joinpath(FIG_DIR, "regional_eta_app_vs_mapt.pdf"))

    open(joinpath(CSV_DIR, "pc1_axis_comparison_summary.txt"), "w") do io
        println(io, "PC1 comparison between syn co-pathology inferences")
        println(io, "MAPT standardized PC1 = ", mapt.pc1)
        println(io, "APP standardized PC1  = ", app.pc1)
        println(io, "MAPT raw eta direction = (", mapt.pc1_beta_raw, ", ", mapt.pc1_gamma_raw, ")")
        println(io, "APP raw eta direction  = (", app.pc1_beta_raw, ", ", app.pc1_gamma_raw, ")")
        println(io, "PC1 angle difference (degrees) = ", angle_deg)
        println(io, "corr(gene eta associations MAPT, APP) = ", gene_r)
        println(io, "corr(regional eta MAPT, APP) = ", eta_r)
    end
end


function main()
    mkpath(FIG_DIR)
    mkpath(CSV_DIR)
    setup_defaults()

    gene_data = CSV.read(GENE_DATA_CSV, DataFrame)
    mapt = fit_gene_axis(MAPT_NAME, "simulations/$(MAPT_NAME).jls", gene_data)
    app = fit_gene_axis(APP_NAME, "simulations/$(APP_NAME).jls", gene_data)

    beta_contrast = posterior_contrast_df(mapt.inference, app.inference, "beta")
    gamma_contrast = posterior_contrast_df(mapt.inference, app.inference, "gamma")
    CSV.write(joinpath(CSV_DIR, "beta_posterior_contrast_app_minus_mapt.csv"), beta_contrast)
    CSV.write(joinpath(CSV_DIR, "gamma_posterior_contrast_app_minus_mapt.csv"), gamma_contrast)
    contrast_volcano_plot(beta_contrast, "beta", joinpath(FIG_DIR, "beta_posterior_contrast_fdr.pdf"))
    contrast_volcano_plot(gamma_contrast, "gamma", joinpath(FIG_DIR, "gamma_posterior_contrast_fdr.pdf"))

    axis_comparison_plots(mapt, app)

    println("Saved figures under $FIG_DIR")
    println("Saved tables under $CSV_DIR")
end


main()
