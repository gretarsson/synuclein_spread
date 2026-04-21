#!/usr/bin/env julia

using PathoSpread
using CairoMakie
using CSV
using DataFrames
using Statistics
using LinearAlgebra
using HypothesisTests
using Distributions
using Printf
using MCMCChains

const HIPPO_NAME = "HIPPO"
const MAPT_NAME = "MAPT"
const APP_NAME = "MAPT+APP"

const HIPPO_PATH = get(ENV, "COPATHOLOGY_HIPPO_PATH", "simulations/hippo_DIFFGA_RETRO.jls")
const MAPT_PATH = "simulations/syn_mapt_DIFFGA_RETRO.jls"
const APP_PATH = "simulations/syn_app_DIFFGA_RETRO.jls"
const GENE_DATA_CSV = "data/avg_Pangea_exp.csv"
const OUTPUT_DIR = get(ENV, "COPATHOLOGY_OUTDIR", "figures/copathology_triplet_noseed")
const TOP_N = 30
const SEED_SITE_BASES = Set(["CA1", "CA3", "DG"])
const PLOT_SIZE = (2200, 1550)
const FILTER_UPDATED = get(ENV, "COPATHOLOGY_FILTER_UPDATED", "false") == "true"
const BETA_POS_ONLY = get(ENV, "COPATHOLOGY_BETA_POS_ONLY", "false") == "true"
const UPDATE_ALPHA = 0.001

const USE_HEMISPHERE_DUPLICATION = true
const STANDARDIZE_REGRESSION_PREDICTORS = true
const FORCE_PC1_POSITIVE_GAMMA = true


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


function zscore_row(values::AbstractVector{<:Real})
    x = Float64.(values)
    sigma = std(x)
    if !isfinite(sigma) || sigma == 0
        return zeros(length(x))
    end
    return (x .- mean(x)) ./ sigma
end


function posterior_summary_triplet(inf_hippo, inf_mapt, inf_app, family::String)
    labels = String.(inf_hippo["labels"])
    labels == String.(inf_mapt["labels"]) || error("HIPPO and MAPT labels differ.")
    labels == String.(inf_app["labels"]) || error("HIPPO and APP labels differ.")

    n_regions = length(labels)
    hippo = PathoSpread.posterior_mean_vector_from_priors(inf_hippo["chain"], inf_hippo["priors"], family, n_regions)
    mapt = PathoSpread.posterior_mean_vector_from_priors(inf_mapt["chain"], inf_mapt["priors"], family, n_regions)
    app = PathoSpread.posterior_mean_vector_from_priors(inf_app["chain"], inf_app["priors"], family, n_regions)

    df = DataFrame(
        region = labels,
        family = fill(family, n_regions),
        hippo = hippo,
        mapt = mapt,
        app = app,
    )
    beta_hippo = PathoSpread.posterior_mean_vector_from_priors(inf_hippo["chain"], inf_hippo["priors"], "beta", n_regions)
    beta_mapt = PathoSpread.posterior_mean_vector_from_priors(inf_mapt["chain"], inf_mapt["priors"], "beta", n_regions)
    beta_app = PathoSpread.posterior_mean_vector_from_priors(inf_app["chain"], inf_app["priors"], "beta", n_regions)
    df.beta_hippo = beta_hippo
    df.beta_mapt = beta_mapt
    df.beta_app = beta_app
    df.delta_mapt_minus_hippo = df.mapt .- df.hippo
    df.delta_app_minus_mapt = df.app .- df.mapt
    df.delta_app_minus_hippo = df.app .- df.hippo
    df.range_triplet = [maximum((h, m, a)) - minimum((h, m, a)) for (h, m, a) in zip(df.hippo, df.mapt, df.app)]
    if FILTER_UPDATED
        hippo_updated = region_update_flags(inf_hippo, family)
        mapt_updated = region_update_flags(inf_mapt, family)
        app_updated = region_update_flags(inf_app, family)
        df.updated_hippo = [get(hippo_updated, r, 0) == 1 for r in df.region]
        df.updated_mapt = [get(mapt_updated, r, 0) == 1 for r in df.region]
        df.updated_app = [get(app_updated, r, 0) == 1 for r in df.region]
        df.updated_all = df.updated_hippo .& df.updated_mapt .& df.updated_app
        df = df[df.updated_all, :]
    end
    if BETA_POS_ONLY
        df.beta_positive_all = (df.beta_hippo .> 0) .& (df.beta_mapt .> 0) .& (df.beta_app .> 0)
        df = df[df.beta_positive_all, :]
    end
    df = df[.!is_seed_region.(df.region), :]
    sort!(df, :range_triplet, rev=true)
    return df
end


function top_heatmap_table(df::DataFrame; top_n::Int=TOP_N)
    top_df = first(df, min(top_n, nrow(df)))
    raw = Matrix(top_df[:, [:hippo, :mapt, :app]])
    zmat = reduce(vcat, [reshape(zscore_row(raw[i, :]), 1, :) for i in axes(raw, 1)])
    return top_df, raw, zmat
end


function pairwise_correlation_matrix(df::DataFrame)
    mat = Matrix{Float64}(undef, 3, 3)
    vals = [df.hippo, df.mapt, df.app]
    for i in 1:3, j in 1:3
        mat[i, j] = cor(vals[i], vals[j])
    end
    return mat
end


function ols_two_predictors(y::AbstractVector, x1::AbstractVector, x2::AbstractVector)
    X = hcat(ones(length(y)), x1, x2)
    beta_hat = X \ y
    resid = y - X * beta_hat
    dof = length(y) - size(X, 2)
    sigma2 = sum(abs2, resid) / dof
    XtX_inv = inv(X' * X)
    se = sqrt.(diag(sigma2 .* XtX_inv))
    pvals = [2 * (1 - cdf(TDist(dof), abs(t))) for t in beta_hat ./ se]
    r2 = 1 - sum(abs2, resid) / sum(abs2, y .- mean(y))
    return (
        intercept = beta_hat[1],
        beta_coef = beta_hat[2],
        gamma_coef = beta_hat[3],
        intercept_p = pvals[1],
        beta_p = pvals[2],
        gamma_p = pvals[3],
        r2 = r2,
    )
end


function pca_2d(X::AbstractMatrix)
    mu = vec(mean(X, dims=1))
    Xc = X .- reshape(mu, 1, :)
    sigma = (Xc' * Xc) / (size(X, 1) - 1)
    eig = eigen(Symmetric(sigma))
    ord = sortperm(eig.values; rev=true)
    eigvals = eig.values[ord]
    eigvecs = eig.vectors[:, ord]
    scores = Xc * eigvecs
    explained = eigvals ./ sum(eigvals)
    return (mean_vec = mu, eigvals = eigvals, eigvecs = eigvecs, scores = scores, explained = explained)
end


function zscore_with_nans(x::AbstractVector)
    out = fill(Float64(NaN), length(x))
    mask = .!isnan.(x)
    any(mask) || return out
    mu = mean(x[mask])
    sigma = std(x[mask])
    if !isfinite(sigma) || sigma == 0
        out[mask] .= 0.0
    else
        out[mask] = (x[mask] .- mu) ./ sigma
    end
    return out
end


function build_region_map(gene_regions, model_regions, nonzero_nodes)
    region_map = Dict{String, Vector{String}}()
    for gr in String.(gene_regions)
        matches = String[]
        if gr in SEED_SITE_BASES
            region_map[gr] = matches
            continue
        end
        L = length(gr)
        for (i, mr) in enumerate(String.(model_regions))
            if !(i in nonzero_nodes)
                continue
            end
            base = strip_hemi(mr)
            if base in SEED_SITE_BASES
                continue
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


function build_hemi_observations(beta_vec, gamma_vec, gene_regions, region_map, model_index)
    row_idx = Int[]
    beta_obs = Float64[]
    gamma_obs = Float64[]

    for (r, gr) in enumerate(String.(gene_regions))
        for mr in region_map[gr]
            idx = model_index[mr]
            push!(row_idx, r)
            push!(beta_obs, beta_vec[idx])
            push!(gamma_obs, gamma_vec[idx])
        end
    end

    return (row_idx=row_idx, beta=beta_obs, gamma=gamma_obs)
end


function fit_gene_axis(name::String, inference_path::String, gene_data::DataFrame)
    inference = load_inference(inference_path)
    gene_regions = String.(gene_data[:, 1])
    gene_ids = String.(names(gene_data)[2:end])
    gene_matrix = Matrix(gene_data[:, 2:end])
    model_regions = String.(inference["labels"])
    nonzero_nodes = nonzero_regions(inference["data"], eps=0.01)
    region_map = build_region_map(gene_regions, model_regions, nonzero_nodes)
    model_index = Dict(model_regions[i] => i for i in eachindex(model_regions))

    beta_mean = PathoSpread.posterior_mean_vector_from_priors(inference["chain"], inference["priors"], "beta", length(model_regions))
    gamma_mean = PathoSpread.posterior_mean_vector_from_priors(inference["chain"], inference["priors"], "gamma", length(model_regions))
    beta_updated_flags = FILTER_UPDATED ? region_update_flags(inference, "beta") : Dict{String, Int}()
    gamma_updated_flags = FILTER_UPDATED ? region_update_flags(inference, "gamma") : Dict{String, Int}()

    obs = build_hemi_observations(beta_mean, gamma_mean, gene_regions, region_map, model_index)
    beta_analysis = obs.beta
    gamma_analysis = obs.gamma
    row_idx = obs.row_idx

    beta_reg = STANDARDIZE_REGRESSION_PREDICTORS ? zscore_with_nans(beta_analysis) : Float64.(beta_analysis)
    gamma_reg = STANDARDIZE_REGRESSION_PREDICTORS ? zscore_with_nans(gamma_analysis) : Float64.(gamma_analysis)
    valid = .!isnan.(beta_reg) .& .!isnan.(gamma_reg)
    if FILTER_UPDATED
        obs_regions = String[]
        for gr in String.(gene_regions)
            append!(obs_regions, region_map[gr])
        end
        updated_mask = [get(beta_updated_flags, mr, 0) == 1 && get(gamma_updated_flags, mr, 0) == 1 for mr in obs_regions]
        valid .&= updated_mask
    end
    if BETA_POS_ONLY
        valid .&= beta_analysis .> 0
    end
    count(valid) > 0 || error("No valid gene-axis observations for $name.")

    mu_beta = mean(beta_analysis[valid])
    mu_gamma = mean(gamma_analysis[valid])
    sd_beta = std(beta_analysis[valid])
    sd_gamma = std(gamma_analysis[valid])

    G = size(gene_matrix, 2)
    beta_coef = fill(Float64(NaN), G)
    gamma_coef = fill(Float64(NaN), G)
    r2 = fill(Float64(NaN), G)

    for g in 1:G
        expr = Float64.(gene_matrix[row_idx, g])
        mask = valid .& .!isnan.(expr)
        if sum(mask) < 4
            continue
        end
        fit = ols_two_predictors(expr[mask], beta_reg[mask], gamma_reg[mask])
        beta_coef[g] = fit.beta_coef
        gamma_coef[g] = fit.gamma_coef
        r2[g] = fit.r2
    end

    coef_mask = .!isnan.(beta_coef) .& .!isnan.(gamma_coef)
    coef_mat = hcat(beta_coef[coef_mask], gamma_coef[coef_mask])
    pca = pca_2d(coef_mat)
    pc1 = copy(pca.eigvecs[:, 1])
    if FORCE_PC1_POSITIVE_GAMMA && pc1[2] < 0
        pc1 .*= -1
    end

    pc1_beta_raw = pc1[1] / sd_beta
    pc1_gamma_raw = pc1[2] / sd_gamma
    pc1_intercept_raw = -(pc1[1] * mu_beta / sd_beta + pc1[2] * mu_gamma / sd_gamma)

    return (
        name = name,
        inference = inference,
        beta_mean = beta_mean,
        gamma_mean = gamma_mean,
        beta_coef = beta_coef,
        gamma_coef = gamma_coef,
        coef_mask = coef_mask,
        coef_mat = coef_mat,
        pca = pca,
        pc1 = pc1,
        pc1_beta_raw = pc1_beta_raw,
        pc1_gamma_raw = pc1_gamma_raw,
        pc1_intercept_raw = pc1_intercept_raw,
        gene_ids = gene_ids,
        r2 = r2,
    )
end


function angle_deg(v1::AbstractVector, v2::AbstractVector)
    return acos(clamp(dot(v1, v2), -1.0, 1.0)) * 180 / pi
end


function draw_corr_panel!(ax::Axis, corrmat::AbstractMatrix, title::String)
    heatmap!(ax, 1:3, 1:3, corrmat; colormap=:viridis, colorrange=(0, 1))
    ax.xticks = (1:3, ["HIPPO", "MAPT", "APP"])
    ax.yticks = (1:3, ["HIPPO", "MAPT", "APP"])
    ax.title = title
    ax.titlesize = 24
    ax.xticklabelrotation = pi / 6
    for i in 1:3, j in 1:3
        text!(ax, i, j; text=@sprintf("%.2f", corrmat[j, i]), align=(:center, :center), color=:white, fontsize=18)
    end
    return nothing
end


function draw_gene_axis_overlay!(ax::Axis, results)
    colors = Dict(HIPPO_NAME => :gray35, MAPT_NAME => :steelblue3, APP_NAME => :firebrick3)
    labels = [HIPPO_NAME, MAPT_NAME, APP_NAME]
    all_x = vcat([res.beta_coef[res.coef_mask] for res in results]...)
    all_y = vcat([res.gamma_coef[res.coef_mask] for res in results]...)

    for res in results
        scatter!(
            ax,
            res.beta_coef[res.coef_mask],
            res.gamma_coef[res.coef_mask];
            color = (colors[res.name], 0.12),
            markersize = 8,
            strokewidth = 0,
        )
    end

    hlines!(ax, [0.0]; color=:gray65, linestyle=:dash, linewidth=2)
    vlines!(ax, [0.0]; color=:gray65, linestyle=:dash, linewidth=2)

    L = 0.42 * min(
        (maximum(all_x) - minimum(all_x)) / maximum(abs.([res.pc1[1] for res in results])),
        (maximum(all_y) - minimum(all_y)) / maximum(abs.([res.pc1[2] for res in results])),
    )

    for res in results
        p1a = -L .* res.pc1
        p1b = L .* res.pc1
        lines!(ax, [p1a[1], p1b[1]], [p1a[2], p1b[2]]; color=colors[res.name], linewidth=7)
    end

    ax.xlabel = "Gene coefficient for z(beta)"
    ax.ylabel = "Gene coefficient for z(gamma)"
    ax.title = "Gene-axis comparison"
    ax.titlesize = 24

    ang_hm = angle_deg(results[1].pc1, results[2].pc1)
    ang_ha = angle_deg(results[1].pc1, results[3].pc1)
    ang_ma = angle_deg(results[2].pc1, results[3].pc1)
    summary = "PC1 angles\nHIPPO-MAPT = $(round(ang_hm, digits=1)) deg\nHIPPO-APP = $(round(ang_ha, digits=1)) deg\nMAPT-APP = $(round(ang_ma, digits=1)) deg"
    text!(
        ax,
        minimum(all_x) + 0.04 * (maximum(all_x) - minimum(all_x) + eps()),
        maximum(all_y) - 0.04 * (maximum(all_y) - minimum(all_y) + eps());
        text = summary,
        align = (:left, :top),
        fontsize = 17,
    )

    elements = [
        MarkerElement(color=(colors[l], 0.4), marker=:circle, markersize=12) for l in labels
    ]
    line_elements = [
        LineElement(color=colors[l], linewidth=5) for l in labels
    ]
    Legend(
        ax.parent[2, 1],
        vcat(elements, line_elements),
        vcat(["$(l) gene coefficients" for l in labels], ["$(l) PC1" for l in labels]),
        orientation = :horizontal,
        tellwidth = false,
        framevisible = false,
    )
    return nothing
end


function add_identity_line!(ax::Axis, x::AbstractVector, y::AbstractVector)
    minval = min(minimum(x), minimum(y))
    maxval = max(maximum(x), maximum(y))
    pad = 0.06 * (maxval - minval + eps())
    lo, hi = minval - pad, maxval + pad
    lines!(ax, [lo, hi], [lo, hi]; color=:gray60, linestyle=:dash, linewidth=2)
    xlims!(ax, lo, hi)
    ylims!(ax, lo, hi)
    return nothing
end


function annotate_top_shifts!(ax::Axis, x::AbstractVector, y::AbstractVector, labels::Vector{String}; top_n::Int=5)
    d = abs.(y .- x)
    ord = sortperm(d; rev=true)[1:min(top_n, length(d))]
    xr = maximum(x) - minimum(x) + eps()
    yr = maximum(y) - minimum(y) + eps()
    for idx in ord
        text!(
            ax,
            x[idx] + 0.012 * xr,
            y[idx] + 0.012 * yr;
            text = labels[idx],
            fontsize = 13,
            align = (:left, :bottom),
            color = :black,
        )
    end
    return nothing
end


function pairwise_scatter_figure(df::DataFrame, family::String)
    pairs = [
        (:hippo, :mapt, "HIPPO vs MAPT"),
        (:hippo, :app, "HIPPO vs MAPT+APP"),
        (:mapt, :app, "MAPT vs MAPT+APP"),
    ]
    fig = Figure(size=(2200, 750))
    for (i, (xcol, ycol, title)) in enumerate(pairs)
        x = Vector{Float64}(df[:, xcol])
        y = Vector{Float64}(df[:, ycol])
        ax = Axis(
            fig[1, i],
            title = "$(uppercase(family)) " * title,
            xlabel = uppercase(String(xcol)),
            ylabel = uppercase(String(ycol)),
            titlesize = 26,
            xlabelsize = 22,
            ylabelsize = 22,
            xticklabelsize = 16,
            yticklabelsize = 16,
        )
        scatter!(ax, x, y; color=(:black, 0.55), markersize=10)
        add_identity_line!(ax, x, y)
        annotate_top_shifts!(ax, x, y, String.(df.region); top_n=6)
        text!(
            ax,
            minimum(x) + 0.05 * (maximum(x) - minimum(x) + eps()),
            maximum(y) - 0.05 * (maximum(y) - minimum(y) + eps());
            text = @sprintf("r = %.3f\nN = %d", cor(x, y), length(x)),
            fontsize = 18,
            align = (:left, :top),
        )
    end
    return fig
end


function gene_axis_figure(results)
    fig = Figure(size=(2200, 1000))
    ax_overlay = Axis(fig[1, 1], title="Gene-axis comparison", xlabel="Gene coefficient for z(beta)", ylabel="Gene coefficient for z(gamma)")
    draw_gene_axis_overlay!(ax_overlay, results)

    angle_mat = [
        0.0 angle_deg(results[1].pc1, results[2].pc1) angle_deg(results[1].pc1, results[3].pc1);
        angle_deg(results[2].pc1, results[1].pc1) 0.0 angle_deg(results[2].pc1, results[3].pc1);
        angle_deg(results[3].pc1, results[1].pc1) angle_deg(results[3].pc1, results[2].pc1) 0.0
    ]
    ax_angle = Axis(fig[1, 2], aspect=1, title="PC1 angle differences", xticklabelrotation=pi/6)
    heatmap!(ax_angle, 1:3, 1:3, angle_mat; colormap=:magma, colorrange=(0, 90))
    ax_angle.xticks = (1:3, [HIPPO_NAME, MAPT_NAME, APP_NAME])
    ax_angle.yticks = (1:3, [HIPPO_NAME, MAPT_NAME, APP_NAME])
    for i in 1:3, j in 1:3
        text!(ax_angle, i, j; text=@sprintf("%.1f", angle_mat[j, i]), align=(:center, :center), color=:white, fontsize=18)
    end

    summary = DataFrame(
        condition = [res.name for res in results],
        pc1_beta = [res.pc1[1] for res in results],
        pc1_gamma = [res.pc1[2] for res in results],
        explained = [res.pca.explained[1] for res in results],
    )
    ax_load = Axis(fig[2, 1:2], title="PC1 loadings by condition", xlabel="PC1 beta loading", ylabel="PC1 gamma loading")
    colors = Dict(HIPPO_NAME => :gray35, MAPT_NAME => :steelblue3, APP_NAME => :firebrick3)
    for row in eachrow(summary)
        scatter!(ax_load, [row.pc1_beta], [row.pc1_gamma]; color=colors[row.condition], markersize=18)
        text!(ax_load, row.pc1_beta, row.pc1_gamma; text=" " * row.condition, align=(:left, :center), fontsize=16)
    end
    hlines!(ax_load, [0.0]; color=:gray70, linestyle=:dash, linewidth=2)
    vlines!(ax_load, [0.0]; color=:gray70, linestyle=:dash, linewidth=2)

    return fig
end


function save_outputs(fig::Figure, beta_df::DataFrame, gamma_df::DataFrame, gene_results)
    mkpath(OUTPUT_DIR)
    csv_dir = joinpath(OUTPUT_DIR, "csv")
    mkpath(csv_dir)

    CSV.write(joinpath(csv_dir, "beta_triplet_summary.csv"), beta_df)
    CSV.write(joinpath(csv_dir, "gamma_triplet_summary.csv"), gamma_df)

    angle_df = DataFrame(
        comparison = ["HIPPO_vs_MAPT", "HIPPO_vs_APP", "MAPT_vs_APP"],
        angle_degrees = [
            angle_deg(gene_results[1].pc1, gene_results[2].pc1),
            angle_deg(gene_results[1].pc1, gene_results[3].pc1),
            angle_deg(gene_results[2].pc1, gene_results[3].pc1),
        ],
    )
    CSV.write(joinpath(csv_dir, "gene_axis_triplet_angles.csv"), angle_df)

    pca_df = DataFrame(
        condition = [res.name for res in gene_results],
        pc1_beta_loading = [res.pc1[1] for res in gene_results],
        pc1_gamma_loading = [res.pc1[2] for res in gene_results],
        pc1_beta_raw = [res.pc1_beta_raw for res in gene_results],
        pc1_gamma_raw = [res.pc1_gamma_raw for res in gene_results],
        pc1_explained_variance = [res.pca.explained[1] for res in gene_results],
    )
    CSV.write(joinpath(csv_dir, "gene_axis_triplet_summary.csv"), pca_df)

    pdf_path = joinpath(OUTPUT_DIR, "hippo_mapt_app_triplet_noseed_summary.pdf")
    png_path = joinpath(OUTPUT_DIR, "hippo_mapt_app_triplet_noseed_summary.png")
    save(pdf_path, fig)
    save(png_path, fig)

    return pdf_path, png_path
end


function main()
    PathoSpread.setup_plot_theme!(font="Arial", base=18, lw=3, markersize=10, dpi=300)

    inf_hippo = load_inference(HIPPO_PATH)
    inf_mapt = load_inference(MAPT_PATH)
    inf_app = load_inference(APP_PATH)

    beta_df = posterior_summary_triplet(inf_hippo, inf_mapt, inf_app, "beta")
    gamma_df = posterior_summary_triplet(inf_hippo, inf_mapt, inf_app, "gamma")

    beta_corr = pairwise_correlation_matrix(beta_df)
    gamma_corr = pairwise_correlation_matrix(gamma_df)

    gene_data = CSV.read(GENE_DATA_CSV, DataFrame)
    gene_results = [
        fit_gene_axis(HIPPO_NAME, HIPPO_PATH, gene_data),
        fit_gene_axis(MAPT_NAME, MAPT_PATH, gene_data),
        fit_gene_axis(APP_NAME, APP_PATH, gene_data),
    ]

    fig = Figure(size=PLOT_SIZE)
    ax_beta_corr = Axis(fig[1, 1], aspect=1)
    ax_gamma_corr = Axis(fig[1, 2], aspect=1)
    draw_corr_panel!(ax_beta_corr, beta_corr, "C. Beta regional similarity")
    draw_corr_panel!(ax_gamma_corr, gamma_corr, "D. Gamma regional similarity")
    Label(fig[2, 1:2], "Initiation sites CA1, CA3, and DG removed. Separate beta, gamma, and gene-axis figures are also saved.", fontsize=18)

    beta_scatter_fig = pairwise_scatter_figure(beta_df, "beta")
    gamma_scatter_fig = pairwise_scatter_figure(gamma_df, "gamma")
    gene_fig = gene_axis_figure(gene_results)

    pdf_path, png_path = save_outputs(fig, beta_df, gamma_df, gene_results)
    beta_pdf = joinpath(OUTPUT_DIR, "beta_triplet_pairwise_scatter.pdf")
    beta_png = joinpath(OUTPUT_DIR, "beta_triplet_pairwise_scatter.png")
    gamma_pdf = joinpath(OUTPUT_DIR, "gamma_triplet_pairwise_scatter.pdf")
    gamma_png = joinpath(OUTPUT_DIR, "gamma_triplet_pairwise_scatter.png")
    gene_pdf = joinpath(OUTPUT_DIR, "gene_axis_triplet_comparison.pdf")
    gene_png = joinpath(OUTPUT_DIR, "gene_axis_triplet_comparison.png")
    save(beta_pdf, beta_scatter_fig)
    save(beta_png, beta_scatter_fig)
    save(gamma_pdf, gamma_scatter_fig)
    save(gamma_png, gamma_scatter_fig)
    save(gene_pdf, gene_fig)
    save(gene_png, gene_fig)

    println("Saved figure:")
    println(pdf_path)
    println(png_path)
    println(beta_pdf)
    println(beta_png)
    println(gamma_pdf)
    println(gamma_png)
    println(gene_pdf)
    println(gene_png)
    println("Saved tables under ", joinpath(OUTPUT_DIR, "csv"))
end


main()
