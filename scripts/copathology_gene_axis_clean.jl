#!/usr/bin/env julia

using PathoSpread
using CSV
using DataFrames
using Statistics
using LinearAlgebra
using HypothesisTests
using MCMCChains
using CairoMakie
using Distributions

const MAPT_PATH = get(ENV, "COPATHOLOGY_MAPT_PATH", "simulations/syn_mapt_DIFFGA_RETRO.jls")
const APP_PATH = get(ENV, "COPATHOLOGY_APP_PATH", "simulations/syn_app_DIFFGA_RETRO.jls")
const MAPT_NAME = splitext(basename(MAPT_PATH))[1]
const APP_NAME = splitext(basename(APP_PATH))[1]
const GENE_DATA_CSV = "data/avg_Pangea_exp.csv"
const OUTROOT = get(ENV, "COPATHOLOGY_OUTROOT", "figures/copathology")
const OUTDIR = joinpath(OUTROOT, "gene_axis")
const CSV_DIR = joinpath(OUTDIR, "csv")
const ZERO_THRESHOLD = 0.01


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


function build_region_map(gene_regions, model_regions, nonzero_nodes)
    region_map = Dict{String, Vector{String}}()
    for gr in String.(gene_regions)
        L = length(gr)
        matches = String[]
        for (i, mr) in enumerate(String.(model_regions))
            if !(i in nonzero_nodes)
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
                if length(mr) == 1 + L || !isletter(mr[min(end, 2 + L)])
                    push!(matches, mr)
                end
            end
        end
        region_map[gr] = matches
    end
    return region_map
end


function build_hemi_observations(beta_vec, gamma_vec, gene_regions, region_map, model_index)
    row_idx = Int[]
    beta = Float64[]
    gamma = Float64[]
    for (r, gr) in enumerate(String.(gene_regions))
        for mr in region_map[gr]
            idx = model_index[mr]
            push!(row_idx, r)
            push!(beta, beta_vec[idx])
            push!(gamma, gamma_vec[idx])
        end
    end
    return (row_idx=row_idx, beta=beta, gamma=gamma)
end


function ols_two_predictors(y::AbstractVector, x1::AbstractVector, x2::AbstractVector)
    X = hcat(ones(length(y)), x1, x2)
    βhat = X \ y
    resid = y - X * βhat
    dof = length(y) - size(X, 2)
    σ2 = sum(abs2, resid) / dof
    XtX_inv = inv(X' * X)
    se = sqrt.(diag(σ2 .* XtX_inv))
    tstats = βhat ./ se
    pvals = [2 * (1 - cdf(TDist(dof), abs(t))) for t in tstats]
    r2 = 1 - sum(abs2, resid) / sum(abs2, y .- mean(y))
    return (intercept=βhat[1], beta_coef=βhat[2], gamma_coef=βhat[3], beta_p=pvals[2], gamma_p=pvals[3], r2=r2)
end


function pca_2d(X::AbstractMatrix)
    μ = vec(mean(X, dims=1))
    Xc = X .- reshape(μ, 1, :)
    Σ = (Xc' * Xc) / (size(X, 1) - 1)
    F = eigen(Symmetric(Σ))
    ord = sortperm(F.values, rev=true)
    eigvals = F.values[ord]
    eigvecs = F.vectors[:, ord]
    scores = Xc * eigvecs
    explained = eigvals ./ sum(eigvals)
    return (mean_vec=μ, eigvecs=eigvecs, eigvals=eigvals, scores=scores, explained=explained)
end


function fit_gene_axis(name::String, inference_path::String, gene_data::DataFrame)
    inference = load_inference(inference_path)
    gene_regions = String.(gene_data[:, 1])
    gene_ids = String.(names(gene_data)[2:end])
    gene_matrix = Matrix(gene_data[:, 2:end])
    model_regions = String.(inference["labels"])
    nonzero_nodes = nonzero_regions(inference["data"], eps=ZERO_THRESHOLD)
    region_map = build_region_map(gene_regions, model_regions, nonzero_nodes)
    model_index = Dict(model_regions[i] => i for i in eachindex(model_regions))

    beta_mean = PathoSpread.posterior_mean_vector_from_priors(inference["chain"], inference["priors"], "beta", length(model_regions))
    gamma_mean = PathoSpread.posterior_mean_vector_from_priors(inference["chain"], inference["priors"], "gamma", length(model_regions))
    obs = build_hemi_observations(beta_mean, gamma_mean, gene_regions, region_map, model_index)

    beta_reg = zscore_with_nans(obs.beta)
    gamma_reg = zscore_with_nans(obs.gamma)
    valid = .!isnan.(beta_reg) .& .!isnan.(gamma_reg)
    μβ = mean(obs.beta[valid])
    μγ = mean(obs.gamma[valid])
    σβ = std(obs.beta[valid])
    σγ = std(obs.gamma[valid])

    G = size(gene_matrix, 2)
    beta_coef = fill(Float64(NaN), G)
    gamma_coef = fill(Float64(NaN), G)
    r2 = fill(Float64(NaN), G)

    for g in 1:G
        expr = Float64.(gene_matrix[obs.row_idx, g])
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
    if pc1[2] < 0
        pc1 .*= -1
    end
    pc1_beta_raw = pc1[1] / σβ
    pc1_gamma_raw = pc1[2] / σγ
    pc1_intercept_raw = -(pc1[1] * μβ / σβ + pc1[2] * μγ / σγ)

    return (
        name=name,
        gene_ids=gene_ids,
        beta_coef=beta_coef,
        gamma_coef=gamma_coef,
        coef_mask=coef_mask,
        pca=pca,
        pc1=pc1,
        pc1_beta_raw=pc1_beta_raw,
        pc1_gamma_raw=pc1_gamma_raw,
        pc1_intercept_raw=pc1_intercept_raw,
        beta_mean=beta_mean,
        gamma_mean=gamma_mean,
    )
end


function coeff_cloud_with_pca(result, title::String)
    x = result.beta_coef[result.coef_mask]
    y = result.gamma_coef[result.coef_mask]
    fig = Figure(size=(1000, 800))
    ax = Axis(fig[1, 1], title=title, xlabel="Gene coefficient for z(beta)", ylabel="Gene coefficient for z(gamma)",
        titlesize=32, xlabelsize=26, ylabelsize=26, xticklabelsize=18, yticklabelsize=18)
    scatter!(ax, x, y; color=(:black, 0.16), markersize=10)
    hlines!(ax, [0.0]; color=:gray60, linestyle=:dash, linewidth=2)
    vlines!(ax, [0.0]; color=:gray60, linestyle=:dash, linewidth=2)
    center = result.pca.mean_vec
    v1 = result.pc1
    xr = extrema(x)
    yr = extrema(y)
    xspan = xr[2] - xr[1]
    yspan = yr[2] - yr[1]
    L1 = 0.36 * min(xspan / max(abs(v1[1]), eps()), yspan / max(abs(v1[2]), eps()))
    p1a = center .- L1 .* v1
    p1b = center .+ L1 .* v1
    lines!(ax, [p1a[1], p1b[1]], [p1a[2], p1b[2]]; color=:firebrick3, linewidth=7)
    text!(ax, 0.02, 0.98; space=:relative,
        text="PC1 = (" * string(round(result.pc1[1], sigdigits=3)) * ", " * string(round(result.pc1[2], sigdigits=3)) * ")\nExplained = " * string(round(100 * result.pca.explained[1], digits=1)) * "%",
        align=(:left, :top), fontsize=18)
    return fig
end


function overlay_figure(mapt, app)
    fig = Figure(size=(1100, 850))
    ax = Axis(fig[1, 1], title="MAPT and APP gene coefficient clouds", xlabel="Gene coefficient for z(beta)", ylabel="Gene coefficient for z(gamma)",
        titlesize=32, xlabelsize=26, ylabelsize=26, xticklabelsize=18, yticklabelsize=18)
    scatter!(ax, mapt.beta_coef[mapt.coef_mask], mapt.gamma_coef[mapt.coef_mask]; color=(:steelblue3, 0.14), markersize=9)
    scatter!(ax, app.beta_coef[app.coef_mask], app.gamma_coef[app.coef_mask]; color=(:firebrick3, 0.14), markersize=9)
    hlines!(ax, [0.0]; color=:gray60, linestyle=:dash, linewidth=2)
    vlines!(ax, [0.0]; color=:gray60, linestyle=:dash, linewidth=2)
    for (res, col) in ((mapt, :steelblue4), (app, :firebrick4))
        vec = res.pc1
        L = 0.42 * min(
            (maximum(vcat(mapt.beta_coef[mapt.coef_mask], app.beta_coef[app.coef_mask])) - minimum(vcat(mapt.beta_coef[mapt.coef_mask], app.beta_coef[app.coef_mask]))) / max(abs(vec[1]), eps()),
            (maximum(vcat(mapt.gamma_coef[mapt.coef_mask], app.gamma_coef[app.coef_mask])) - minimum(vcat(mapt.gamma_coef[mapt.coef_mask], app.gamma_coef[app.coef_mask]))) / max(abs(vec[2]), eps()),
        )
        p1a = -L .* vec
        p1b = L .* vec
        lines!(ax, [p1a[1], p1b[1]], [p1a[2], p1b[2]]; color=col, linewidth=8)
    end
    return fig
end


function pc1_only_figure(mapt, app)
    fig = Figure(size=(900, 900))
    ax = Axis(fig[1, 1], title="PC1 directions only", xlabel="beta direction", ylabel="gamma direction",
        titlesize=32, xlabelsize=26, ylabelsize=26, xticklabelsize=18, yticklabelsize=18, aspect=1)
    limits = (-1.1, 1.1)
    xlims!(ax, limits...)
    ylims!(ax, limits...)
    hlines!(ax, [0.0]; color=:gray60, linestyle=:dash, linewidth=2)
    vlines!(ax, [0.0]; color=:gray60, linestyle=:dash, linewidth=2)
    lines!(ax, [0, mapt.pc1[1]], [0, mapt.pc1[2]]; color=:steelblue4, linewidth=10)
    lines!(ax, [0, app.pc1[1]], [0, app.pc1[2]]; color=:firebrick4, linewidth=10)
    scatter!(ax, [mapt.pc1[1], app.pc1[1]], [mapt.pc1[2], app.pc1[2]]; color=[:steelblue4, :firebrick4], markersize=18)
    angle_deg = acos(clamp(dot(mapt.pc1, app.pc1), -1.0, 1.0)) * 180 / pi
    text!(ax, 0.02, 0.98; space=:relative,
        text="MAPT = (" * string(round(mapt.pc1[1], sigdigits=3)) * ", " * string(round(mapt.pc1[2], sigdigits=3)) * ")\nAPP = (" * string(round(app.pc1[1], sigdigits=3)) * ", " * string(round(app.pc1[2], sigdigits=3)) * ")\nAngle = " * string(round(angle_deg, digits=2)) * " deg",
        align=(:left, :top), fontsize=18)
    return fig
end


function main()
    mkpath(OUTDIR)
    mkpath(CSV_DIR)
    PathoSpread.setup_plot_theme!(font="Arial", base=20, lw=3, markersize=12, dpi=300)

    gene_data = CSV.read(GENE_DATA_CSV, DataFrame)
    mapt = fit_gene_axis(MAPT_NAME, MAPT_PATH, gene_data)
    app = fit_gene_axis(APP_NAME, APP_PATH, gene_data)

    save(joinpath(OUTDIR, "mapt_coeff_cloud_pca.pdf"), coeff_cloud_with_pca(mapt, "MAPT coefficient cloud with PCA"))
    save(joinpath(OUTDIR, "app_coeff_cloud_pca.pdf"), coeff_cloud_with_pca(app, "APP coefficient cloud with PCA"))
    save(joinpath(OUTDIR, "mapt_app_overlay.pdf"), overlay_figure(mapt, app))
    save(joinpath(OUTDIR, "pc1_angle_only.pdf"), pc1_only_figure(mapt, app))

    CSV.write(joinpath(CSV_DIR, "mapt_pca_summary.csv"), DataFrame(pc1_beta=[mapt.pc1[1]], pc1_gamma=[mapt.pc1[2]], pc1_explained=[mapt.pca.explained[1]]))
    CSV.write(joinpath(CSV_DIR, "app_pca_summary.csv"), DataFrame(pc1_beta=[app.pc1[1]], pc1_gamma=[app.pc1[2]], pc1_explained=[app.pca.explained[1]]))
    CSV.write(joinpath(CSV_DIR, "pc1_angle_summary.csv"), DataFrame(angle_degrees=[acos(clamp(dot(mapt.pc1, app.pc1), -1.0, 1.0)) * 180 / pi]))

    println("Saved gene-axis figures to $OUTDIR")
end


main()
