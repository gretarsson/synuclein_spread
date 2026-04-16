#!/usr/bin/env julia

using CSV, DataFrames, Statistics, LinearAlgebra, Printf
using HypothesisTests
using StatsBase
using Plots
using Measures

# ============================================================
# SETTINGS
# ============================================================

const DEFAULT_DIR1 = "results/gene_correlation_pca_axis_HEMIDUP_beta_gt_0_UPDATED"
const DEFAULT_DIR2 = "results/gene_correlation_pca_axis_HIPPO_HEMIDUP_beta_gt_0_UPDATED"

const LABEL1 = "Striatum"
const LABEL2 = "Hippocampus"

const OUTROOT = "results/compare_gene_correlation_pca_axis_STR_vs_HIPPO"
const OUTCSV  = joinpath(OUTROOT, "csv")
const OUTFIG  = joinpath(OUTROOT, "figures")
const OUTTXT  = joinpath(OUTROOT, "txt")

mkpath(OUTROOT)
mkpath(OUTCSV)
mkpath(OUTFIG)
mkpath(OUTTXT)

# plotting
const FIGSIZE_SQUARE = (900, 900)
const FIGSIZE_RECT   = (1100, 850)

const GUIDEFS  = 28
const TICKFS   = 22
const LEGFS    = 25
const ANNFS    = 25
const LW_MAIN  = 8
const LW_AUX   = 5.0
const MS_SCAT  = 15
const ALPHA_SCAT = 0.20
const STROKE_W = 1.2

const TOPN_OVERLAP = 250

default(
    guidefontsize = GUIDEFS,
    tickfontsize = TICKFS,
    legendfontsize = LEGFS,
    linewidth = LW_MAIN,
    markerstrokewidth = STROKE_W,
    size = FIGSIZE_SQUARE
)

# ============================================================
# HELPERS
# ============================================================

function format_p(p::Real)
    if isnan(p)
        return "NaN"
    elseif p < 1e-4
        return "< 1e-4"
    else
        return @sprintf("%.3g", p)
    end
end

function zscore_safe(x::AbstractVector)
    μ = mean(x)
    σ = std(x)
    if !isfinite(σ) || σ == 0
        return zeros(length(x))
    end
    return (x .- μ) ./ σ
end

function fisher_ci(r::Real, n::Int; alpha=0.05)
    if n <= 3 || abs(r) >= 1
        return (NaN, NaN)
    end
    z = atanh(r)
    se = 1 / sqrt(n - 3)
    zcrit = quantile(Normal(), 1 - alpha/2)
    lo = tanh(z - zcrit * se)
    hi = tanh(z + zcrit * se)
    return (lo, hi)
end

function rank_overlap(set1::AbstractVector, set2::AbstractVector)
    return length(intersect(Set(set1), Set(set2)))
end

function jaccard_index(set1::AbstractVector, set2::AbstractVector)
    A = Set(set1)
    B = Set(set2)
    denom = length(union(A, B))
    denom == 0 && return NaN
    return length(intersect(A, B)) / denom
end

function hypergeom_overlap_pval(k::Int, K::Int, n::Int, N::Int)
    # probability of overlap >= k when drawing n from N with K marked
    # Hypergeometric(N, K, n)
    hg = Hypergeometric(K, N-K, n)
    maxk = min(K, n)
    s = 0.0
    for x in k:maxk
        s += pdf(hg, x)
    end
    return min(max(s, 0.0), 1.0)
end

function find_one_file(dir::AbstractString, pattern::Regex)
    files = readdir(dir; join=true)
    hits = filter(f -> occursin(pattern, basename(f)), files)
    length(hits) == 1 || error("Expected exactly one match for $(pattern) in $dir, found $(length(hits)).\nMatches:\n$(join(hits, "\n"))")
    return hits[1]
end

function load_analysis_folder(dir::AbstractString)
    csvdir = joinpath(dir, "csv")
    txtdir = joinpath(dir, "txt")

    isdir(csvdir) || error("Missing csv dir: $csvdir")
    isdir(txtdir) || error("Missing txt dir: $txtdir")

    beta_csv  = find_one_file(csvdir, r"_gene_corr_beta_multireg.*\.csv$")
    gamma_csv = find_one_file(csvdir, r"_gene_corr_gamma_multireg.*\.csv$")
    eta_csv   = find_one_file(csvdir, r"_gene_corr_eta_from_pca_beta_gamma.*\.csv$")
    pca_csv   = find_one_file(csvdir, r"_pca_on_beta_gamma_coefficients.*\.csv$")
    run_txt   = find_one_file(txtdir, r"_run_summary.*\.txt$")

    beta  = CSV.read(beta_csv, DataFrame)
    gamma = CSV.read(gamma_csv, DataFrame)
    eta   = CSV.read(eta_csv, DataFrame)
    pca   = CSV.read(pca_csv, DataFrame)

    return (
        dir = dir,
        beta = beta,
        gamma = gamma,
        eta = eta,
        pca = pca,
        beta_csv = beta_csv,
        gamma_csv = gamma_csv,
        eta_csv = eta_csv,
        pca_csv = pca_csv,
        run_txt = run_txt
    )
end

function rename_metric_df(df::DataFrame, valname::AbstractString, lab::AbstractString)
    out = select(df, :gene, :r, :p_un, :p_bonf, :p_fdr, :n_used)
    rename!(out,
        :r      => Symbol("$(valname)_$(lab)"),
        :p_un   => Symbol("p_un_$(valname)_$(lab)"),
        :p_bonf => Symbol("p_bonf_$(valname)_$(lab)"),
        :p_fdr  => Symbol("p_fdr_$(valname)_$(lab)"),
        :n_used => Symbol("n_used_$(valname)_$(lab)")
    )
    return out
end

function compare_vectors(x::AbstractVector, y::AbstractVector)
    mask = .!isnan.(x) .& .!isnan.(y)
    xv = Float64.(x[mask])
    yv = Float64.(y[mask])
    n = length(xv)
    n >= 3 || error("Need at least 3 paired observations.")

    pearson_r = cor(xv, yv)
    pearson_p = pvalue(CorrelationTest(xv, yv))

    rx = tiedrank(xv)
    ry = tiedrank(yv)
    spearman_r = cor(rx, ry)
    spearman_p = pvalue(CorrelationTest(rx, ry))

    concordance = mean(sign.(xv) .== sign.(yv))

    # regression through origin
    slope_origin = sum(xv .* yv) / sum(xv .* xv)

    # OLS with intercept
    μx = mean(xv)
    μy = mean(yv)
    slope_ols = sum((xv .- μx) .* (yv .- μy)) / sum((xv .- μx).^2)
    intercept_ols = μy - slope_ols * μx

    mae = mean(abs.(xv .- yv))
    rmse = sqrt(mean((xv .- yv).^2))

    return (
        n = n,
        mask = mask,
        x = xv,
        y = yv,
        pearson_r = pearson_r,
        pearson_p = pearson_p,
        spearman_r = spearman_r,
        spearman_p = spearman_p,
        concordance = concordance,
        slope_origin = slope_origin,
        slope_ols = slope_ols,
        intercept_ols = intercept_ols,
        mae = mae,
        rmse = rmse
    )
end

function get_pc1(pca_df::DataFrame; use_raw::Bool=false)
    row = pca_df[pca_df.component .== "PC1", :]
    nrow(row) == 1 || error("Expected one PC1 row.")

    if use_raw
        v = [Float64(row.beta_loading_raw[1]), Float64(row.gamma_loading_raw[1])]
    else
        v = [Float64(row.beta_loading[1]), Float64(row.gamma_loading[1])]
    end

    evr = Float64(row.explained_variance_ratio[1])
    return v, evr
end

function orient_same_direction(vref::AbstractVector, v::AbstractVector)
    return dot(vref, v) < 0 ? -v : v
end

function angle_deg(v1::AbstractVector, v2::AbstractVector)
    u1 = v1 / norm(v1)
    u2 = v2 / norm(v2)
    c = clamp(dot(u1, u2), -1.0, 1.0)
    return acosd(c)
end

function axis_comparison_stats(pca1::DataFrame, pca2::DataFrame; use_raw::Bool=false)
    v1, evr1 = get_pc1(pca1; use_raw=use_raw)
    v2, evr2 = get_pc1(pca2; use_raw=use_raw)
    v2o = orient_same_direction(v1, v2)

    cos_sim = dot(v1 / norm(v1), v2o / norm(v2o))
    ang = angle_deg(v1, v2o)

    return (
        v1 = v1,
        v2 = v2o,
        evr1 = evr1,
        evr2 = evr2,
        cos_sim = cos_sim,
        angle_deg = ang,
        use_raw = use_raw
    )
end

function scatter_compare_plot(
    x::AbstractVector,
    y::AbstractVector;
    xlabel::AbstractString,
    ylabel::AbstractString,
    outfile::AbstractString,
    markercolor = RGB(0.23, 0.35, 0.62),
    squarelims = true,
    annotate_stats = true
)
    stats = compare_vectors(x, y)

    xv = stats.x
    yv = stats.y

    xmin = min(minimum(xv), minimum(yv))
    xmax = max(maximum(xv), maximum(yv))
    pad = 0.06 * (xmax - xmin + eps())
    lo = xmin - pad
    hi = xmax + pad

    plt = scatter(
        xv, yv;
        xlabel = xlabel,
        ylabel = ylabel,
        markersize = MS_SCAT,
        markercolor = markercolor,
        markeralpha = ALPHA_SCAT,
        markerstrokewidth = STROKE_W,
        label = false,
        title = "",
        size = FIGSIZE_SQUARE,
        left_margin = 5mm,
        right_margin = 4mm,
        top_margin = 5mm,
        bottom_margin = 4mm,
        legend = false
    )

    # identity line y = x
    plot!(
        plt,
        [lo, hi], [lo, hi];
        color = :gray40,
        lw = LW_AUX,
        linestyle = :dash,
        label = false
    )

    # OLS with intercept
    xfit = range(lo, hi, length=200)
    yfit_ols = stats.intercept_ols .+ stats.slope_ols .* xfit

    plot!(
        plt,
        xfit, yfit_ols;
        color = :black,
        lw = LW_MAIN,
        label = false
    )

    hline!(plt, [0.0]; color=:gray50, lw=LW_AUX, linestyle=:dash, label=false)
    vline!(plt, [0.0]; color=:gray50, lw=LW_AUX, linestyle=:dash, label=false)

    if squarelims
        xlims!(plt, lo, hi)
        ylims!(plt, lo, hi)
    end

    if annotate_stats
        ann = "Pearson r = $(round(stats.pearson_r, sigdigits=3)), p = $(format_p(stats.pearson_p))\n"
        annotate!(
            plt,
            (lo + 0.03*(hi-lo), hi + 0.05*(hi-lo),
             text(ann, ANNFS, :black, :left, :top))
        )
    end

    savefig(plt, outfile)
    return plt, stats
end

function pca_axis_plot(
    v1::AbstractVector,
    v2::AbstractVector,
    evr1::Real,
    evr2::Real,
    outfile::AbstractString;
    use_raw::Bool=false
)
    u1 = v1 / norm(v1)
    u2 = orient_same_direction(u1, v2 / norm(v2))

    xlab = use_raw ? L"\beta\ \mathrm{loading}" : L"z(\beta)\ \mathrm{loading}"
    ylab = use_raw ? L"\gamma\ \mathrm{loading}" : L"z(\gamma)\ \mathrm{loading}"

    plt = plot(
        xlabel = xlab,
        ylabel = ylab,
        xlims = (-1.1, 1.1),
        ylims = (-1.1, 1.1),
        aspect_ratio = :equal,
        legend = :bottomright,
        title = "",
        size = FIGSIZE_SQUARE,
        guidefontsize = 1.25*GUIDEFS,
        left_margin = 6mm,
        right_margin = 4mm,
        top_margin = 4mm,
        bottom_margin = 4mm
    )

    hline!(plt, [0.0]; color=:gray50, lw=LW_AUX, linestyle=:dash, label=false)
    vline!(plt, [0.0]; color=:gray50, lw=LW_AUX, linestyle=:dash, label=false)

    plot!(plt, [0, u1[1]], [0, u1[2]];
        arrow = :arrow, lw = 20, color = RGB(0.10,0.10,0.10),
        label = "$(LABEL1) PC1",
        legendfontsize=LEGFS
    )
    plot!(plt, [0, u2[1]], [0, u2[2]];
        arrow = :arrow, lw = 20, color = RGB(0.45,0.45,0.45),
        label = "$(LABEL2) PC1",
        legendfontsize=LEGFS
    )

    scatter!(plt, [u1[1], u2[1]], [u1[2], u2[2]];
        markersize = 12,
        markercolor = [:black, :gray40],
        markerstrokewidth = 0,
        label = false
    )

    cos_sim = dot(u1, u2)
    ang = angle_deg(u1, u2)

    ann =
          "cosine similarity = $(round(cos_sim, sigdigits=3))\n" *
          "angle = $(round(ang, digits=2))°\n"
    annotate!(plt, (-1.02, 0.01, text(ann, ANNFS, :black, :left, :top)))

    savefig(plt, outfile)
    return plt
end

function overlap_barplot(pos1, pos2, neg1, neg2, outfile::AbstractString)
    ov_pos = rank_overlap(pos1, pos2)
    ov_neg = rank_overlap(neg1, neg2)

    jac_pos = jaccard_index(pos1, pos2)
    jac_neg = jaccard_index(neg1, neg2)

    vals = [ov_pos, ov_neg]
    labels = ["Top +η genes", "Top -η genes"]

    plt = bar(
        labels, vals;
        xlabel = "",
        ylabel = "Overlap count",
        color = RGB(0.45, 0.33, 0.67),
        alpha = 0.9,
        label = false,
        title = "",
        size = (900, 800),
        left_margin = 7mm,
        right_margin = 4mm,
        top_margin = 4mm,
        bottom_margin = 6mm
    )

    annotate!(plt, (1, vals[1] + 0.03*maximum(vals), text("J = $(round(jac_pos, digits=3))", ANNFS, :black, :center)))
    annotate!(plt, (2, vals[2] + 0.03*maximum(vals), text("J = $(round(jac_neg, digits=3))", ANNFS, :black, :center)))

    savefig(plt, outfile)
    return plt
end

# ============================================================
# LOAD INPUTS
# ============================================================

dir1 = length(ARGS) >= 1 ? ARGS[1] : DEFAULT_DIR1
dir2 = length(ARGS) >= 2 ? ARGS[2] : DEFAULT_DIR2

println("Loading:")
println("  1: $dir1")
println("  2: $dir2")

A1 = load_analysis_folder(dir1)
A2 = load_analysis_folder(dir2)

# ============================================================
# MERGE TABLES
# ============================================================

beta1  = rename_metric_df(A1.beta,  "beta",  "str")
beta2  = rename_metric_df(A2.beta,  "beta",  "hip")
gamma1 = rename_metric_df(A1.gamma, "gamma", "str")
gamma2 = rename_metric_df(A2.gamma, "gamma", "hip")
eta1   = rename_metric_df(A1.eta,   "eta",   "str")
eta2   = rename_metric_df(A2.eta,   "eta",   "hip")

merged = innerjoin(beta1, beta2, on=:gene)
merged = innerjoin(merged, gamma1, on=:gene)
merged = innerjoin(merged, gamma2, on=:gene)
merged = innerjoin(merged, eta1, on=:gene)
merged = innerjoin(merged, eta2, on=:gene)

merged_csv = joinpath(OUTCSV, "merged_gene_metrics_striatum_vs_hippocampus.csv")
CSV.write(merged_csv, merged)

println("Matched genes: $(nrow(merged))")

# ============================================================
# METRIC COMPARISONS
# ============================================================

βstats = compare_vectors(merged.beta_str, merged.beta_hip)
γstats = compare_vectors(merged.gamma_str, merged.gamma_hip)
ηstats = compare_vectors(merged.eta_str, merged.eta_hip)

# PCA comparison
pca_stats_z   = axis_comparison_stats(A1.pca, A2.pca; use_raw=false)
pca_stats_raw = axis_comparison_stats(A1.pca, A2.pca; use_raw=true)

# ============================================================
# TOP GENE OVERLAPS FOR ETA
# ============================================================

sort_idx_str_desc = sortperm(merged.eta_str, rev=true)
sort_idx_hip_desc = sortperm(merged.eta_hip, rev=true)
sort_idx_str_asc  = sortperm(merged.eta_str, rev=false)
sort_idx_hip_asc  = sortperm(merged.eta_hip, rev=false)

top_pos_str = merged.gene[sort_idx_str_desc[1:min(TOPN_OVERLAP, nrow(merged))]]
top_pos_hip = merged.gene[sort_idx_hip_desc[1:min(TOPN_OVERLAP, nrow(merged))]]
top_neg_str = merged.gene[sort_idx_str_asc[1:min(TOPN_OVERLAP, nrow(merged))]]
top_neg_hip = merged.gene[sort_idx_hip_asc[1:min(TOPN_OVERLAP, nrow(merged))]]

Ngenes = nrow(merged)
K = min(TOPN_OVERLAP, Ngenes)
ov_pos = rank_overlap(top_pos_str, top_pos_hip)
ov_neg = rank_overlap(top_neg_str, top_neg_hip)

ov_pos_p = hypergeom_overlap_pval(ov_pos, K, K, Ngenes)
ov_neg_p = hypergeom_overlap_pval(ov_neg, K, K, Ngenes)

overlap_df = DataFrame(
    set = ["top_positive_eta", "top_negative_eta"],
    top_n = [K, K],
    overlap = [ov_pos, ov_neg],
    jaccard = [jaccard_index(top_pos_str, top_pos_hip), jaccard_index(top_neg_str, top_neg_hip)],
    hypergeom_p = [ov_pos_p, ov_neg_p]
)

CSV.write(joinpath(OUTCSV, "eta_top_gene_overlap_summary.csv"), overlap_df)

CSV.write(joinpath(OUTCSV, "top_positive_eta_genes_striatum.csv"), DataFrame(gene=top_pos_str))
CSV.write(joinpath(OUTCSV, "top_positive_eta_genes_hippocampus.csv"), DataFrame(gene=top_pos_hip))
CSV.write(joinpath(OUTCSV, "top_negative_eta_genes_striatum.csv"), DataFrame(gene=top_neg_str))
CSV.write(joinpath(OUTCSV, "top_negative_eta_genes_hippocampus.csv"), DataFrame(gene=top_neg_hip))

# ============================================================
# PLOTS
# ============================================================

plt_beta, _ = scatter_compare_plot(
    merged.beta_str, merged.beta_hip;
    xlabel = "$(LABEL1) gene coefficient for z(β)",
    ylabel = "$(LABEL2) gene coefficient for z(β)",
    outfile = joinpath(OUTFIG, "compare_beta_coefficients_striatum_vs_hippocampus.png")
)

plt_gamma, _ = scatter_compare_plot(
    merged.gamma_str, merged.gamma_hip;
    xlabel = "$(LABEL1) gene coefficient for z(γ)",
    ylabel = "$(LABEL2) gene coefficient for z(γ)",
    outfile = joinpath(OUTFIG, "compare_gamma_coefficients_striatum_vs_hippocampus.png"),
    markercolor = RGB(0.60, 0.20, 0.22)
)

plt_eta, _ = scatter_compare_plot(
    merged.eta_str, merged.eta_hip;
    xlabel = "$(LABEL1) gene correlation with η",
    ylabel = "$(LABEL2) gene correlation with η",
    outfile = joinpath(OUTFIG, "compare_eta_gene_correlations_striatum_vs_hippocampus.png"),
    markercolor = RGB(0.35, 0.30, 0.58)
)

plt_pca_z = pca_axis_plot(
    pca_stats_z.v1, pca_stats_z.v2, pca_stats_z.evr1, pca_stats_z.evr2,
    joinpath(OUTFIG, "compare_pc1_axes_striatum_vs_hippocampus_zspace.png");
    use_raw = false
)

plt_pca_raw = pca_axis_plot(
    pca_stats_raw.v1, pca_stats_raw.v2, pca_stats_raw.evr1, pca_stats_raw.evr2,
    joinpath(OUTFIG, "compare_pc1_axes_striatum_vs_hippocampus_rawspace.png");
    use_raw = true
)

pca_panel = plot(
    plt_pca_z, plt_pca_raw;
    layout = (1, 2),
    size = (1800, 800)
)
savefig(pca_panel, joinpath(OUTFIG, "compare_pc1_axes_striatum_vs_hippocampus_zspace_and_rawspace.png"))

plt_overlap = overlap_barplot(
    top_pos_str, top_pos_hip, top_neg_str, top_neg_hip,
    joinpath(OUTFIG, "compare_eta_top_gene_overlaps_striatum_vs_hippocampus.png")
)

# combined panel
panel = plot(
    plt_beta, plt_gamma, plt_eta, plt_pca_z;
    layout = (2,2),
    size = (1800, 1600)
)
savefig(panel, joinpath(OUTFIG, "comparison_panel_striatum_vs_hippocampus.png"))

# ============================================================
# SUMMARY TEXT
# ============================================================

open(joinpath(OUTTXT, "comparison_summary.txt"), "w") do io
    println(io, "Comparison of gene/PCA analyses")
    println(io)
    println(io, "Input folders:")
    println(io, "  Striatum    = ", dir1)
    println(io, "  Hippocampus = ", dir2)
    println(io)
    println(io, "Matched genes = ", nrow(merged))
    println(io)

    println(io, "[Beta coefficient comparison]")
    println(io, "  Pearson r   = ", βstats.pearson_r)
    println(io, "  Pearson p   = ", βstats.pearson_p)
    println(io, "  Spearman ρ  = ", βstats.spearman_r)
    println(io, "  Spearman p  = ", βstats.spearman_p)
    println(io, "  Sign agreement = ", βstats.concordance)
    println(io, "  OLS slope through origin = ", βstats.slope_origin)
    println(io, "  OLS slope with intercept = ", βstats.slope_ols)
    println(io, "  OLS intercept = ", βstats.intercept_ols)
    println(io, "  MAE = ", βstats.mae)
    println(io, "  RMSE = ", βstats.rmse)
    println(io)

    println(io, "[Gamma coefficient comparison]")
    println(io, "  Pearson r   = ", γstats.pearson_r)
    println(io, "  Pearson p   = ", γstats.pearson_p)
    println(io, "  Spearman ρ  = ", γstats.spearman_r)
    println(io, "  Spearman p  = ", γstats.spearman_p)
    println(io, "  Sign agreement = ", γstats.concordance)
    println(io, "  OLS slope through origin = ", γstats.slope_origin)
    println(io, "  OLS slope with intercept = ", γstats.slope_ols)
    println(io, "  OLS intercept = ", γstats.intercept_ols)
    println(io, "  MAE = ", γstats.mae)
    println(io, "  RMSE = ", γstats.rmse)
    println(io)

    println(io, "[Eta gene-correlation comparison]")
    println(io, "  Pearson r   = ", ηstats.pearson_r)
    println(io, "  Pearson p   = ", ηstats.pearson_p)
    println(io, "  Spearman ρ  = ", ηstats.spearman_r)
    println(io, "  Spearman p  = ", ηstats.spearman_p)
    println(io, "  Sign agreement = ", ηstats.concordance)
    println(io, "  OLS slope through origin = ", ηstats.slope_origin)
    println(io, "  OLS slope with intercept = ", ηstats.slope_ols)
    println(io, "  OLS intercept = ", ηstats.intercept_ols)
    println(io, "  MAE = ", ηstats.mae)
    println(io, "  RMSE = ", ηstats.rmse)
    println(io)

    println(io, "[PC1 axis comparison: z space]")
    println(io, "  Striatum PC1 = ", pca_stats_z.v1)
    println(io, "  Hippocampus PC1 (oriented) = ", pca_stats_z.v2)
    println(io, "  Cosine similarity = ", pca_stats_z.cos_sim)
    println(io, "  Angle (deg) = ", pca_stats_z.angle_deg)
    println(io, "  Striatum PC1 explained variance = ", pca_stats_z.evr1)
    println(io, "  Hippocampus PC1 explained variance = ", pca_stats_z.evr2)
    println(io)

    println(io, "[PC1 axis comparison: raw space]")
    println(io, "  Striatum PC1 = ", pca_stats_raw.v1)
    println(io, "  Hippocampus PC1 (oriented) = ", pca_stats_raw.v2)
    println(io, "  Cosine similarity = ", pca_stats_raw.cos_sim)
    println(io, "  Angle (deg) = ", pca_stats_raw.angle_deg)
    println(io, "  Striatum PC1 explained variance = ", pca_stats_raw.evr1)
    println(io, "  Hippocampus PC1 explained variance = ", pca_stats_raw.evr2)
    println(io)

    println(io, "[Top eta gene overlap]")
    println(io, "  Top N = ", K)
    println(io, "  Positive eta overlap = ", ov_pos)
    println(io, "  Positive eta Jaccard = ", jaccard_index(top_pos_str, top_pos_hip))
    println(io, "  Positive eta hypergeom p = ", ov_pos_p)
    println(io, "  Negative eta overlap = ", ov_neg)
    println(io, "  Negative eta Jaccard = ", jaccard_index(top_neg_str, top_neg_hip))
    println(io, "  Negative eta hypergeom p = ", ov_neg_p)
    println(io)

    println(io, "Output files:")
    println(io, "  merged CSV = ", merged_csv)
    println(io, "  figures dir = ", OUTFIG)
end

println("\nDone.")
println("Outputs written under: $OUTROOT")