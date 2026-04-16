#!/usr/bin/env julia

using Random
using LinearAlgebra
using Statistics
using CairoMakie
using MultivariateStats

# ============================================================
# SETTINGS
# ============================================================

Random.seed!(12)

n_genes = 2500

# Desired dominant direction in (beta, gamma) space
# Example: stronger fall, weaker growth  ~  (-0.73, 0.68)
pc1_true = [-0.73, 0.68]
pc1_true ./= norm(pc1_true)

# Orthogonal direction
pc2_true = [-pc1_true[2], pc1_true[1]]
pc2_true ./= norm(pc2_true)

# Variance along PC1 vs PC2
σ1 = 0.18   # spread along dominant axis
σ2 = 0.06   # spread orthogonal to dominant axis

# Optional isotropic noise
σ_noise = 0.015

output_png = "figures/overview/toy_pca_beta_gamma_space.png"

# ============================================================
# GENERATE TOY DATA
# ============================================================

# Scores along the two latent axes
s1 = σ1 .* randn(n_genes)
s2 = σ2 .* randn(n_genes)

# Construct point cloud in (beta, gamma)-coefficient space
X = hcat(s1, s2) * hcat(pc1_true, pc2_true)' .+
    σ_noise .* randn(n_genes, 2)

x = X[:, 1]   # toy coefficient for z(beta)
y = X[:, 2]   # toy coefficient for z(gamma)

# ============================================================
# PCA
# ============================================================

# MultivariateStats expects variables in rows, observations in columns
M = permutedims(X)  # 2 x n_genes

pca_model = fit(PCA, M; maxoutdim=2)

# Principal directions are columns
P = projection(pca_model)   # 2 x 2
pc1 = P[:, 1]
pc2 = P[:, 2]

# Fix sign of PC1 so it roughly matches the intended direction
if dot(pc1, pc1_true) < 0
    pc1 .*= -1
end
if det(hcat(pc1, pc2)) < 0
    pc2 .*= -1
end

# Explained variance ratio
λ = principalvars(pca_model)
explained = λ ./ sum(λ)

println("PC1 = ", round.(pc1, digits=3))
println("PC2 = ", round.(pc2, digits=3))
println("Explained variance ratio = ", round.(explained, digits=3))

# ============================================================
# PLOT
# ============================================================

# Axis limits
xmin, xmax = extrema(x)
ymin, ymax = extrema(y)

pad_x = 0.08 * (xmax - xmin)
pad_y = 0.08 * (ymax - ymin)

xmin -= pad_x
xmax += pad_x
ymin -= pad_y
ymax += pad_y

# Center for PCA lines
μx = mean(x)
μy = mean(y)

# Line lengths
L1 = 0.55 * min(xmax - xmin, ymax - ymin)
L2 = 0.35 * min(xmax - xmin, ymax - ymin)

fig = Figure(size = (1000, 700))
ax = Axis(
    fig[1, 1],
    xlabel = "corr(gene,β)",
    ylabel = "corr(gene,γ)",
    xlabelsize = 60,
    ylabelsize = 60,
    xticklabelsize = 40,
    yticklabelsize = 40,
    xgridvisible = true,
    ygridvisible = true,
)

# Scatter cloud
scatter!(
    ax, x, y;
    markersize = 45,
    alpha = 0.45
)

# Dashed zero axes
hlines!(ax, [0.0], color = :gray, linestyle = :dash, linewidth = 2)
vlines!(ax, [0.0], color = :gray, linestyle = :dash, linewidth = 2)

# PC2 line
#lines!(
#    ax,
#    [μx - L2 * pc2[1], μx + L2 * pc2[1]],
#    [μy - L2 * pc2[2], μy + L2 * pc2[2]];
#    color = :gray40,
#    linewidth = 16,
#    label = "PC2"
#)

# PC1 line
lines!(
    ax,
    [μx - L1 * pc1[1], μx + L1 * pc1[1]],
    [μy - L1 * pc1[2], μy + L1 * pc1[2]];
    color = :black,
    linewidth = 20,
    label = "PC1"
)


axislegend(ax; position = :rt, labelsize = 45)

xlims!(ax, xmin, xmax)
ylims!(ax, ymin, ymax)

# Text annotation
txt = "PC1 = ($(round(pc1[1], digits=2)), $(round(pc1[2], digits=2)))\n" *
      "var explained = $(round(100 * explained[1], digits=1))%"

#text!(
#    ax,
#    xmin + 0.04 * (xmax - xmin),
#    ymax - 0.08 * (ymax - ymin),
#    text = txt,
#    align = (:left, :top),
#    fontsize = 18
#)

save(output_png, fig)
println("Saved figure to: $output_png")