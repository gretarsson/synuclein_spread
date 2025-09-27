using Serialization
include("helpers.jl")
using PrettyTables, DataFrames

# Read inference results
simulations = [
    #"simulations/DIFF_EUCL",
    #"simulations/DIFF_ANTERO",
    #"simulations/DIFF_RETRO",
    #"simulations/DIFF_BIDIR",
    #
    #"simulations/DIFFG_EUCL",
    #"simulations/DIFFG_ANTERO",
    #"simulations/DIFFG_RETRO",
    #"simulations/DIFFG_BIDIR",
    ##
    #"simulations/DIFFGA_EUCL",
    #"simulations/DIFFGA_ANTERO",
    #"simulations/DIFFGA_RETRO",
    #"simulations/DIFFGA_BIDIR",
    ##
    #"simulations/DIFFGAM_ANTERO",
    #"simulations/DIFFGAM_RETRO",
    #"simulations/DIFFGAM_BIDIR",
    #
    "simulations/DIFF_RETRO",
    "simulations/DIFFG_RETRO",
    "simulations/DIFFGA_RETRO",
    "simulations/DIFFGAM_RETRO",

]
model_names = [
    #"DIFF euclidean", 
    #"DIFF anterograde", 
    #"DIFF retrograde", 
    #"DIFF bidirectional", 
    #
    #"DIFFG euclidean", 
    #"DIFFG anterograde", 
    #"DIFFG retrograde", 
    #"DIFFG bidirectional", 
    ##
    #"DIFFGA euclidean", 
    #"DIFFGA anterograde", 
    #"DIFFGA retrograde", 
    #"DIFFGA bidirectional", 
    ##
    #"DIFFGAM anterograde", 
    #"DIFFGAM retrograde", 
    #"DIFFGAM bidirectional", 
    #
    "DIFF retrograde", 
    "DIFFG retrograde", 
    "DIFFGA retrograde", 
    "DIFFGAM retrograde", 
]
inferences = []
for simulation in simulations
    push!(inferences, deserialize(simulation * ".jl"))
end

# Compute WAIC, AIC, BIC, MSE, and Frobenius covariance norm for models
waic_vals = Float64[]
se_waic_vals = Float64[]
aic_vals  = Float64[]
bic_vals  = Float64[]
mse_vals  = Float64[]
covnorm_vals = Float64[]
regcov = []

# needed for paired WAIC scores
waic_i_list = Vector{Vector{Float64}}()
n_useds     = Int[]
pwaic_vals  = Float64[]   # optional: if you returned p_waic

for inference in inferences
    waic, se_waic, waic_i, lppd, p_waic, n_used = compute_waic(inference; S=300)
    push!(waic_vals, waic)
    push!(se_waic_vals, se_waic)
    push!(waic_i_list, waic_i)
    push!(n_useds, n_used)
    aic, bic = compute_aic_bic(inference)
    push!(aic_vals, aic)
    push!(bic_vals, bic)
    mse = compute_mse_mc(inference)
    push!(mse_vals, mse)
    regional_cov = compute_regional_correlations(inference)
    #covnorm = mean(abs.(regional_cov))  # avg |r|
    covnorm = mean((regional_cov).^2)  # avg R^2
    push!(covnorm_vals, covnorm)
    push!(regcov, regional_cov)
end

# Compute delta metrics relative to the best (lowest) value
min_waic = minimum(waic_vals)
min_aic  = minimum(aic_vals)
min_bic  = minimum(bic_vals)
min_mse  = minimum(mse_vals)

# Handle the case where all covnorm_vals are NaN
valid_cov = filter(!isnan, covnorm_vals)  # values are already ≥0; no need for abs here
if isempty(valid_cov)
    min_cov   = NaN
    delta_cov = fill(NaN, length(covnorm_vals))
else
    min_cov   = minimum(valid_cov)
    delta_cov = [c - min_cov for c in covnorm_vals]  # NaN stays NaN here automatically
end

delta_waic = [w - min_waic for w in waic_vals]
delta_aic  = [a - min_aic for a in aic_vals]
delta_bic  = [b - min_bic for b in bic_vals]
delta_mse  = [m - min_mse for m in mse_vals]
delta_cov  = [c - min_cov for c in covnorm_vals]

# Build a DataFrame to display the results
df = DataFrame(
    Model   = model_names,
    WAIC    = round.(waic_vals, digits=0),
    ∆WAIC   = round.(delta_waic, digits=0),
    AIC     = round.(aic_vals, digits=0),
    ∆AIC    = round.(delta_aic, digits=0),
    BIC     = round.(bic_vals, digits=0),
    ∆BIC    = round.(delta_bic, digits=0),
    MSE     = round.(mse_vals, digits=6),
    ∆MSE    = round.(delta_mse, digits=6),
    ParCor = round.(covnorm_vals, digits=4),
    ∆ParCor   = round.(delta_cov, digits=4)
)

# Print LaTeX table
pretty_table(df; formatters = ft_printf("%5d"), backend = Val(:latex))

# ─── reorder DataFrame so deltas come last ─────────────────────────────────────
ordered = [
  :Model,
  :WAIC, :AIC, :BIC,      # main big metrics
  :MSE,  :ParCor,         # main small metrics
  :∆WAIC, :∆AIC, :∆BIC,   # deltas for the big metrics
  :∆MSE,  :∆ParCor        # deltas for the small metrics
]
df2 = df[:, ordered]

# ─── print LaTeX table with mixed formatting ───────────────────────────────────
pretty_table(
  df2;
  formatters = (
    ft_printf("%s",        1),     # Model (string)
    ft_printf("%7.0f",   2:4),     # WAIC, AIC, BIC  (zero‑decimal floats)
    ft_printf("%.2e",    5:6),     # MSE, ParCor     (sci‑notation 6 d.p.)
    ft_printf("%7.0f",   7:9),     # ∆WAIC, ∆AIC, ∆BIC (zero‑decimal floats)
    ft_printf("%.2e",     10),     # ∆MSE           (sci‑notation 6 d.p.)
    ft_printf("%.2e",     11)      # ∆ParCor        (sci‑notation 4 d.p.)
  ),
  backend = Val(:latex),
)



# PAIRED WAIC COMPARISON
# ─── Paired ΔWAIC ± SE(Δ) vs the best model ───────────────────────────────────
using Statistics

best_ix = argmin(waic_vals)
ref_waic_i = waic_i_list[best_ix]
n = length(ref_waic_i)
@assert all(length(wi) == n for wi in waic_i_list)

delta_waic_paired = Float64[]
se_delta_waic     = Float64[]
for j in eachindex(waic_i_list)
    if j == best_ix
        push!(delta_waic_paired, 0.0)
        push!(se_delta_waic, 0.0)
    else
        d_i = waic_i_list[j] .- ref_waic_i
        Δ   = sum(d_i)
        SEΔ = sqrt(n * var(d_i))
        push!(delta_waic_paired, Δ)
        push!(se_delta_waic, SEΔ)
    end
end

# Build with simple column names, then set pretty headers in pretty_table
df_pairs = DataFrame(
    Model = model_names,
    ΔWAIC_vs_best = round.(delta_waic_paired, digits=1),
    SE_ΔWAIC      = round.(se_delta_waic, digits=1),
)

pretty_table(
    df_pairs;
    header = ["Model", "ΔWAIC (vs best)", "SE(ΔWAIC)"],
    backend = Val(:latex),
)



using CairoMakie, Statistics, Printf, Colors

# classify with 95% bands
low  = delta_waic_paired .- 2 .* se_delta_waic
high = delta_waic_paired .+ 2 .* se_delta_waic
best_ix = argmin(waic_vals)

classes = map(1:length(model_names)) do i
    if i == best_ix
        :best
    elseif low[i] <= 0.0 <= high[i]
        :tied
    else
        :worse
    end
end

# aesthetics
blue  = RGBf(0/255,71/255,171/255);
red   = RGBf(185/255,40/255,40/255);
grayc = RGBf(0.35,0.35,0.35);

marker_for(c) = c===:best ? :star5 : :circle
color_for(c)  = c===:best ? blue : (c===:tied ? grayc : red)

# order rows by Δ (best first if ties exist)
ord = sortperm(delta_waic_paired)
ys  = collect(1:length(ord))

fig = Figure(resolution=(1100, 350 + 44length(model_names)), figure_padding = (20, 20, 20, 20));
ax  = Axis(fig[1,1];
    title="ΔWAIC ± 2·SE(Δ) vs best",
    titlesize=25,
    xlabel="ΔWAIC",
    xlabelsize=24,
    xticklabelsize=24,
    yticklabelsize=26,
    yticks=(ys, model_names[ord]),
)

for (row, idx) in enumerate(ord)
    Δ   = delta_waic_paired[idx]
    SE2 = 2*se_delta_waic[idx]
    c   = classes[idx]
    y   = ys[row]

    # draw error bar for non-best only
    if c !== :best
        errorbars!(ax, [Δ], [y], [SE2], [SE2];
            direction=:x, whiskerwidth=25, linewidth=10, color=color_for(c), alpha=0.9)
    end

    # point (best = big blue star only)
    CairoMakie.scatter!(ax, [Δ], [y];
        markersize = c===:best ? 45 : 25,
        marker     = marker_for(c),
        color      = color_for(c),
        strokecolor=:black, strokewidth=1.2)

    # label "Δ ± 2SE" offset up & right (to avoid overlap)
    txt = @sprintf("%.0f ± %.0f", Δ, SE2)
    text!(ax, Δ, y;
        text=txt,
        align=(:left,:bottom),
        offset=(14, 10),          # ← right & up
        fontsize=24,
        color=:black)
end

# reference at zero
vlines!(ax, [0.0]; color=:gray, linestyle=:dash, linewidth=4)
# Add headroom inside the axis so top text isn’t cut off
Makie.ylims!(ax, 0.5, length(model_names) + 0.5)

# compute tight bounds from the drawn bars
xmax = maximum(delta_waic_paired .+ 2 .* se_delta_waic)
xmin = min(0.0, minimum(delta_waic_paired .- 2 .* se_delta_waic))
# add a sensible right pad (>= 50 units or 6% of span)
span = max(xmax - xmin, 1e-9)
pad  = max(0.2 * span, 50.0)
Makie.xlims!(ax, xmin-0.1*pad, xmax + pad)

# save figure
fig
save("figures/model_comparison/models_delta_waic_vs_best.pdf", fig)
