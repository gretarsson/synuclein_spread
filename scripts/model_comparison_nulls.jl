using PathoSpread
using Serialization, Statistics
using CairoMakie, Colors, Printf

# ───────────────────────────────────────────────────────────────
# SETTINGS
# ───────────────────────────────────────────────────────────────
mode = :seed   # or :seed
sim_true = "simulations/DIFFGA_RETRO"
out_pdf  = "figures/model_comparison/nulls/DIFFGA_$(String(mode))_WAIC_box.pdf"

sim_paths = mode == :shuffle ? [
    "simulations/DIFFGA_shuffle_2",
    "simulations/DIFFGA_shuffle_8",
    "simulations/DIFFGA_shuffle_12",
    "simulations/DIFFGA_shuffle_13",
    "simulations/DIFFGA_shuffle_16",
    "simulations/DIFFGA_shuffle_18",
    "simulations/DIFFGA_shuffle_20",
    "simulations/DIFFGA_shuffle_21",
    "simulations/DIFFGA_shuffle_24",
    "simulations/DIFFGA_shuffle_27",
    "simulations/DIFFGA_shuffle_31",
    "simulations/DIFFGA_shuffle_34",
    "simulations/DIFFGA_shuffle_35",
    "simulations/DIFFGA_shuffle_36",
    "simulations/DIFFGA_shuffle_39",
    "simulations/DIFFGA_shuffle_40",
    "simulations/DIFFGA_shuffle_41",
    "simulations/DIFFGA_shuffle_42",
    "simulations/DIFFGA_shuffle_46",
    "simulations/DIFFGA_shuffle_47",
    "simulations/DIFFGA_shuffle_57",
    "simulations/DIFFGA_shuffle_61",
    "simulations/DIFFGA_shuffle_66",
    "simulations/DIFFGA_shuffle_68",
    "simulations/DIFFGA_shuffle_88",
    "simulations/DIFFGA_shuffle_89",
    "simulations/DIFFGA_shuffle_92",
    "simulations/DIFFGA_shuffle_97",
    "simulations/DIFFGA_shuffle_99",
] : [
    "simulations/DIFFGA_seed_4",
    "simulations/DIFFGA_seed_5",
    "simulations/DIFFGA_seed_6",
    "simulations/DIFFGA_seed_26",
    "simulations/DIFFGA_seed_67",
    "simulations/DIFFGA_seed_71",
    "simulations/DIFFGA_seed_80",
    "simulations/DIFFGA_seed_81",
    "simulations/DIFFGA_seed_88",
    "simulations/DIFFGA_seed_95",
    "simulations/DIFFGA_seed_98",
    "simulations/DIFFGA_seed_104",
    "simulations/DIFFGA_seed_105",
]

# ───────────────────────────────────────────────────────────────
# LOAD WAIC VALUES
# ───────────────────────────────────────────────────────────────
function get_waic_values(sim_paths; S=300)
    vals = Float64[]
    for sp in sim_paths
        inf = load_inference(sp * ".jls")
        waic, _, _, _, _, _ = compute_waic(inf; S=S)
        if isfinite(waic)
            push!(vals, waic)
        else
            @warn "Skipping non-finite WAIC for $sp"
        end
    end
    return vals
end

waic_nulls = get_waic_values(sim_paths; S=300)
true_inf   = load_inference(sim_true * ".jls")
true_waic, _, _, _, _, _ = compute_waic(true_inf; S=300)

println(@sprintf("Mean shuffle WAIC = %.2f ± %.2f", mean(waic_nulls), std(waic_nulls)))
println(@sprintf("True model WAIC   = %.2f", true_waic))

# ───────────────────────────────────────────────────────────────
# PLOT
# ───────────────────────────────────────────────────────────────
waic_nulls = filter(<(0), waic_nulls)
c_null = RGBf(0.4, 0.4, 0.4);
c_true = RGBf(0/255, 71/255, 171/255);

fig = Figure(size=(500,600));
ax = Axis(fig[1,1];
    ylabel="WAIC",
    xticks=([], []),
    titlesize=26, ylabelsize=34, yticklabelsize=24)

# --- Boxplot ---
group = ones(length(waic_nulls))
boxplot!(ax, group, waic_nulls;
    color=c_null, mediancolor=:black,
    whiskercolor=:black, whiskerlinewidth=5, medianlinewidth=5,
    strokecolor=:black, outliercolor=:transparent,
    show_notch=false, show_outliers=false, width=0.2)

# --- Scatter individual null points ---
scatter!(ax, 1 .+ 0.1 .* (rand(length(waic_nulls)) .- 0.5),
         waic_nulls; color=:black, alpha=0.7, markersize=18)

# --- True model ---
scatter!(ax, [1.0], [true_waic];
         color=c_true, marker=:star5, markersize=36)

# --- Adjust limits (automatic zoom) ---
#Makie.autolimits!(ax)
Makie.xlims!(ax, 0.85, 1.15)  # e.g. (0.9, 1.1) for very tight framing


# Compute percentile & p-value
pval = (1 + count(x -> x <= true_waic, waic_nulls)) / (1 + length(waic_nulls))
percentile = 100 * pval

# Compose label text
if pval <= (1 / (length(waic_nulls) + 1))
    label_text = @sprintf("Best model\np < %.3f", 1 / (length(waic_nulls) + 1))
else
    label_text = @sprintf("Percentile %.1f%%\np = %.3f", percentile, pval)
end

# --- Annotation (above the star, multiline, centered) ---
text!(ax, 1.0, true_waic;
      text = label_text,
      align = (:left, :bottom),
      offset = (20, 0),          # 20px above the star
      fontsize = 24,
      color = :black)

# Optional top label
#text!(ax, 1.0, maximum(waic_nulls)*1.02;
#      text = @sprintf("Empirical p = %.3f", pval),
#      align = (:center,:bottom), fontsize=18, color=:black)


# --- Save ---
mkpath(dirname(out_pdf))
save(out_pdf, fig)
fig