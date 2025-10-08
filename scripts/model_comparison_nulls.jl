using PathoSpread
using Serialization, Statistics
using CairoMakie, Colors, Printf

# ───────────────────────────────────────────────────────────────
# SETTINGS
# ───────────────────────────────────────────────────────────────
mode = :seed   # or :seed
sim_true = "simulations/DIFFGA_RETRO.jls"
out_pdf  = "figures/model_comparison/nulls/DIFFGA_$(String(mode))_WAIC_box.pdf"

# Automatically collect all relevant simulation files
all_files = readdir("simulations"; join=true)

if mode == :shuffle
    sim_paths = filter(f -> occursin(r"DIFFGA_shuffle_\d+$", splitext(basename(f))[1]), all_files)
elseif mode == :seed
    sim_paths = filter(f -> occursin(r"DIFFGA_seed_\d+$", splitext(basename(f))[1]), all_files)
else
    error("Unknown mode: $mode. Must be :shuffle or :seed.")
end

println("Found $(length(sim_paths)) $(String(mode)) models.")

# ───────────────────────────────────────────────────────────────
# LOAD WAIC VALUES
# ───────────────────────────────────────────────────────────────
function get_waic_values(sim_paths; S=300)
    vals = Float64[]
    for sp in sim_paths
        inf = load_inference(sp)
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
true_inf   = load_inference(sim_true)
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


# --- Save ---
mkpath(dirname(out_pdf))
save(out_pdf, fig)
fig


# ───────────────────────────────────────────────────────────────
# ADDITIONAL ANALYSIS: Distance from true seed vs ΔWAIC
# ───────────────────────────────────────────────────────────────
if mode == :seed
    using Graphs, LinearAlgebra, DelimitedFiles
    using SimpleWeightedGraphs

    println("\nComputing seed distance vs ΔWAIC relationship...")

    # Load reference Laplacian / adjacency matrix
    # (Adjust this line according to your connectivity file)
    W_labels = readdlm("data/W_labeled_filtered.csv", ',')  # e.g., your PathoSpread helper
    W = Array{Float64}(W_labels[2:end, 2:end]) 
    W ./= maximum(W[W .> 0])

    # take reciprocal of weights (larger weights means closer distance)
    W = 1.0 ./ (W .+ eps())

    g = SimpleWeightedDiGraph(W)

    # Identify seed indices directly from inference files
    ref_idx = true_inf["seed_idx"]  # reference model’s seed index

    seed_indices = []
    for sp in sim_paths
        inf = load_inference(sp)
        if haskey(inf, "seed_idx")
            push!(seed_indices, inf["seed_idx"])
        else
            @warn "Missing seed_idx in $sp — skipping"
        end
    end

    # Ensure they're integers
    seed_indices = Int.(seed_indices)

    # Compute all-pairs shortest paths (using edge weights as distances)
    D = floyd_warshall_shortest_paths(g).dists

    # Compute distance to true seed for each null model
    dist_to_ref = [D[ref_idx, s] for s in seed_indices]

    # Compute ΔWAIC relative to true model
    delta_waic = waic_nulls .- true_waic

    # ─── Plot relationship ─────────────────────────────────────
    out_pdf2 = replace(out_pdf, "_WAIC_box.pdf" => "_seed_distance_vs_WAIC.pdf")
    fig2 = Figure(size=(700,500));
    ax2 = Axis(fig2[1,1];
        xlabel = "Shortest-path distance to true seed",
        ylabel = "ΔWAIC (seed − true)",
        titlesize = 28, xlabelsize = 24, ylabelsize = 24,
        yticklabelsize = 18, xticklabelsize = 18);

    scatter!(ax2, dist_to_ref, delta_waic;
             color = RGBf(0.2, 0.2, 0.2), markersize = 12, alpha = 0.8)

    mkpath(dirname(out_pdf2))
    save(out_pdf2, fig2)
    println("Saved seed-distance vs ΔWAIC figure → $out_pdf2")
end
fig2