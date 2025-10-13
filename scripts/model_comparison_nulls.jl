#!/usr/bin/env julia
using PathoSpread, Serialization, Statistics, CairoMakie, Colors, Printf, MCMCChains, Serialization, StatsBase
using Random
Random.seed!(12345)


mode = :seed
sim_true = "simulations/DIFFGA_seed_74.jls"
waic_cache_file = "results/waic_cache/DIFFGA_$(String(mode))_waic_all.jls"
waic_cache_dir  = "results/waic_cache"
out_pdf = "figures/model_comparison/nulls/DIFFGA_$(String(mode))_WAIC_box.pdf"

# Load WAICs
waic_all = deserialize(waic_cache_file)

# Rebuild list of paths
all_files = readdir("simulations"; join=true)
sim_paths = if mode == :seed
    filter(f -> occursin(r"DIFFGA_seed_\d+$", splitext(basename(f))[1]), all_files)
else
    filter(f -> occursin(r"DIFFGA_shuffle_\d+$", splitext(basename(f))[1]), all_files)
end


# Remove the corresponding file (e.g., DIFFGA_seed_74.jls)
sim_paths = filter(f -> !occursin(r"DIFFGA_seed_74", f), sim_paths)

# Make sure cache and file list align by basename
names_paths = basename.(sim_paths)
cache_len = length(waic_all)

if cache_len != length(names_paths)
    @warn "Length mismatch: $(cache_len) cached WAICs vs $(length(names_paths)) sim_paths."
end

function is_converged_param(inf;
    param_index::Int = 1,
    rhat_thresh::Real = 1.4,
    ess_thresh::Real = 50.)

    ch = inf["chain"]
    diag = MCMCChains.ess_rhat(ch)

    # Extract diagnostics
    rhat_vals = collect(skipmissing(diag.nt.rhat))
    ess_bulk  = hasproperty(diag.nt, :ess_bulk) ?
                    collect(skipmissing(diag.nt.ess_bulk)) :
                    collect(skipmissing(diag.nt.ess))
    ess_tail  = hasproperty(diag.nt, :ess_tail) ?
                    collect(skipmissing(diag.nt.ess_tail)) :
                    ess_bulk

    # Handle invalid index
    if param_index > length(rhat_vals)
        error("Parameter index $(param_index) exceeds available parameters ($(length(rhat_vals)))")
    end

    return (rhat_vals[param_index] < rhat_thresh) &&
           (ess_bulk[param_index] > ess_thresh) &&
           (ess_tail[param_index] > ess_thresh)
end
"""
check_prior_update(inf; param_index=1, delta_thresh=0.3, shrink_thresh=0.95)

Returns `true` if the specified parameter shows meaningful update from its prior.
Uses normalized mean/median shift and posterior shrinkage.
"""
function check_prior_update(inf; param_index=1, delta_thresh=1., shrink_thresh=0.95)
    ch = inf["chain"]
    arr = Array(ch)
    post_med = vec(mapslices(median, arr; dims=1))
    post_sd  = vec(mapslices(std, arr; dims=1))

    pri = inf["priors"]
    priors_vec = collect(values(pri))  # assumes priors stored in consistent order

    # Fallback if priors_vec too short
    if param_index > length(priors_vec)
        @warn "Parameter index $param_index exceeds number of priors ($(length(priors_vec)))"
        return false
    end

    prior_dist = priors_vec[param_index]

    μ₀ = mean(prior_dist)
    σ₀ = std(prior_dist)
    μp = post_med[param_index]
    σp = post_sd[param_index]

    Δ = abs(μp - μ₀) / σ₀
    shrink = σp / σ₀

    return (Δ > delta_thresh) && (shrink < shrink_thresh)
end

# Build dictionary {basename => WAIC}
# This assumes your caching script saved values in same order as sim_paths at the time.
waic_dict = Dict(names_paths[i] => waic_all[i] for i in 1:min(length(names_paths), length(waic_all)))

# Filter only those with a cached WAIC
sim_paths = filter(sp -> haskey(waic_dict, basename(sp)), sim_paths)

# Now perform convergence filtering
using Base.Threads

keep = falses(length(sim_paths))
@threads for i in eachindex(sim_paths)
    sp = sim_paths[i]
    inf = load_inference(sp)
    # (1) convergence (as you prefer, e.g., is_converged or is_converged_param)
    conv = is_converged_param(inf; param_index=1, rhat_thresh=1.4, ess_thresh=100)

    # (2) prior update
    if mode == :seed
        updated = check_prior_update(inf; param_index=1, delta_thresh=0.2, shrink_thresh=0.90)
    else
        updated = true
    end

    keep[i] = conv && updated
    #if mode == :shuffle  # don't bother with this for shuffle
    #    keep[i] = true
    #end
end

sim_paths = sim_paths[keep]
waic_nulls = [waic_dict[basename(sp)] for sp in sim_paths if isfinite(waic_dict[basename(sp)])]
println("→ Using $(length(waic_nulls)) converged WAICs out of $(length(waic_dict)) cached values.")


# Load true model WAIC
true_inf = load_inference(sim_true)
true_waic, _, _, _, _, _ = compute_waic(true_inf; S=1000)



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
         waic_nulls; color=:black, alpha=0.7, markersize=30)

# --- True model ---
#scatter!(ax, [1.0], [true_waic];
#         color=c_true, marker=:star5, markersize=36)
scatter!(ax, [1.0], [true_waic];
         color=c_true, markersize=30)

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
#text!(ax, 1.0, true_waic;
#      text = label_text,
#      align = (:left, :bottom),
#      offset = (20, 0),          # 20px above the star
#      fontsize = 24,
#      color = :black)


# --- Save ---
mkpath(dirname(out_pdf))
save(out_pdf, fig)
fig

# ───────────────────────────────────────────────
# Save a clipped version (1–99 percentile y-range)
# ───────────────────────────────────────────────
if mode == :seed
    # Sort WAIC values
    sorted = sort(waic_nulls)

    # Drop only the top 2%
    n = length(sorted)
    cut = max(1, round(Int, 0.05 * n))  # number of points to drop from top

    hi = sorted[end - cut]               # empirical 98% cutoff
    lo = minimum(sorted)                 # keep full lower range

    # Add small padding (e.g. 5%)
    span = hi - lo
    pad = 0.05 * span
    lo_padded = lo - pad
    hi_padded = hi + pad

    println("Clipping WAIC plot to [$(round(lo_padded,digits=1)), $(round(hi_padded,digits=1))] (with 5% padding)")

    # Apply and save clipped version
    Makie.ylims!(ax, lo_padded, hi_padded)
    out_pdf_clipped = replace(out_pdf, ".pdf" => "_clipped.pdf")
    save(out_pdf_clipped, fig)
    println("Saved clipped WAIC plot → $out_pdf_clipped")
    fig
end

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

    # --- IDENTIFY NEARLY IDENTICAL WAIC VALUES ---
    tol = 1e2  # tolerance for considering two WAICs equal (tune as needed)

    # Pair each seed index with its WAIC
    pairs = collect(zip(seed_indices, waic_nulls))

    # Sort by WAIC to make clusters easier to see
    sort!(pairs, by = x -> x[2])

    println("\nChecking for nearly identical WAICs (tolerance = $(tol)):")

    duplicate_groups = Vector{Vector{Int}}()
    current_group = [first(pairs)[1]]
    prev_waic = first(pairs)[2]

    for (seed, w) in pairs[2:end]
        if abs(w - prev_waic) < tol
            push!(current_group, seed)
        else
            if length(current_group) > 1
                push!(duplicate_groups, copy(current_group))
            end
            current_group = [seed]
            prev_waic = w
        end
    end
    if length(current_group) > 1
        push!(duplicate_groups, current_group)
    end

    if isempty(duplicate_groups)
        println("No near-identical WAIC groups found.")
    else
        println("Found $(length(duplicate_groups)) near-identical WAIC groups:")
        for (i, grp) in enumerate(duplicate_groups)
            wval = pairs[findfirst(x -> x[1] == grp[1], pairs)][2]
            println("  Group $i: seeds = $(grp)  → WAIC ≈ $(round(wval, digits=3))")
        end
    end

    # ────────────────────────────────────────────────
    # Compute pathology strength per region
    # ────────────────────────────────────────────────
    data_file = "data/total_path.csv"
    w_file = "data/W_labeled_filtered.csv"

    data, timepoints = process_pathology(data_file; W_csv=w_file)
    # data has dimensions: (subjects, timepoints, regions)
    println(size(data))

    # Compute max pathology per region (across all subjects/timepoints)
    max_path = [
        maximum(skipmissing(data[i, :, :])) for i in 1:size(data, 1)
    ]
    

    # Extract max_path for each seed
    seed_max_path = max_path[seed_indices]

    # ────────────────────────────────────────────────
    # Plot 2: ΔWAIC vs. max pathology in seeded region
    # ────────────────────────────────────────────────
    out_pdf3 = replace(out_pdf, "_WAIC_box.pdf" => "_seed_maxpath_vs_WAIC.pdf")
    fig3 = Figure(size=(700,500))
    ax3 = Axis(fig3[1,1];
        xlabel = "Max pathology in seeded region",
        ylabel = "ΔWAIC (seed − true)",
        titlesize = 28, xlabelsize = 24, ylabelsize = 24,
        yticklabelsize = 18, xticklabelsize = 18)
    scatter!(ax3, seed_max_path, delta_waic;
             color = RGBf(0.2, 0.2, 0.2), markersize = 12, alpha = 0.8)
    mkpath(dirname(out_pdf3))
    save(out_pdf3, fig3)
    println("Saved seed-maxpath vs ΔWAIC figure → $out_pdf3")

    # --- Optional: print correlation coefficient ---
    println("\nCorrelation between ΔWAIC and max pathology: ",
        round(cor(seed_max_path, delta_waic), digits=3))

end

# check what Seeds are lowest and highest
if mode == :seed
    # Pair seed IDs with WAIC values
    seed_ids = [parse(Int, match(r"seed_(\d+)$", splitext(basename(f))[1]).captures[1]) for f in sim_paths]

    # Sort by WAIC for easy inspection
    sorted_pairs = sort(collect(zip(seed_ids, waic_nulls)), by = x -> x[2])

    println("\n─── WAIC extremes ───")
    println("True WAIC:")
    println(true_waic)
    println("Lowest WAICs:")
    for (sid, w) in first(sorted_pairs, min(5, length(sorted_pairs)))
        println("  seed = $(sid), WAIC = $(round(w, digits=2))")
    end
    println("Highest WAICs:")
    for (sid, w) in last(sorted_pairs, min(5, length(sorted_pairs)))
        println("  seed = $(sid), WAIC = $(round(w, digits=2))")
    end
end