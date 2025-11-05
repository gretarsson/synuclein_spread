#!/usr/bin/env julia
# ============================================================
# Batch plotting of inference results (handles T{n} + hippo)
# ============================================================
using PathoSpread
using Glob
using Printf

# --- SETTINGS ---
indir  = "simulations"
outdir = "figures"

# --- FIND ALL MERGED INFERENCE FILES ---
pattern = joinpath(indir, "*.jls")
paths = sort(glob(pattern))

# Exclude chain parts (_C1.jls, _C2.jls, etc.)
merged_paths = filter(p -> !occursin(r"_C\d+\.jls", p), paths)

if isempty(merged_paths)
    error("No merged inference files found in $indir")
end

foreach(println, merged_paths)
println("🎨 Found $(length(merged_paths)) merged inference files:")

# --- RUN PLOT FOR EACH ---
setup_plot_theme!()  # set global plotting settings

for p in merged_paths
    simulation = splitext(basename(p))[1]
    outprefix  = joinpath(outdir, simulation)
    println("\n▶️  Plotting results for $simulation ...")

    # --- Load inference ---
    inference_obj = load_inference(p)

    # --- Select dataset based on prefix ---
    if startswith(simulation, "hippo_")
        data_file = "data/hippocampal/hippocampal_syn_only.csv"
    else
        data_file = "data/total_path.csv"
    end

    # --- Handle T{n} cases (holdout / retrodiction) ---
    if occursin(r"T\d+$", simulation)
        println("   Detected holdout simulation → including full data + retrodiction plot")

        # Load full dataset for in-sample/out-of-sample comparison
        data_full, timepoints_full = PathoSpread.process_pathology(
            data_file; W_csv="data/W_labeled_filtered.csv"
        )

        # Standard inference plot with full data overlay
        plot_inference(
            inference_obj,
            outprefix;
            full_data=data_full,
            full_timepoints=timepoints_full
        )

    else
        # --- Normal case: just plot posterior fits ---
        plot_inference(inference_obj, outprefix)
    end

    @printf("✅ Saved plots → %s*\n", outprefix)
end

println("\n🎉 Done generating plots for all inferences!")
