#!/usr/bin/env julia
using PathoSpread, Serialization, Statistics, Distributed, MCMCChains

# ───────────────────────────────────────────────────────────────
# SETTINGS
# ───────────────────────────────────────────────────────────────
mode = :shuffle
S = 300  # number of posterior samples per WAIC computation
addprocs(8)  # adjust to available cores

@everywhere using PathoSpread, Serialization, Statistics, MCMCChains

all_files = readdir("simulations"; join=true)

if mode == :shuffle
    sim_paths = filter(f -> occursin(r"DIFFGA_shuffle_\d+$", splitext(basename(f))[1]), all_files)
elseif mode == :seed
    sim_paths = filter(f -> occursin(r"DIFFGA_seed_\d+$", splitext(basename(f))[1]), all_files)
else
    error("Unknown mode: $mode")
end

println("Found $(length(sim_paths)) $(String(mode)) models.")

mkpath("results/waic_cache")

# ───────────────────────────────────────────────────────────────
# WAIC COMPUTATION + CACHING
# ───────────────────────────────────────────────────────────────
@everywhere function compute_and_cache_waic(sp::String; S=300)
    cache_file = replace(sp, "simulations/" => "results/waic_cache/")
    cache_file = replace(cache_file, ".jls" => "_waic.jls")

    # Skip if cached
    if isfile(cache_file)
        return deserialize(cache_file)
    end

    try
        inf = load_inference(sp)
        w, _, _, _, _, _ = compute_waic(inf; S=S)
        if isfinite(w)
            serialize(cache_file, w)
            return w
        else
            return missing
        end
    catch err
        @warn "Failed computing WAIC for $sp" err
        return missing
    end
end

# Parallel compute
waic_vals = pmap(sp -> compute_and_cache_waic(sp; S=S), sim_paths)
serialize("results/waic_cache/DIFFGA_$(String(mode))_waic_all.jls", waic_vals)

println("Saved $(count(!ismissing, waic_vals)) / $(length(waic_vals)) WAICs.")
# kill workers
rmprocs(workers())
