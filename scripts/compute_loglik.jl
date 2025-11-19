#!/usr/bin/env julia

using Distributed
addprocs()          # or addprocs(4)

@everywhere using Serialization
@everywhere using PathoSpread
@everywhere using OrderedCollections

using ProgressMeter   # progress bar only on the main process

# ============================================================
# Input files
# ============================================================

const inference_files = Dict(
    #"DIFF_RETRO"      => "simulations/DIFF_RETRO.jls",
    #"DIFF_ANTERO"     => "simulations/DIFF_ANTERO.jls",
    #"DIFF_BIDIR"      => "simulations/DIFF_BIDIR.jls",
    #"DIFF_EUCL"       => "simulations/DIFF_EUCL.jls",

    #"DIFFG_RETRO"     => "simulations/DIFFG_RETRO.jls",
    #"DIFFG_ANTERO"    => "simulations/DIFFG_ANTERO.jls",
    #"DIFFG_BIDIR"     => "simulations/DIFFG_BIDIR.jls",
    #"DIFFG_EUCL"      => "simulations/DIFFG_EUCL.jls",

    #"DIFFGA_RETRO"    => "simulations/DIFFGA_RETRO.jls",
    #"DIFFGA_ANTERO"   => "simulations/DIFFGA_ANTERO_CUT.jls",
    #"DIFFGA_BIDIR"    => "simulations/DIFFGA_BIDIR.jls",
    #"DIFFGA_EUCL"     => "simulations/DIFFGA_EUCL.jls",
    "DIFFGA_EUCL"     => "simulations/DIFFGA_EUCL_CUT.jls",
)

println("==============================================")
println("Parallel log-likelihood computation")
println("Workers: ", nworkers())
println("==============================================")


# ============================================================
# Worker task — no progress bar here
# ============================================================

@everywhere function compute_and_save_loglik_worker(model_key::String, path::String)
    inf = PathoSpread.load_inference(path)

    # Skip if already computed
    #if haskey(inf, "loglik_mat") && haskey(inf, "loglik_rhat")
    #    return (model_key, :skipped)
    #end

    # Compute full loglik
    loglik_mat, loglik_rhat = PathoSpread.loglik(inf)

    # Add keys ONLY
    inf["loglik_mat"]  = loglik_mat
    inf["loglik_rhat"] = loglik_rhat

    # Save back to same file
    open(path, "w") do io
        serialize(io, inf)
    end

    return (model_key, :done)
end


# ============================================================
# Run in parallel with a single progress bar
# ============================================================

n_files = length(inference_files)
prog = Progress(n_files; desc="Processing inference files", dt=0.5)

# Launch tasks
futures = [
    @spawn compute_and_save_loglik_worker(model_key, path)
    for (model_key, path) in inference_files
]

# Wait for each and update progress bar
for fut in futures
    fetch(fut)   # returns (model_key, status)
    next!(prog)
end

println("\n==============================================")
println("All log-likelihood computations finished.")
println("==============================================")
