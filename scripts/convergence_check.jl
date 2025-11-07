#!/usr/bin/env julia
# ============================================================
# Convergence check: flag models with any parameter not converged
# ============================================================
using PathoSpread
using Glob
using Printf
using MCMCChains

indir = "simulations"
rhat_threshold = 1.1

# --- FIND ALL MERGED INFERENCE FILES ---
pattern = joinpath(indir, "*.jls")
paths = sort(glob(pattern))
merged_paths = filter(p -> !occursin(r"_C\d+\.jls", p), paths)

println("🔍 Found $(length(merged_paths)) merged inference files.\n")

nonconverged_models = String[]

for p in merged_paths
    simulation = splitext(basename(p))[1]
    println("📊 Checking convergence for $simulation ...")

    inference = load_inference(p)
    chains = inference["chain"]

    rhat_obj = MCMCChains.MCMCDiagnosticTools.rhat(chains)
    rhat_vals = rhat_obj.nt.rhat
    param_names = rhat_obj.nt.parameters

    # find parameters exceeding threshold
    bad_idx = findall(>(rhat_threshold), rhat_vals)

    if isempty(bad_idx)
        println(@sprintf("   ✅ All parameters converged (R̂ < %.2f)\n", rhat_threshold))
    else
        println(@sprintf("   ⚠️  %d parameters did not converge (R̂ ≥ %.2f):", length(bad_idx), rhat_threshold))
        for i in bad_idx
            println(@sprintf("      %s  →  R̂ = %.3f", param_names[i], rhat_vals[i]))
        end
        println()
        push!(nonconverged_models, simulation)
    end
end

println("🎉 Done checking convergence for all inferences!\n")

if isempty(nonconverged_models)
    println("✅ All models converged across all parameters.")
else
    println("⚠️  The following models had ≥1 parameter with R̂ ≥ $(rhat_threshold):")
    for name in nonconverged_models
        println("   - $name")
    end
end


