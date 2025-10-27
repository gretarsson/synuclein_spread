#!/usr/bin/env julia
# Compare posterior means for each local parameter family across two inferences.
# Colors indicate pathology presence:
#   red = pathology in both datasets
#   blue = pathology only in A
#   green = pathology only in B
#   gray = none in either

using PathoSpread
using MCMCChains
using Statistics, DataFrames, CSV, Printf
using StatsPlots, Distributions

# -------------------------------------------------------------
# SETTINGS
# -------------------------------------------------------------
inferenceA_path = "simulations/DIFFGAM_RETRO.jls"
inferenceB_path = "simulations/hippo_DIFFGAM_RETRO_posterior_prior_C2.jls"
out_prefix      = "results/injection_comparison/"
pathology_thresh = 0.05   # threshold for nonzero pathology

# -------------------------------------------------------------
# LOAD INFERENCES
# -------------------------------------------------------------
println("Loading inferences …")
infA = load_inference(inferenceA_path)
infB = load_inference(inferenceB_path)
chainA, chainB = infA["chain"], infB["chain"]
priorsA = infA["priors"]

# -------------------------------------------------------------
# Identify pathology presence per region in both datasets
# -------------------------------------------------------------
dataA = infA["data"]
dataB = infB["data"]
n_regions = size(dataA, 1)
@assert size(dataB, 1) == n_regions "Mismatch: dataA and dataB differ in number of regions"

# Helper function to test for any value > threshold (handle missings)
function region_has_pathology(data::Array{T,3}, thresh) where T
    n_regions = size(data, 1)
    flags = Vector{Bool}(undef, n_regions)
    for i in 1:n_regions
        vals = skipmissing(vec(data[i, :, :]))
        flags[i] = any(vals .> thresh)
    end
    return flags
end

has_pathology_A = region_has_pathology(dataA, pathology_thresh)
has_pathology_B = region_has_pathology(dataB, pathology_thresh)
has_pathology_both = has_pathology_A .& has_pathology_B

println("Regions with pathology (>$(pathology_thresh)):")
println("  A only: ", count(has_pathology_A .& .!has_pathology_B))
println("  B only: ", count(has_pathology_B .& .!has_pathology_A))
println("  Both:   ", count(has_pathology_both))
println("  None:   ", count(.!has_pathology_A .& .!has_pathology_B))

# -------------------------------------------------------------
# Determine local/global parameter families from prior names
# -------------------------------------------------------------
prior_names = collect(keys(priorsA))
families = unique(first.(split.(prior_names, "[")))  # e.g., ["beta", "gamma", "sigma"]
family_counts = Dict(f => count(name -> startswith(name, f*"["), prior_names) for f in families)
local_families = [f for (f, c) in family_counts if c == n_regions]
global_families = setdiff(families, local_families)

println("Identified local parameter families: ", local_families)
println("Global parameter families (density plots): ", global_families)

# -------------------------------------------------------------
# Helper to get all samples of a parameter, across all chains
# -------------------------------------------------------------
function get_param_samples(chain::Chains, i::Int)
    nd = ndims(chain.value)
    if nd == 2
        return vec(chain.value[:, i])
    elseif nd == 3
        return vec(chain.value[:, i, :])
    else
        error("Unexpected chain.value dimension: $nd")
    end
end

# -------------------------------------------------------------
# Loop through each local parameter family
# -------------------------------------------------------------
mkpath(out_prefix)

for fam in local_families
    println("\nAnalyzing family: $fam …")

    # Indices of this parameter family in the priors
    idxs = findall(name -> startswith(name, fam*"["), prior_names)

    # Compute posterior means in A and B
    μA = [mean(get_param_samples(chainA, i)) for i in idxs]
    μB = [mean(get_param_samples(chainB, i)) for i in idxs]
    reg_idx = [parse(Int, split(split(prior_names[i], "[")[2], "]")[1]) for i in idxs]

    # Determine pathology pattern for each region
    cat = Vector{String}(undef, length(reg_idx))
    for i in eachindex(reg_idx)
        if reg_idx[i] > n_regions
            cat[i] = "none"
            continue
        end
        a = has_pathology_A[reg_idx[i]]
        b = has_pathology_B[reg_idx[i]]
        if a && b
            cat[i] = "both"
        elseif a
            cat[i] = "A only"
        elseif b
            cat[i] = "B only"
        else
            cat[i] = "none"
        end
    end

    # Color mapping
    color_map = Dict(
        "A only" => :dodgerblue,
        "B only" => :green3,
        "both"   => :red,
        "none"   => :gray
    )
    colors = [color_map[c] for c in cat]

    # Plot
    plt = scatter(μA, μB;
        color = colors,
        xlabel = "Posterior mean (inference A)",
        ylabel = "Posterior mean (inference B)",
        title = "$fam parameters\n(red = both, blue = A only, green = B only)",
        legend = false,
        markersize = 6,
        linewidth = 2)
    plot!(identity, c=:black, ls=:dash)

    savefig("$(out_prefix)/$(fam)_scatter.pdf")

    # Save table
    df = DataFrame(index=reg_idx, μA=μA, μB=μB, diff=μB.-μA, category=cat)
    CSV.write("$(out_prefix)/$(fam)_summary.csv", df)
    println("  Saved → $(out_prefix)/$(fam)_scatter.pdf  and  $(fam)_summary.csv")
end

println("\nLocal parameter comparison done.")

# -------------------------------------------------------------
# Compare global parameters: posterior densities (A vs B)
# -------------------------------------------------------------
println("\n--- Comparing global parameters ---")

mkpath("$(out_prefix)/global_params")
pairs_priors = collect(priorsA)

for fam in global_families
    println("Analyzing global family: $fam …")

    # Indices of this family in priors
    idxs = findall(name -> startswith(name, fam*"["), prior_names)
    if isempty(idxs)
        idxs = findall(==("$(fam)"), prior_names)
    end

    for i in idxs
        name = prior_names[i]
        samplesA = get_param_samples(chainA, i)
        samplesB = get_param_samples(chainB, i)

        plt = density(samplesA; label="Inference A", linewidth=2, color=:dodgerblue)
        density!(samplesB; label="Inference B", linewidth=2, color=:red)
        xlabel!("Parameter value")
        ylabel!("Density")
        title!("Global parameter: $name")

        savefig("$(out_prefix)/global_params/$(name)_density.pdf")

        df = DataFrame(
            param = name,
            meanA = mean(samplesA),
            meanB = mean(samplesB),
            sdA   = std(samplesA),
            sdB   = std(samplesB)
        )
        CSV.write("$(out_prefix)/global_params/$(name)_summary.csv", df)

        println("  Saved → $(out_prefix)/global_params/$(name)_density.pdf")
    end
end

println("\nDone.")
