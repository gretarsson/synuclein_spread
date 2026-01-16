#!/usr/bin/env julia

using PathoSpread
using CSV, DataFrames
using Statistics
using Distributions
using Plots

###############################################################
# SETTINGS
###############################################################

simulation = "DIFFGA_RETRO"

zero_threshold = 0.01     # threshold for "nonzero pathology"
UPDATE_Z_THR  = 0.5*0.5       # posterior mean must shift > 0.5 prior SD
UPDATE_KL_THR = 0.5*0.1       # KL divergence threshold

outdir = "results/update_check"
isdir(outdir) || mkpath(outdir)

println("=== Checking parameter updates for simulation: $simulation ===")

###############################################################
# LOAD INFERENCE AND BASIC OBJECTS
###############################################################

inference = load_inference("simulations/" * simulation * ".jls")

chain          = inference["chain"]
priors         = inference["priors"]
model_regions  = inference["labels"]
param_names    = collect(keys(priors))        # strings: "beta[1]", "gamma[1]", etc.
N_regions      = length(model_regions)

# Identify β and γ indices in chain
beta_ind  = findall(k -> startswith(k, "beta["),  param_names)
gamma_ind = findall(k -> startswith(k, "gamma["), param_names)

Nb = length(beta_ind)
Ng = length(gamma_ind)

println("Found $Nb beta parameters and $Ng gamma parameters.")

###############################################################
# IDENTIFY NONZERO-PATHOLOGY REGIONS
###############################################################

data = inference["data"]        # dims: region × time × sample
nonzero_nodes = nonzero_regions(data, eps = zero_threshold)

println("Nonzero pathology regions = $(length(nonzero_nodes)) / $N_regions")

###############################################################
# HELPER FUNCTION: extract posterior draws using parameter INDEX
###############################################################

function get_post(chain, param_index::Int)
    # Iterations × Chains array of samples for this parameter
    arr = Array(chain[:, param_index, :])
    return vec(arr)   # flatten to 1D vector: all iterations × chains
end

###############################################################
# KL divergence for two Normals (prior vs. posterior approx)
###############################################################

function KL_normal(μp, σp, μq, σq)
    return 0.5 * (
        (σq/σp)^2 +
        ((μq - μp)/σp)^2 -
        1 -
        log((σq/σp)^2)
    )
end

###############################################################
# STORAGE ARRAYS
###############################################################

prior_mean_beta  = zeros(Float64, Nb)
prior_sd_beta    = zeros(Float64, Nb)
post_mean_beta   = zeros(Float64, Nb)
post_sd_beta     = zeros(Float64, Nb)
z_beta           = zeros(Float64, Nb)
kl_beta          = zeros(Float64, Nb)

prior_mean_gamma = zeros(Float64, Ng)
prior_sd_gamma   = zeros(Float64, Ng)
post_mean_gamma  = zeros(Float64, Ng)
post_sd_gamma    = zeros(Float64, Ng)
z_gamma          = zeros(Float64, Ng)
kl_gamma         = zeros(Float64, Ng)

###############################################################
# COMPUTE PRIOR–POSTERIOR DISTANCES FOR β
###############################################################

println("\n=== Computing update metrics for beta ===")

for j in 1:Nb
    pname = param_names[beta_ind[j]]    # e.g. "beta[17]"
    prior_dist = priors[pname]          # Distribution object

    prior_mean_beta[j] = mean(prior_dist)
    prior_sd_beta[j]   = std(prior_dist)

    post_vals = get_post(chain, beta_ind[j])
    post_mean_beta[j] = mean(post_vals)
    post_sd_beta[j]   = std(post_vals)

    # shift in posterior mean relative to prior SD
    z_beta[j]  = abs(post_mean_beta[j] - prior_mean_beta[j]) / prior_sd_beta[j]

    # KL divergence (Normal approximation)
    kl_beta[j] = KL_normal(prior_mean_beta[j], prior_sd_beta[j],
                           post_mean_beta[j], post_sd_beta[j])
end

###############################################################
# COMPUTE PRIOR–POSTERIOR DISTANCES FOR γ
###############################################################

println("\n=== Computing update metrics for gamma ===")

for j in 1:Ng
    pname = param_names[gamma_ind[j]]
    prior_dist = priors[pname]

    prior_mean_gamma[j] = mean(prior_dist)
    prior_sd_gamma[j]   = std(prior_dist)

    post_vals = get_post(chain, gamma_ind[j])
    post_mean_gamma[j] = mean(post_vals)
    post_sd_gamma[j]   = std(post_vals)

    z_gamma[j]  = abs(post_mean_gamma[j] - prior_mean_gamma[j]) / prior_sd_gamma[j]
    kl_gamma[j] = KL_normal(prior_mean_gamma[j], prior_sd_gamma[j],
                            post_mean_gamma[j], post_sd_gamma[j])
end

###############################################################
# DETERMINE UPDATED PARAMETERS
###############################################################

updated_beta  = (z_beta .> UPDATE_Z_THR) .| (kl_beta .> UPDATE_KL_THR)
updated_gamma = (z_gamma .> UPDATE_Z_THR) .| (kl_gamma .> UPDATE_KL_THR)

println("\n=== UPDATE SUMMARY ===")
println("Beta updated    = $(count(updated_beta)) out of $Nb")
println("Gamma updated   = $(count(updated_gamma)) out of $Ng")

# Overlap with nonzero pathology
overlap_beta  = intersect(nonzero_nodes, findall(updated_beta))
overlap_gamma = intersect(nonzero_nodes, findall(updated_gamma))

println("\n=== Overlap with nonzero pathology regions ===")

N_nonzero = length(nonzero_nodes)
N_beta    = length(beta_mean)
N_gamma   = length(gamma_mean)
N_up_beta = count(updated_beta)
N_up_gamma = count(updated_gamma)

println("Total model regions                   : $(length(model_regions))")
println("Regions with nonzero pathology        : $N_nonzero")
println("Updated β regions                     : $N_up_beta")
println("Updated γ regions                     : $N_up_gamma")

println("\nbeta ∩ nonzero regions  : $(length(overlap_beta))  " *
        "($(round(length(overlap_beta)/N_up_beta*100; digits=1))% of updated beta)")
println("gamma ∩ nonzero regions : $(length(overlap_gamma)) " *
        "($(round(length(overlap_gamma)/N_up_gamma*100; digits=1))% of updated gamma)")
println("")


###############################################################
# VISUALIZATION
###############################################################

# zero_gamma[i] = true if region i had max pathology < threshold
zero_gamma = trues(length(model_regions))
zero_gamma[nonzero_nodes] .= false    # nonzero_nodes = indices with pathology > threshold

scatter(z_gamma,
    xlabel="Gamma parameter index",
    ylabel="Posterior–prior mean shift (z-score)",
    title="Gamma Prior→Posterior Shift",
    c=Int.(updated_gamma),
    markersize=6,
    colorbar=true)
savefig("$outdir/gamma_z_shift_zeroregion.png")

scatter(kl_gamma,
    xlabel="Gamma parameter index",
    ylabel="KL divergence",
    title="Gamma KL Divergence Prior→Posterior",
    c=Int.(updated_gamma),
    markersize=6,
    colorbar=true)
savefig("$outdir/gamma_kl_zeroregion.png")

println("\nSaved gamma update plots to $outdir/")
    
    

scatter(z_gamma,
    xlabel="Gamma parameter index",
    ylabel="Posterior–prior mean shift (z-score)",
    title="Gamma Prior→Posterior Shift",
    c=Int.(zero_gamma),
    markersize=6,
    colorbar=true)
savefig("$outdir/gamma_z_shift_zeroregion.png")

scatter(kl_gamma,
    xlabel="Gamma parameter index",
    ylabel="KL divergence",
    title="Gamma KL Divergence Prior→Posterior",
    c=Int.(zero_gamma),
    markersize=6,
    colorbar=true)
savefig("$outdir/gamma_kl_zeroregion.png")



###############################################################
# SAVE TABLE
###############################################################

#df = DataFrame(
#    region = model_regions,
#    nonzero = [i in nonzero_nodes for i in 1:N_regions],
#    beta_updated = updated_beta,
#    gamma_updated = updated_gamma,
#    z_beta = z_beta,
#    z_gamma = z_gamma,
#    kl_beta = kl_beta,
#    kl_gamma = kl_gamma
#)
#
#CSV.write("$outdir/parameter_update_table.csv", df)
#println("Wrote update table to $outdir/parameter_update_table.csv")
#
#println("\n=== Done ===\n")
