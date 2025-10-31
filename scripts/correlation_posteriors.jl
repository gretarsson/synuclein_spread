#!/usr/bin/env julia
# Compute correlation between global parameters rho and alpha
# (handles multiple chains, plots posterior scatter)

using PathoSpread
using MCMCChains
using Statistics, StatsPlots, Printf

# -------------------------------------------------------------
# SETTINGS
# -------------------------------------------------------------
inference_path  = "simulations/hippo_DIFFGAM_RETRO.jls"   # change if needed
out_prefix      = "results/injection_comparison"
param_rho_name  = Symbol("p[1]")
param_alpha_name = Symbol("p[2]")

# -------------------------------------------------------------
# LOAD INFERENCE
# -------------------------------------------------------------
println("Loading inference …")
inf = load_inference(inference_path)
chain = inf["chain"]

# -------------------------------------------------------------
# Helper: flatten all samples across chains for a given parameter name
# -------------------------------------------------------------
function get_param_samples(chain::Chains, name::Symbol)
    nms = names(chain)
    idxs = findall(x -> x == name, nms)
    isempty(idxs) && error("Parameter '$name' not found. Available: $(nms)")
    i = first(idxs)
    v = chain.value
    nd = ndims(v)
    if nd == 2
        return vec(v[:, i])
    elseif nd == 3
        return vec(v[:, i, :])
    else
        error("Unexpected chain.value dimension: $nd")
    end
end

# -------------------------------------------------------------
# EXTRACT SAMPLES
# -------------------------------------------------------------
rho_samples   = get_param_samples(chain, param_rho_name)
alpha_samples = get_param_samples(chain, param_alpha_name)

println("Number of samples per parameter: ", length(rho_samples))
println("rho mean = ", @sprintf("%.3f", mean(rho_samples)), 
        ", alpha mean = ", @sprintf("%.3f", mean(alpha_samples)))

# -------------------------------------------------------------
# CORRELATION
# -------------------------------------------------------------
corr_val = cor(rho_samples, alpha_samples)
println(@sprintf("Correlation(rho, alpha) = %.3f", corr_val))

# -------------------------------------------------------------
# PLOT
# -------------------------------------------------------------
mkpath(out_prefix)
plt = scatter(rho_samples, alpha_samples;
    alpha=0.3,
    markersize=3,
    xlabel="rho",
    ylabel="alpha",
    title="Posterior correlation (rho vs alpha): r = $(round(corr_val, digits=3))",
    legend=false)
savefig("$(out_prefix)/rho_alpha_correlation.pdf")
println("Saved plot → $(out_prefix)/rho_alpha_correlation.pdf")

println("Done.")
