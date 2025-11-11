#!/usr/bin/env julia
# ============================================================
# Cross-site sensitivity (reverse direction)
# Use all model parameters from hippocampal inference
# but use initial conditions (seeds) from striatal inference,
# and evaluate fit (R²) against striatal data.
# ============================================================

using PathoSpread
using Statistics, Printf, DataFrames, CSV, StatsPlots, MCMCChains, DifferentialEquations, OrderedCollections

# -------------------------------------------------------------
# SETTINGS
# -------------------------------------------------------------
inference_hippo_path    = "simulations/hippo_DIFFG_RETRO_posterior_prior.jls"  # parameters from here
inference_striatum_path = "simulations/DIFFG_RETRO.jls"                        # seeds + data from here
out_prefix              = "results/injection_comparison_sims/DIFFG"
save_figures            = true

# -------------------------------------------------------------
# LOAD INFERENCES
# -------------------------------------------------------------
println("Loading inferences …")
inf_hippo    = load_inference(inference_hippo_path)
inf_striatum = load_inference(inference_striatum_path)

# -------------------------------------------------------------
# Rename chain parameters (p[n] → prior key names)
# -------------------------------------------------------------
function rename_chain_params!(inf)
    ch = inf["chain"]
    priorkeys = collect(keys(inf["priors"]))
    mapping = Dict{String,String}()

    for oldsym in names(ch)
        s = String(oldsym)
        if occursin(r"^p\[\d+\]$", s)
            n = parse(Int, match(r"\d+", s).match)
            mapping[s] = priorkeys[n]
        end
    end

    if !isempty(mapping)
        inf["chain"] = MCMCChains.replacenames(ch, mapping)
    end

    return inf["chain"]
end

rename_chain_params!(inf_hippo)
rename_chain_params!(inf_striatum)
println("✓ Inferences loaded and parameter names aligned to priors.")

# -------------------------------------------------------------
# EXTRACT POSTERIOR MODES
# -------------------------------------------------------------
function get_mode_params(chain::Chains)
    lp = chain[:lp]
    imax = argmax(lp)
    param_names = names(chain, :parameters)
    return OrderedDict(name => chain[name][imax] for name in param_names)
end

mode_hippo    = get_mode_params(inf_hippo["chain"])
mode_striatum = get_mode_params(inf_striatum["chain"])

# -------------------------------------------------------------
# BUILD PARAMETER DICTIONARY: all hippocampal except seeds
# -------------------------------------------------------------
params_cross = copy(mode_hippo)

seed_keys = filter(name -> occursin(r"^seed_", String(name)), keys(mode_hippo))
for s in seed_keys
    delete!(params_cross, s)  # remove hippocampal seeds
end
println("→ Using hippocampal parameters except seeds (removed $(length(seed_keys))).")

# -------------------------------------------------------------
# BUILD INITIAL CONDITION (u0) FROM STRIATAL SEED PARAMETERS
# -------------------------------------------------------------
println("Building initial condition (u0) from striatum seeds …")
seed_keys_str = filter(name -> occursin(r"^seed_", String(name)), keys(mode_striatum))
@assert !isempty(seed_keys_str) "No seed_ parameters found in striatum chain."

labels_striatum = inf_striatum["labels"]
n_regions = length(inf_striatum["u0"])
u0_cross = zeros(n_regions)

for skey in seed_keys_str
    region_name = replace(String(skey), "seed_" => "")
    idx = findfirst(isequal(region_name), labels_striatum)
    @assert !isnothing(idx) "Region $region_name from $skey not found in striatum labels."
    u0_cross[idx] = mode_striatum[skey]
end

println("→ Constructed u0 with $(count(!=(0.0), u0_cross)) nonzero seed regions.")

# -------------------------------------------------------------
# CONVERT PARAMETER DICTIONARY TO VECTOR (ORDERED LIKE PRIORS)
# -------------------------------------------------------------
param_syms = collect(keys(mode_hippo))
sigma_idx  = findfirst(isequal(:σ), param_syms)
@assert !isnothing(sigma_idx) "No σ found — check parameter ordering."

dynamic_syms = param_syms[1:(sigma_idx-1)]
p_cross_vec  = [haskey(params_cross, s) ? params_cross[s] : mode_hippo[s] for s in dynamic_syms]

println("→ Using $(length(p_cross_vec)) dynamic parameters from hippocampal posterior.")

# -------------------------------------------------------------
# SIMULATE SYSTEM ON STRIATUM DATA
# -------------------------------------------------------------
println("Simulating ODE using hippocampal parameters + striatal seeds …")

Ltuple     = inf_striatum["L"]
N          = Ltuple[2]
factors    = inf_striatum["factors"]
tspan      = (0.0, inf_striatum["timepoints"][end])
save_times = inf_striatum["timepoints"]

prob = ODEProblem(
    (du, u, p, t) -> odes[inf_striatum["ode"]](du, u, p, t; L=Ltuple, factors=factors),
    u0_cross, tspan
)

sol = solve(prob, Tsit5(); p=p_cross_vec, saveat=save_times)

predicted = Array(sol[1:N, :])
observed  = Array(PathoSpread.mean3(inf_striatum["data"]))
@assert size(predicted) == size(observed) "Shape mismatch between simulated and observed data."

# -------------------------------------------------------------
# COMPUTE R²
# -------------------------------------------------------------
r2_val = PathoSpread.r2_score(vec(predicted), vec(observed))
println(@sprintf(
    "Cross-site R² (hippocampal params + striatal seeds): %.4f",
    r2_val
))

# -------------------------------------------------------------
# PLOT AND SAVE RESULTS
# -------------------------------------------------------------
if save_figures
    mkpath(out_prefix)
    plt = scatter(vec(observed), vec(predicted);
        xlabel = "Observed (striatum data)",
        ylabel = "Predicted (hippocampal params + striatum seeds)",
        title  = @sprintf("Cross-site prediction (R² = %.3f)", r2_val),
        legend = false,
        alpha  = 0.5,
        markersize = 4)
    plot!(identity, c=:gray, ls=:dash)
    savefig("$(out_prefix)/predicted_vs_observed_cross_site.pdf")
    println("Saved plot → $(out_prefix)/predicted_vs_observed_cross_site.pdf")
end

CSV.write("$(out_prefix)/cross_site_r2.csv", DataFrame(R2 = [r2_val]))
println("✓ Done.")
