#!/usr/bin/env julia
# Cross-site sensitivity: use globals from striatum inference with
# regionals from hippocampus inference, simulate ODE, and compute R² vs. hippocampus data.

using PathoSpread
using Statistics, Printf, DataFrames, CSV, StatsPlots, MCMCChains, DifferentialEquations

# -------------------------------------------------------------
# SETTINGS
# -------------------------------------------------------------
inference_striatum_path = "simulations/DIFFGA_RETRO.jls"       # globals from here
inference_hippo_path    = "simulations/hippo_DIFFGA_RETRO_posterior_prior.jls"  # regionals + data from here
out_prefix              = "results/injection_comparison/cross_site_global_striatum"
save_figures            = true

using MCMCChains

# -------------------------------------------------------------
# LOAD INFERENCES
# -------------------------------------------------------------
println("Loading inferences …")
inf_striatum = load_inference(inference_striatum_path)
inf_hippo    = load_inference(inference_hippo_path)

# -------------------------------------------------------------
# rename chain parameter names from p[n] → corresponding prior key
# -------------------------------------------------------------
function rename_chain_params!(inf)
    ch = inf["chain"]
    priorkeys = collect(keys(inf["priors"]))
    mapping = Dict{String,String}()

    # Build name mapping dictionary
    for oldsym in names(ch)
        s = String(oldsym)
        if occursin(r"^p\[\d+\]$", s)
            n = parse(Int, match(r"\d+", s).match)
            mapping[s] = priorkeys[n]
        end
    end

    # Only replace the names we have a mapping for
    if !isempty(mapping)
        inf["chain"] = MCMCChains.replacenames(ch, mapping)
    end

    return inf["chain"]
end

rename_chain_params!(inf_striatum)
rename_chain_params!(inf_hippo)

println("✓ Inferences loaded and parameter names aligned to priors.")


chain_striatum = inf_striatum["chain"]
chain_hippo    = inf_hippo["chain"]
priors_hippo   = inf_hippo["priors"]
data_hippo     = inf_hippo["data"]
tspan          = (0, inf_hippo["timepoints"][end])
println("✓ Inferences loaded and parameter names aligned.")


# -------------------------------------------------------------
# EXTRACT POSTERIOR MODES
# -------------------------------------------------------------
using OrderedCollections
function get_mode_params(chain::Chains)
    lp = chain[:lp]
    imax = argmax(lp)
    return OrderedDict(name => chain[name][imax] for name in names(chain))
end

mode_striatum = get_mode_params(chain_striatum)
mode_hippo    = get_mode_params(chain_hippo)

# -------------------------------------------------------------
# SPLIT PARAMETERS INTO GLOBAL / REGIONAL
# -------------------------------------------------------------
# Identify global parameter names (e.g., rho, alpha, sigma, tau)
allnames = collect(keys(mode_hippo))
global_keys = filter(name -> occursin(r"^(rho|alpha|theta|σ|tau|lambda|λ|noise|σy)$", String(name)), allnames)
local_keys  = filter(name -> occursin(r"\[", String(name)), allnames)


println("Global parameters detected: ", global_keys[1:min(end,10)])
println("Regional parameters detected: ", first(local_keys, 5), " …")

# Construct cross-site parameter dictionary
params_cross = copy(mode_hippo)
for g in global_keys
    if haskey(mode_striatum, g)
        params_cross[g] = mode_striatum[g]
    end
end

# -------------------------------------------------------------
# BUILD INITIAL CONDITION (u0) FROM POSTERIOR MODES
# -------------------------------------------------------------
println("Building initial condition (u0) from seed parameters …")

seed_keys = filter(name -> occursin(r"^seed_", String(name)), keys(mode_hippo))
@assert !isempty(seed_keys) "No seed_ parameters found in chain — cannot construct u0."

labels = inf_hippo["labels"]
n_regions = length(inf_hippo["u0"])
u0_cross = zeros(n_regions)

for skey in seed_keys
    s = String(skey)
    region_name = replace(s, "seed_" => "")
    idx = findfirst(isequal(region_name), labels)
    @assert !isnothing(idx) "Region $region_name from $skey not found in labels."
    u0_cross[idx] = mode_hippo[skey]
end

println("→ Constructed u0 with $(count(!=(0.0), u0_cross)) nonzero seed regions.")


# -------------------------------------------------------------
# CONVERT PARAMETER DICTIONARY TO VECTOR (ORDERED LIKE PRIORS)
# -------------------------------------------------------------
p_cross_vec = collect(values(params_cross))


# -------------------------------------------------------------
# SIMULATE SYSTEM
# -------------------------------------------------------------
println("Building and simulating ODE with mixed parameters …")
prob = PathoSpread.make_ode_problem(
    odes["DIFFGA"],                # ODE function
    labels = inf_hippo["labels"],       # region labels
    Ltuple = inf_hippo["L"],       # Laplacian or connectivity matrices
    factors = inf_hippo["factors"],     # scaling factors
    u0 = u0_cross,               # initial state, posterior mode seeds from hippocampus
    timepoints = inf_hippo["timepoints"]
)
@show prob
@show odes["DIFFGA"]
@show inf_hippo["timepoints"]
# Standard DifferentialEquations solve
sol = solve(prob, Tsit5(); p=params_cross, saveat=inf_hippo["timepoints"])

# Align simulated output with observed data
predicted = Array(sol)
observed  = Array(data_hippo)
@assert size(predicted) == size(observed) "Shape mismatch between simulated and observed data."

# -------------------------------------------------------------
# COMPUTE R²
# -------------------------------------------------------------
r2_val = r2_score(vec(observed), vec(predicted))
println(@sprintf("Cross-site R² (globals from striatum, regionals from hippocampus): %.4f", r2_val))

# -------------------------------------------------------------
# PLOT
# -------------------------------------------------------------
if save_figures
    mkpath(out_prefix)
    plt = scatter(vec(observed), vec(predicted);
        xlabel = "Observed (hippocampus data)",
        ylabel = "Predicted (using striatum globals)",
        title  = @sprintf("Cross-site prediction (R² = %.3f)", r2_val),
        legend = false,
        alpha  = 0.5,
        markersize = 4)
    plot!(identity, c=:gray, ls=:dash)
    savefig("$(out_prefix)/predicted_vs_observed_cross_site.pdf")
    println("Saved plot → $(out_prefix)/predicted_vs_observed_cross_site.pdf")
end

# Save summary
CSV.write("$(out_prefix)/cross_site_r2.csv",
          DataFrame(R2 = [r2_val]))

println("✓ Done.")
