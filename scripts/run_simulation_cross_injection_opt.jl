#!/usr/bin/env julia
# ============================================================
# Cross-site seed amplitude optimization
# Fix all model parameters to hippocampal posterior,
# keep the same seed region(s) from striatum inference,
# and vary only the seed amplitude(s) to maximize R² vs striatum data.
# ============================================================

using PathoSpread
using Statistics, Printf, DataFrames, CSV, StatsPlots, MCMCChains, DifferentialEquations, OrderedCollections

# -------------------------------------------------------------
# SETTINGS
# -------------------------------------------------------------
inference_hippo_path    = "simulations/mean_hippo_DIFFG_RETRO_posterior_prior.jls"
inference_striatum_path = "simulations/mean_DIFFG_RETRO.jls"
out_prefix              = "results/injection_comparison_sims/mean_DIFFG_seedamp_BS"
save_figures            = true

# Seed amplitude sweep
seed_range = range(1.0, 1.0; length=1)
println("Seed amplitude search range: [$(first(seed_range)), $(last(seed_range))] ($(length(seed_range)) steps)")

# Flag
use_striatum_globals = false   # replace globals with striatum values

# -------------------------------------------------------------
# LOAD + NORMALIZE PARAM NAMES
# -------------------------------------------------------------
println("Loading inferences …")
inf_hippo    = load_inference(inference_hippo_path)
inf_striatum = load_inference(inference_striatum_path)

# filter last timepoints if needed
#inf_striatum["data"] = inf_striatum["data"][:,6:end,:]
#inf_striatum["timepoints"] = inf_striatum["timepoints"][6:end]

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
println("✓ Inferences loaded and parameter names aligned.")

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
# SEED REGIONS FROM STRIATUM (TARGET)
# -------------------------------------------------------------
seed_keys_str = filter(name -> occursin(r"^seed_", String(name)), keys(mode_striatum))
@assert !isempty(seed_keys_str) "No seed_ parameters found in striatum chain."

labels_striatum = inf_striatum["labels"]
n_regions = length(labels_striatum)
seed_indices = Int[]
for skey in seed_keys_str
    region_name = replace(String(skey), "seed_" => "")
    idx = findfirst(isequal(region_name), labels_striatum)
    @assert !isnothing(idx) "Region $region_name from $skey not found in labels."
    push!(seed_indices, idx)
end
println("→ Seeding same region(s) as striatum: ", labels_striatum[seed_indices])

# -------------------------------------------------------------
# BUILD PARAMETER DICTIONARY
# -------------------------------------------------------------
params_cross = copy(mode_hippo)
# drop seed params (ICs)
for s in filter(name -> occursin(r"^seed_", String(name)), keys(params_cross))
    delete!(params_cross, s)
end

# optionally replace global params
if use_striatum_globals
    global_keys = filter(name -> occursin(r"^(rho|alpha|theta|tau|λ|lambda|sigma|σ|noise|σy)$", String(name)), keys(params_cross))
    println("→ Replacing $(length(global_keys)) global parameters with striatum values:")
    for g in global_keys
        if haskey(mode_striatum, g)
            params_cross[g] = mode_striatum[g]
            println("   - $(g) ← striatum $(round(mode_striatum[g], digits=4))")
        end
    end
else
    println("→ Keeping all global parameters from hippocampus.")
end

# -------------------------------------------------------------
# BUILD PARAMETER VECTOR
# -------------------------------------------------------------
param_syms = collect(keys(mode_hippo))
sigma_idx  = findfirst(isequal(:σ), param_syms)
@assert !isnothing(sigma_idx) "No σ found — check parameter ordering."
dynamic_syms = param_syms[1:(sigma_idx-1)]
p_cross_vec  = [haskey(params_cross, s) ? params_cross[s] : mode_hippo[s] for s in dynamic_syms]

# -------------------------------------------------------------
# DATA + ODE SETUP (STRIATUM TARGET)
# -------------------------------------------------------------
Ltuple     = inf_striatum["L"]
N          = Ltuple[2]
factors    = inf_striatum["factors"]
tspan      = (0.0, inf_striatum["timepoints"][end])
save_times = inf_striatum["timepoints"]
observed   = Array(PathoSpread.mean3(inf_striatum["data"]))

# -------------------------------------------------------------
# METRIC
# -------------------------------------------------------------
function r2_score(y::AbstractArray, ŷ::AbstractArray)
    mask = .!ismissing.(y) .& .!ismissing.(ŷ)
    yv, ŷv = vec(y[mask]), vec(ŷ[mask])
    ss_res = sum((yv .- ŷv).^2)
    ss_tot = sum((yv .- mean(yv)).^2)
    return 1 - ss_res / ss_tot
end

# -------------------------------------------------------------
# SWEEP SEED AMPLITUDE
# -------------------------------------------------------------
results = DataFrame(amplitude = Float64[], R2 = Float64[])

println("Starting amplitude sweep …")
for amp in seed_range
    u0 = zeros(length(inf_striatum["u0"]))
    u0[seed_indices] .= amp

    prob = ODEProblem(
        (du, u, p, t) -> odes[inf_striatum["ode"]](du, u, p, t; L=Ltuple, factors=factors),
        u0, tspan
    )

    sol = solve(prob, Tsit5(); p=p_cross_vec, saveat=save_times)
    predicted = Array(sol[1:N, :])

    R2 = r2_score(observed, predicted)
    push!(results, (amp, R2))
    @printf("→ amp = %.3f  R² = %.4f\n", amp, R2)
end

best_idx = argmax(results.R2)
best_amp = results.amplitude[best_idx]
best_r2  = results.R2[best_idx]
println(@sprintf("\n✓ Optimal seed amplitude = %.3f  (R² = %.4f)", best_amp, best_r2))

# -------------------------------------------------------------
# SAVE RESULTS + PLOTS
# -------------------------------------------------------------
mkpath(out_prefix)
CSV.write("$(out_prefix)/seed_amplitude_search.csv", results)

if save_figures
    plt = plot(results.amplitude, results.R2;
        xlabel="Seed amplitude", ylabel="R²", lw=2,
        title=@sprintf("Seed amplitude optimization (best = %.2f, R² = %.3f)", best_amp, best_r2))
    scatter!([best_amp], [best_r2], c=:red, label="best")
    savefig("$(out_prefix)/seed_amplitude_search.pdf")
    println("Saved plot → $(out_prefix)/seed_amplitude_search.pdf")
end

# -------------------------------------------------------------
# PREDICTED VS OBSERVED FOR BEST AMPLITUDE
# -------------------------------------------------------------
println("Re-simulating at best amplitude to plot predicted vs observed …")
u0_best = zeros(length(inf_striatum["u0"]))
u0_best[seed_indices] .= best_amp

prob_best = ODEProblem(
    (du, u, p, t) -> odes[inf_striatum["ode"]](du, u, p, t; L=Ltuple, factors=factors),
    u0_best, tspan
)
sol_best = solve(prob_best, Tsit5(); p=p_cross_vec, saveat=save_times)
pred_best = Array(sol_best[1:N, :])

r2_check = r2_score(observed, pred_best)
@printf("Verification: re-simulated best amplitude R² = %.4f\n", r2_check)

if save_figures
    plt2 = scatter(vec(observed), vec(pred_best);
        xlabel="Observed (striatum data)",
        ylabel="Predicted (best amp = $(round(best_amp,digits=3)))",
        title=@sprintf("Predicted vs Observed (R² = %.3f)", r2_check),
        legend=false, alpha=0.5, markersize=4)
    plot!(identity, c=:gray, ls=:dash)
    savefig("$(out_prefix)/predicted_vs_observed_best_amp.pdf")
    println("Saved plot → $(out_prefix)/predicted_vs_observed_best_amp.pdf")
end

println("✓ Done.")
