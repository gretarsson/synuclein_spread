#!/usr/bin/env julia
# ============================================================
# Cross-site seed amplitude optimization
# Fix all model parameters to STRIATUM posterior,
# update only regions with pathology in hippocampus but not in striatum,
# use hippocampal seed(s) (same amplitude each),
# and evaluate on hippocampal data.
# ============================================================
using PathoSpread
using Statistics, Printf, DataFrames, CSV, StatsPlots, MCMCChains, DifferentialEquations, OrderedCollections

# -------------------------------------------------------------
# SETTINGS
# -------------------------------------------------------------
inference_striatum_path = "simulations/DIFFGAM_RETRO.jls"               # trained
inference_hippo_path    = "simulations/hippo_DIFFGAM_RETRO_posterior_prior_CUT.jls"  # target
out_prefix              = "results/injection_comparison_sims/mean_DIFFG_cross_striatum_to_hippocampus"
save_figures            = true

# Seed amplitude sweep (all hippocampal seeds get the same amplitude)
seed_range = range(0.7, 0.7; length=1)
println("Seed amplitude search range: [$(first(seed_range)), $(last(seed_range))] ($(length(seed_range)) steps)")

# Pathology threshold
pathology_thresh = 0.05

# Flag
use_striatum_globals = true   # keep striatum globals; we're testing its generalization

# -------------------------------------------------------------
# LOAD + NORMALIZE PARAM NAMES
# -------------------------------------------------------------
println("Loading inferences …")
inf_striatum = load_inference(inference_striatum_path)
inf_hippo    = load_inference(inference_hippo_path)

function rename_chain_params!(inf)
    ch = inf["chain"]
    priorkeys = collect(keys(inf["priors"]))
    mapping = Dict{String,String}()
    for oldsym in names(ch)
        s = String(oldsym)
        if occursin(r"^p\\[\\d+\\]$", s)
            n = parse(Int, match(r"\\d+", s).match)
            mapping[s] = priorkeys[n]
        end
    end
    if !isempty(mapping)
        inf["chain"] = MCMCChains.replacenames(ch, mapping)
    end
    return inf["chain"]
end

rename_chain_params!(inf_striatum)
rename_chain_params!(inf_hippo)
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

mode_striatum = get_mode_params(inf_striatum["chain"])
mode_hippo    = get_mode_params(inf_hippo["chain"])

# -------------------------------------------------------------
# PATHOLOGY PRESENCE
# -------------------------------------------------------------
function region_has_pathology(data::Array{T,3}, thresh) where T
    n_regions = size(data, 1)
    flags = Vector{Bool}(undef, n_regions)
    for i in 1:n_regions
        vals = skipmissing(vec(data[i, :, :]))
        flags[i] = any(vals .> thresh)
    end
    return flags
end

has_path_str = region_has_pathology(inf_striatum["data"], pathology_thresh)
has_path_hip = region_has_pathology(inf_hippo["data"], pathology_thresh)

# Identify regions to update: no pathology in striatum but pathology in hippocampus
regions_to_update = findall((!has_path_str[i]) && has_path_hip[i] for i in 1:length(has_path_hip))
println("→ Updating $(length(regions_to_update)) regions with hippocampal parameters.")

# -------------------------------------------------------------
# SEED REGIONS FROM HIPPOCAMPUS
# -------------------------------------------------------------
seed_keys_hip = filter(name -> occursin(r"^seed_", String(name)), keys(mode_hippo))
@assert !isempty(seed_keys_hip) "No seed_ parameters found in hippocampal chain."

labels_hippo = inf_hippo["labels"]
n_regions = length(labels_hippo)
seed_indices = Int[]
for skey in seed_keys_hip
    region_name = replace(String(skey), "seed_" => "")
    idx = findfirst(isequal(region_name), labels_hippo)
    @assert !isnothing(idx) "Region $region_name from $skey not found in labels."
    push!(seed_indices, idx)
end
println("→ Using hippocampal seed region(s): ", labels_hippo[seed_indices])
seed_indices = inf_hippo["seed_idx"]

# -------------------------------------------------------------
# BUILD PARAMETER DICTIONARY
# -------------------------------------------------------------
params_cross = copy(mode_striatum)  # base: striatum params

# drop seed params
for s in filter(name -> occursin(r"^seed_", String(name)), keys(params_cross))
    delete!(params_cross, s)
end

# update selected local parameters from hippocampus
# --- Helper: extract region index from symbols like :y0[3], :ydelta[17], etc. ---
function region_index_from_sym(sym::Symbol)
    s = String(sym)
    m = match(r"\[(\d+)\]", s)  # ✅ only one backslash needed inside a raw string
    isnothing(m) && return nothing
    return parse(Int, m.captures[1])
end

# --- Update selected region parameters from hippocampus ---
n_updates = 0
for (k, v) in mode_hippo
    ridx = region_index_from_sym(k)
    if !isnothing(ridx) && ridx in regions_to_update
        if haskey(params_cross, k)
            old_val = params_cross[k]
            params_cross[k] = v
            global n_updates += 1
            @printf("Updated %s (region %d): %.4f → %.4f\n", String(k), ridx, old_val, v)
        end
    end
end
println("✓ Updated $n_updates regional parameters from hippocampus.")


# optionally replace global params (usually we keep striatum)
if !use_striatum_globals
    global_keys = filter(name -> occursin(r"^(rho|alpha|theta|tau|λ|lambda|sigma|σ|noise|σy)$", String(name)), keys(mode_hippo))
    for g in global_keys
        if haskey(mode_hippo, g)
            params_cross[g] = mode_hippo[g]
        end
    end
    println("✓ Global parameters replaced from hippocampus.")
else
    println("→ Keeping all global parameters from striatum.")
end

# -------------------------------------------------------------
# BUILD PARAMETER VECTOR (matching hippocampal ordering)
# -------------------------------------------------------------
param_syms = collect(keys(mode_hippo))
sigma_idx  = findfirst(isequal(:σ), param_syms)
@assert !isnothing(sigma_idx) "No σ found — check parameter ordering."
dynamic_syms = param_syms[1:(sigma_idx-1)]
p_cross_vec  = [haskey(params_cross, s) ? params_cross[s] : mode_striatum[s] for s in dynamic_syms]
p_cross_vec[end] = 0

# -------------------------------------------------------------
# DATA + ODE SETUP (HIPPOCAMPUS TARGET)
# -------------------------------------------------------------
Ltuple     = inf_hippo["L"]
N          = Ltuple[2]
factors    = inf_hippo["factors"]
tspan      = (0.0, inf_hippo["timepoints"][end])
save_times = inf_hippo["timepoints"]
observed   = Array(PathoSpread.mean3(inf_hippo["data"]))

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
# SWEEP SEED AMPLITUDE (shared across seeds)
# -------------------------------------------------------------
results = DataFrame(amplitude = Float64[], R2 = Float64[])
println("Starting amplitude sweep …")

for amp in seed_range
    u0 = zeros(length(inf_hippo["u0"]))
    u0[seed_indices] .= amp  # equal amplitude for all hippocampal seeds

    prob = ODEProblem(
        (du, u, p, t) -> odes[inf_hippo["ode"]](du, u, p, t; L=Ltuple, factors=factors),
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
u0_best = zeros(length(inf_hippo["u0"]))
u0_best[seed_indices] .= best_amp

prob_best = ODEProblem(
    (du, u, p, t) -> odes[inf_hippo["ode"]](du, u, p, t; L=Ltuple, factors=factors),
    u0_best, tspan
)
sol_best = solve(prob_best, Tsit5(); p=p_cross_vec, saveat=save_times)
pred_best = Array(sol_best[1:N, :])
r2_check = r2_score(observed, pred_best)
@printf("Verification: re-simulated best amplitude R² = %.4f\n", r2_check)

if save_figures
    plt2 = scatter(vec(observed), vec(pred_best);
        xlabel="Observed (hippocampal data)",
        ylabel="Predicted (best amp = $(round(best_amp,digits=3)))",
        title=@sprintf("Predicted vs Observed (R² = %.3f)", r2_check),
        legend=false, alpha=0.5, markersize=4)
    plot!(identity, c=:gray, ls=:dash)
    savefig("$(out_prefix)/predicted_vs_observed_best_amp.pdf")
    println("Saved plot → $(out_prefix)/predicted_vs_observed_best_amp.pdf")
end

println("✓ Done.")
