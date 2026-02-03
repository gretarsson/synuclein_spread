# ============================================================
# Posterior-mean forward simulation from a .jls inference file
# - Load inference dict saved via Serialization.serialize(.jls)
# - Take posterior means of parameters (and seed if Bayesian)
# - Simulate ODE at data timepoints
# - Plot all regional trajectories in one figure (sanity check)
# - Save region × time matrix to CSV (rows = region names)
# ============================================================

using Serialization
using Statistics
using MCMCChains
using DifferentialEquations
using DataFrames
using CSV
using Plots
using PathoSpread

# ----------------------------
# User: set this path
# ----------------------------
inference_path = "simulations/DIFFGA_RETRO.jls"   # <-- CHANGE ME
out_csv_path   = "results/simulated_spread/DIFFGA_RETRO.csv"

# ----------------------------
# Load inference dictionary
# ----------------------------
inference = load_inference(inference_path)

# ----------------------------
# Basic objects we expect
# ----------------------------
@assert haskey(inference, "chain")      "inference missing key: \"chain\""
@assert haskey(inference, "priors")     "inference missing key: \"priors\""
@assert haskey(inference, "labels")     "inference missing key: \"labels\""
@assert haskey(inference, "timepoints") "inference missing key: \"timepoints\""
@assert haskey(inference, "L")          "inference missing key: \"L\""
@assert haskey(inference, "ode")        "inference missing key: \"ode\""
@assert haskey(inference, "sol_idxs")   "inference missing key: \"sol_idxs\""
@assert haskey(inference, "u0")         "inference missing key: \"u0\""

chain      = inference["chain"]
priors     = inference["priors"]
labels     = inference["labels"]
timepoints = inference["timepoints"]
L          = inference["L"][1]
Ltuple     = inference["L"]
W          = -L
ode_name   = inference["ode"]
sol_idxs   = inference["sol_idxs"]
seed   = inference["seed_idx"]

# These must exist in your session / project:
# - `odes` (a Dict mapping ode_name => RHS function)
# - `make_ode_problem` (your helper that builds ODEProblem)
@assert @isdefined(odes)             "You must have `odes` defined in the session (Dict of RHS functions)."
@assert @isdefined(make_ode_problem) "You must have `make_ode_problem` defined in the session."

@assert haskey(odes, ode_name) "odes does not contain key \"$ode_name\"."
ode_rhs = odes[ode_name]

# ----------------------------
# Posterior mean parameters
# Assumption (matches your earlier code):
# - parameters are stored as chain[:p]
# - σ exists as separate parameter :σ
# - number of ODE parameters = index of \"σ\" in keys(priors) - 1
# ----------------------------
#ks = collect(keys(priors))
#σ_idx = findfirst(==("σ"), ks)
#@assert σ_idx !== nothing "Could not infer number of ODE parameters because \"σ\" not found in priors keys."
#N_pars = σ_idx - 1
#@assert N_pars > 0 "Inferred N_pars ≤ 0. Something is off with priors ordering/contents."

# ----------------------------
# Posterior mean parameters
# Parameters are stored as scalars: p[1], p[2], ..., p[N]
# ----------------------------

chain_param_syms = names(chain, :parameters)
# Find all p[i] parameters
p_syms = filter(s -> startswith(String(s), "p["), chain_param_syms)
@assert !isempty(p_syms) "No parameters of the form p[i] found in chain."
# Sort by index i
p_syms_sorted = sort(
    p_syms,
    by = s -> parse(Int, replace(String(s), r"[^\d]" => ""))
)
# Number of ODE parameters
N_pars = length(p_syms_sorted)
# Posterior mean parameter vector (in correct order)
p = [mean(chain[s]) for s in p_syms_sorted]
# Noise parameter (separate)
@assert :σ in chain_param_syms "σ not found in chain parameters."
σ = mean(chain[:σ])

# ----------------------------
# Posterior mean initial condition (seed handling)
# ----------------------------
u0 = copy(inference["u0"])

@assert haskey(inference, "bayesian_seed") "inference missing key: \"bayesian_seed\""
bayesian_seed = inference["bayesian_seed"]


# find posterior mode
u0 = copy(inference["u0"])
if get(inference, "bayesian_seed", false)
    # Get parameter names and seed indices
    par_names = names(chain, :parameters)
    seed_ch_idx = findall(n -> startswith(String(n), "seed"), par_names)
    isempty(seed_ch_idx) && error("No seed parameters found in chain.")
    seed_ch_idx = sort(seed_ch_idx)

    # If single seed
    if isa(seed, Int)
        #u0[seed] = chain.value[argmax[1], seed_ch_idx[1], argmax[2]]  # posterior mode
        u0[seed] = mean(chain.value[:, seed_ch_idx[1], :])  # average posterior
    else
        # Multiple seeds — assume same order as priors
        for (i, sidx) in enumerate(seed)
            #u0[sidx] = chain.value[argmax[1], seed_ch_idx[i], argmax[2]]  # posterior mode
            u0[sidx] = mean(chain.value[:, seed_ch_idx[i], :])  # average posterior
        end
    end

else
    # Fixed seeding (non-Bayesian)
    if isa(seed, Int)
        u0[seed] = inference["seed_value"]
    else
        u0[seed] .= inference["seed_value"]
    end
end

# ----------------------------
# Build ODE problem and solve at data timepoints
# ----------------------------
factors = [1. for _ in 1:N_pars]  # no scaling
prob = make_ode_problem(ode_rhs;
labels     = labels,
Ltuple     = Ltuple,
factors    = factors,
u0         = u0,
timepoints = inference["timepoints"],
seed_indices = inference["seed_idx"],
)

sol = solve(
    prob,
    Tsit5();
    saveat = timepoints,
    reltol = 1e-8,
    abstol = 1e-8,
    p = p
)

# Extract (regions × time) matrix
X = Array(sol[sol_idxs, :])

# ----------------------------
# Plot: all trajectories in one plot
# ----------------------------
plt = plot(
    xlabel = "Time",
    ylabel = "Pathology",
    title  = "Posterior mean forward simulation",
    legend = false,
    ylims = (0, 2.0)
)

for i in 1:size(X, 1)
    plot!(plt, timepoints, X[i, :]; lw=1, alpha=0.5)
end

display(plt)

# ----------------------------
# Save to CSV:
# rows = regions (labels)
# columns = timepoints (as strings)
# ----------------------------
@assert length(labels) == size(X, 1) "labels length does not match number of simulated regions."

df = DataFrame(region = labels)
for (j, t) in enumerate(timepoints)
    # Use string(t) as column name; this keeps exact timepoint values.
    df[!, string(t)] = X[:, j]
end

CSV.write(out_csv_path, df)

println("Wrote CSV: $(abspath(out_csv_path))")


# ============================  
data = mean3(inference["data"])
error_X = (data .- X).^2

# Convert missing -> NaN (Float64)
error_X_clean = coalesce.(error_X, NaN)

# Sanity check
@assert size(error_X_clean) == size(X)

# Build DataFrame
err_df = DataFrame(region = labels)
for (j, t) in enumerate(timepoints)
    err_df[!, string(t)] = error_X_clean[:, j]
end

# Write CSV
err_csv_path = replace(out_csv_path, ".csv" => "_squared_error.csv")
CSV.write(err_csv_path, err_df)

println("Wrote squared-error CSV: $(abspath(err_csv_path))")



# ============================
# Save per-region mean parameter vectors X[i] using p + priors order
# ============================

prior_keys = String.(collect(keys(priors)))

# Find all base names X that appear as X[i]
bases = sort(unique(
    match(r"^([A-Za-z_]\w*)\[\d+\]$", k).captures[1]
    for k in prior_keys
    if occursin(r"^[A-Za-z_]\w*\[\d+\]$", k)
))

for base in bases
    idxs = findall(k -> occursin(Regex("^$(base)\\[\\d+\\]\$"), k), prior_keys)
    isempty(idxs) && continue

    # Only write those that are truly region-wise vectors
    if length(idxs) != length(labels)
        continue
    end

    vals = p[idxs]  # this is the whole point: p aligns with priors order

    df_param = DataFrame(region = labels, mean_value = vals)
    param_csv_path = replace(out_csv_path, ".csv" => "_mean_$(base).csv")
    CSV.write(param_csv_path, df_param)

    println("Wrote per-region mean parameter CSV: $(abspath(param_csv_path))")
end

