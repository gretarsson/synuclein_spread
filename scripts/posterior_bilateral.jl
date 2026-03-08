using Serialization
using Statistics
using CSV
using DataFrames
using MCMCChains
using PathoSpread

# ============================================================
# SET THIS
# ============================================================

base = "new_hippo_DIFFGA_RETRO"
inference_file = "simulations/"*base*".jls"
output_dir = "figures/posterior/"

mkpath(output_dir)

# ============================================================
# LOAD INFERENCE
# ============================================================

println("Loading inference...")
inference = deserialize(inference_file)

chain  = inference["chain"]
priors = inference["priors"]
labels = String.(inference["labels"])
ode_name = inference["ode"]

prior_keys = collect(keys(priors))

println("Loaded inference: ", inference_file)
println("ODE: ", ode_name)
println("Number of regions: ", length(labels))

# ============================================================
# HELPER: posterior samples for p[i]
# ============================================================

function posterior_samples_from_p(chain::Chains, i::Int)
    var = Symbol("p[$i]")

    if !(var in names(chain))
        error("Could not find $(var) in chain.")
    end

    vals = chain[var]
    return vec(Array(vals))
end

# ============================================================
# HELPER: get ordered prior names for a parameter family
# e.g. beta[1], beta[2], ...
# ============================================================

function parameter_names_by_prefix(prior_keys::Vector{String}, prefix::String)
    names_ = filter(n -> startswith(n, prefix * "["), prior_keys)
    sort!(names_, by = n -> parse(Int, match(Regex("^" * prefix * "\\[(\\d+)\\]\$"), n).captures[1]))
    return names_
end

# ============================================================
# HELPER: posterior means for a parameter family from p[i]
# ============================================================

function posterior_mean_vector_by_prefix(chain::Chains, prior_keys::Vector{String}, prefix::String)
    names_ = parameter_names_by_prefix(prior_keys, prefix)

    means = Float64[]
    for pname in names_
        idx = findfirst(==(pname), prior_keys)
        idx === nothing && error("Could not find $pname in prior keys.")
        push!(means, mean(posterior_samples_from_p(chain, idx)))
    end

    return means
end

# ============================================================
# HELPER: write region,value csv
# ============================================================

function write_region_param_csv(outfile, regions, values, colname)
    if length(regions) != length(values)
        error("Length mismatch: $(length(regions)) regions vs $(length(values)) values.")
    end

    df = DataFrame(region = regions)
    df[!, Symbol(colname)] = values
    CSV.write(outfile, df)
end

# ============================================================
# DETERMINE WHETHER MODEL IS BILATERAL
# ============================================================

is_bilateral = occursin("_bilateral", ode_name)

println("Bilateral model: ", is_bilateral)

# ============================================================
# EXTRACT GROUP- OR REGION-LEVEL POSTERIOR MEANS
# ============================================================

beta_means_raw  = posterior_mean_vector_by_prefix(chain, prior_keys, "beta")
gamma_means_raw = posterior_mean_vector_by_prefix(chain, prior_keys, "gamma")

# ============================================================
# MAP TO REGIONS
# ============================================================

if is_bilateral
    region_group = PathoSpread.build_region_groups(labels)
    M = maximum(region_group)

    println("Number of bilateral groups: ", M)

    if length(beta_means_raw) != M
        error("Bilateral model: found $(length(beta_means_raw)) beta parameters, but expected $M from build_region_groups(labels).")
    end

    if length(gamma_means_raw) != M
        error("Bilateral model: found $(length(gamma_means_raw)) gamma parameters, but expected $M from build_region_groups(labels).")
    end

    beta_region_means  = beta_means_raw[region_group]
    gamma_region_means = gamma_means_raw[region_group]
else
    N = length(labels)

    if length(beta_means_raw) != N
        error("Non-bilateral model: found $(length(beta_means_raw)) beta parameters, but expected $N regions.")
    end

    if length(gamma_means_raw) != N
        error("Non-bilateral model: found $(length(gamma_means_raw)) gamma parameters, but expected $N regions.")
    end

    beta_region_means  = beta_means_raw
    gamma_region_means = gamma_means_raw
end

# ============================================================
# SAVE
# ============================================================

beta_file  = joinpath(output_dir, "$(base)_beta_optimal.csv")
gamma_file = joinpath(output_dir, "$(base)_gamma_optimal.csv")

write_region_param_csv(beta_file,  labels, beta_region_means,  "beta_mean_post")
write_region_param_csv(gamma_file, labels, gamma_region_means, "gamma_mean_post")

println("Saved:")
println("  ", beta_file)
println("  ", gamma_file)