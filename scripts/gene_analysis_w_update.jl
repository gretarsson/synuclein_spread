using PathoSpread, CSV, DataFrames, Statistics, ProgressMeter, Random, HypothesisTests, Plots
using Serialization, Distributions, OrderedCollections, MCMCChains, StatsBase

# Settings
# Choose how to summarise the posterior across samples:
#   :mean   -> posterior mean
#   :median -> posterior median
#   :map    -> joint MAP (mode) via max log-posterior (:lp)
const PARAM_SUMMARY = :mean

# to skip zero regions or not, and how to define zero regions (maximum over time below zero_threshold)
const skip_zero_regions = false
const zero_threshold = 0.01

# significance level
const α = 0.05

# updated-region filtering
const UPDATED_ONLY = true
const UPDATE_ALPHA = 0.001

"""
    fdr_bh(pvals::AbstractVector{<:Real})

Compute Benjamini–Hochberg FDR-adjusted p-values.
Input and output vectors have the same ordering.
"""
function fdr_bh(pvals::AbstractVector{<:Real})
    N = length(pvals)
    p = Float64.(pvals)

    p_sorted, idx = sort(p), sortperm(p)
    q_raw = p_sorted .* (N ./ (1:N))

    q_sorted = similar(q_raw)
    q_sorted[end] = min(q_raw[end], 1.0)
    for i in (N-1):-1:1
        q_sorted[i] = min(q_raw[i], q_sorted[i+1])
    end

    q = similar(q_sorted)
    q[idx] = q_sorted
    return q
end

"""
Load posterior samples for p[i], assuming the chain stores parameters as Symbol("p[i]").
Returns a Vector{Float64} or nothing if missing.
"""
function posterior_samples_from_p(chain::Chains, i::Int)
    var = Symbol("p[$i]")
    if !(var in names(chain))
        return nothing
    end
    vals = chain[var]
    return vec(Array(vals))
end

"""
Return parameter names with the form prefix[1], prefix[2], ..., sorted by index.
"""
function parameter_names_by_prefix(priors, prefix::AbstractString)
    names_ = filter(n -> startswith(n, prefix * "["), collect(keys(priors)))
    sort!(names_, by = n -> parse(Int, match(Regex("^" * prefix * "\\[(\\d+)\\]"), n).captures[1]))
    return names_
end

"""
One-sample KS test:
    H0: posterior samples are distributed as the prior
Returns:
    pvalue::Float64
    updated::Int
"""
function posterior_vs_prior_update_flag(samples::AbstractVector, prior_dist; alpha::Float64=UPDATE_ALPHA)
    test = ApproximateOneSampleKSTest(samples, prior_dist)
    p = pvalue(test)
    updated = Int(p < alpha)
    return p, updated
end

"""
Build Dict(region_name => updated flag) for a given parameter prefix, e.g. "beta" or "gamma".
Assumes inference["labels"] aligns with sorted prefix[i] parameters.
"""
function region_update_flags(inference::Dict, prefix::AbstractString; alpha::Float64=UPDATE_ALPHA)
    priors = inference["priors"]
    chain = inference["chain"]
    regions = String.(inference["labels"])

    param_names = parameter_names_by_prefix(priors, prefix)
    prior_keys = collect(keys(priors))

    if length(param_names) != length(regions)
        error("Length mismatch for $prefix: $(length(param_names)) parameters vs $(length(regions)) labels.")
    end

    flags = Dict{String,Int}()

    for (region, pname) in zip(regions, param_names)
        idx = findfirst(==(pname), prior_keys)
        idx === nothing && error("Could not find $pname in prior keys.")

        samples = posterior_samples_from_p(chain, idx)
        samples === nothing && error("Missing posterior samples for $pname at p[$idx].")

        prior_dist = priors[pname]
        _, updated = posterior_vs_prior_update_flag(samples, prior_dist; alpha=alpha)
        flags[String(region)] = updated
    end

    return flags
end

# Read Gene expression dataset
gene_data = CSV.read("data/avg_Pangea_exp.csv", DataFrame)

# find region and gene IDs, and convert to matrix
gene_regions = String.(gene_data[:, 1])
gene_IDs = names(gene_data)[2:end]
gene_matrix = Matrix(gene_data[:, 2:end])

# Read Disease Spreading Model
simulation = "DIFFGA_RETRO"
inference = load_inference("simulations/" * simulation * ".jls")

# Extract raw data used in inference
nonzero_nodes = nonzero_regions(inference["data"], eps=zero_threshold)

# Extract parameters of interest
chain = inference["chain"]
priors = inference["priors"]
model_regions = String.(inference["labels"])
parameter_names = collect(keys(priors))

# find indices of beta and gamma parameters in chain
beta_ind = findall(k -> startswith(k, "beta["), parameter_names)
gamma_ind = findall(k -> startswith(k, "gamma["), parameter_names)

# precompute updated flags if requested
beta_updated_flags  = UPDATED_ONLY ? region_update_flags(inference, "beta";  alpha=UPDATE_ALPHA) : Dict{String,Int}()
gamma_updated_flags = UPDATED_ONLY ? region_update_flags(inference, "gamma"; alpha=UPDATE_ALPHA) : Dict{String,Int}()

# find regions common to both gene data and model
# maps from region in gene data to matching regions in model
region_map = Dict{String, Vector{String}}()
for gr in gene_regions
    gr_str = String(gr)
    L = length(gr_str)
    matches = String[]

    for (i, mr) in enumerate(model_regions)
        # skip zero regions if specified
        if skip_zero_regions && !(i in nonzero_nodes)
            continue
        end

        mr_str = String(mr)

        # must start with 'i' or 'c'
        if !(startswith(mr_str, "i") || startswith(mr_str, "c"))
            continue
        end

        if length(mr_str) < 1 + L
            continue
        end

        mr_sub = mr_str[2:1+L]

        if mr_sub == gr_str
            if length(mr_str) == 1 + L
                push!(matches, mr_str)
                continue
            end

            next_char = mr_str[2+L]
            if !isletter(next_char)
                push!(matches, mr_str)
            end
        end
    end

    region_map[gr_str] = matches
end

# look-up for model regions
model_index = Dict(model_regions[i] => i for i in eachindex(model_regions))

#######################################################################
# === FREQUENTIST ANALYSIS USING SUMMARY PARAMETER FIELD ==============
#######################################################################
println("\n=== Running frequentist analysis using summary parameter fields ===")
println("UPDATED_ONLY = $(UPDATED_ONLY)")
if UPDATED_ONLY
    println("UPDATE_ALPHA = $(UPDATE_ALPHA)")
end

# Extract posterior arrays: iter × param × chain
beta_post  = Array(chain[:, beta_ind, :])
gamma_post = Array(chain[:, gamma_ind, :])

# Posterior mean across iterations and chains
beta_mean   = vec(mean(beta_post,  dims=1))
gamma_mean  = vec(mean(gamma_post, dims=1))

# Posterior median across iterations and chains
beta_median  = vec(median(beta_post,  dims=1))
gamma_median = vec(median(gamma_post, dims=1))

# log-posterior for each sample × chain
lp_mat = Array(chain[:lp])

# Find MAP index (linear)
imax = argmax(lp_mat)

# Convert linear index to sample + chain indices
idx = CartesianIndices(lp_mat)[imax]
i_idx = idx[1]
c_idx = idx[2]

println("MAP sample = $(i_idx), chain = $(c_idx), lp = $(lp_mat[i_idx, c_idx])")
beta_map  = vec(Array(chain[i_idx, beta_ind,  c_idx]))
gamma_map = vec(Array(chain[i_idx, gamma_ind, c_idx]))

# Pick which summary to use
function pick_summary(par_name::String)
    if PARAM_SUMMARY == :mean
        return par_name == "beta" ? beta_mean : gamma_mean
    elseif PARAM_SUMMARY == :median
        return par_name == "beta" ? beta_median : gamma_median
    elseif PARAM_SUMMARY == :map
        return par_name == "beta" ? beta_map : gamma_map
    else
        error("Unknown PARAM_SUMMARY = $PARAM_SUMMARY")
    end
end

function pick_updated_flags(par_name::String)
    if par_name == "beta"
        return beta_updated_flags
    elseif par_name == "gamma"
        return gamma_updated_flags
    else
        error("Unknown parameter name: $par_name")
    end
end

R = length(gene_regions)
G = size(gene_matrix, 2)

mkpath("results/gene_correlation")

for par_name in ["beta", "gamma"]

    println("  Using $(PARAM_SUMMARY) summary for $par_name")

    par_vec = pick_summary(par_name)
    updated_flags = pick_updated_flags(par_name)

    # Build regional parameter vector aligned with gene regions
    par_gene = Array{Float64}(undef, R)
    n_model_matches = zeros(Int, R)
    n_model_matches_after_filter = zeros(Int, R)

    for (r, gr) in enumerate(gene_regions)
        regions = region_map[gr]
        n_model_matches[r] = length(regions)

        if UPDATED_ONLY
            regions = [mr for mr in regions if get(updated_flags, mr, 0) == 1]
        end
        n_model_matches_after_filter[r] = length(regions)

        if isempty(regions)
            par_gene[r] = NaN
            continue
        end

        idxs = [model_index[mr] for mr in regions]
        par_gene[r] = mean(par_vec[idxs])
    end

    println("  Regions with at least one matched model region before filter: ",
            count(>(0), n_model_matches))
    if UPDATED_ONLY
        println("  Regions with at least one matched model region after UPDATED filter: ",
                count(>(0), n_model_matches_after_filter))
    end

    #####################
    # --- Correlations --
    #####################
    println("  Computing correlations for $par_name ...")

    r_vec = Array{Float64}(undef, G)
    p_vec = Array{Float64}(undef, G)
    n_used_vec = Array{Int}(undef, G)

    for g in 1:G
        expr = gene_matrix[:, g]
        mask = .!isnan.(par_gene)

        n_used_vec[g] = sum(mask)

        if n_used_vec[g] < 3
            r_vec[g] = NaN
            p_vec[g] = NaN
            continue
        end

        r_vec[g] = cor(expr[mask], par_gene[mask])
        p_vec[g] = pvalue(CorrelationTest(expr[mask], par_gene[mask]))
    end

    # Multiple testing corrections
    p_bonf = [isnan(p) ? NaN : min(p * G, 1.0) for p in p_vec]
    p_fdr  = fdr_bh(p_vec)

    ########################
    # === Save CSV output ==
    ########################
    out_df = DataFrame(
        gene   = gene_IDs,
        r      = Float32.(r_vec),
        p_un   = Float32.(p_vec),
        p_bonf = Float32.(p_bonf),
        p_fdr  = Float32.(p_fdr),
        n_used = n_used_vec
    )

    suffix = ""
    if skip_zero_regions
        suffix *= "_NONZERO"
    end
    if UPDATED_ONLY
        suffix *= "_UPDATED"
    end

    out_path = "results/gene_correlation/$(simulation)_gene_corr_$(par_name)_$(String(PARAM_SUMMARY))$(suffix).csv"
    CSV.write(out_path, out_df)
    println("  Saved frequentist summary to $out_path")

    # === Count significant hits (ignoring NaNs)
    valid = .!isnan.(p_vec)

    println("\nSignificant genes for $par_name (summary = $(PARAM_SUMMARY)):")
    println("  Uncorrected p < $(α)      : ", count(i -> valid[i] && p_vec[i]   < α, eachindex(p_vec)))
    println("  Bonferroni p < $(α)       : ", count(i -> valid[i] && p_bonf[i] < α, eachindex(p_bonf)))
    println("  FDR BH p < $(α)           : ", count(i -> valid[i] && p_fdr[i]  < α, eachindex(p_fdr)))

    ###############################
    # === Save TXT lists of genes ==
    ###############################
    idx_un   = [i for i in eachindex(p_vec)   if valid[i] && p_vec[i]   < α]
    idx_fdr  = [i for i in eachindex(p_fdr)   if valid[i] && p_fdr[i]   < α]
    idx_bonf = [i for i in eachindex(p_bonf)  if valid[i] && p_bonf[i]  < α]

    genes_un   = gene_IDs[idx_un]
    genes_fdr  = gene_IDs[idx_fdr]
    genes_bonf = gene_IDs[idx_bonf]

    base = "results/gene_correlation/$(simulation)_siggenes_$(par_name)_$(String(PARAM_SUMMARY))$(suffix)"

    open(base * "_uncorrected.txt", "w") do io
        for g in genes_un
            println(io, g)
        end
    end
    open(base * "_fdr.txt", "w") do io
        for g in genes_fdr
            println(io, g)
        end
    end
    open(base * "_bonferroni.txt", "w") do io
        for g in genes_bonf
            println(io, g)
        end
    end
    println("  Wrote significant-gene TXT files for $par_name")
end

println("\n=== Frequentist analysis with $(PARAM_SUMMARY) summary complete ===\n")

#################################
# === Save all gene names ======
#################################
open("results/gene_correlation/PANGEA_all_genes.txt", "w") do io
    for g in gene_IDs
        println(io, g)
    end
end

println("\nSaved list of ALL gene names to results/gene_correlation/PANGEA_all_genes.txt")