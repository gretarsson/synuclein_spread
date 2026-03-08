using PathoSpread, CSV, DataFrames, Statistics, ProgressMeter, Random, HypothesisTests, Plots
using Serialization, MCMCChains, StatsBase

# Settings
# Choose how to summarise the posterior across samples:
#   :mean   -> posterior mean
#   :median -> posterior median
#   :map    -> joint MAP (mode) via max log-posterior (:lp)
const PARAM_SUMMARY = :mean   # <- change here if you want :mean or :median or :map

# to skip zero regions or not, and how to define zero regions (maximum over time below zero_threshold)
const skip_zero_regions = false
const zero_threshold = 0.01

# significance level
const α = 0.05

# ============================================================
# HELPERS
# ============================================================

"""
    fdr_bh(pvals::AbstractVector{<:Real})

Compute Benjamini–Hochberg FDR-adjusted p-values.
Input and output vectors have the same ordering.
"""
function fdr_bh(pvals::AbstractVector{<:Real})
    N = length(pvals)
    p = Float64.(pvals)

    # keep NaNs in place
    valid = .!isnan.(p)
    p_valid = p[valid]

    if isempty(p_valid)
        return fill(NaN, N)
    end

    p_sorted, idx = sort(p_valid), sortperm(p_valid)

    q_raw = p_sorted .* (length(p_valid) ./ (1:length(p_valid)))

    q_sorted = similar(q_raw)
    q_sorted[end] = min(q_raw[end], 1.0)
    for i in (length(p_valid)-1):-1:1
        q_sorted[i] = min(q_raw[i], q_sorted[i+1])
    end

    q_valid = similar(q_sorted)
    q_valid[idx] = q_sorted

    q = fill(NaN, N)
    q[valid] = q_valid
    return q
end

"""
    posterior_samples_from_p(chain::Chains, i::Int)

Extract posterior samples for p[i].
"""
function posterior_samples_from_p(chain::Chains, i::Int)
    var = Symbol("p[$i]")
    if !(var in names(chain))
        error("Could not find $(var) in chain.")
    end
    vals = chain[var]
    return vec(Array(vals))
end

"""
    ordered_param_names(prior_keys, prefix)

Return e.g. ["beta[1]", "beta[2]", ...] in numeric order.
"""
function ordered_param_names(prior_keys::Vector{String}, prefix::String)
    names_ = filter(n -> startswith(n, prefix * "["), prior_keys)
    sort!(names_, by = n -> parse(Int, match(Regex("^" * prefix * "\\[(\\d+)\\]\$"), n).captures[1]))
    return names_
end

"""
    posterior_summary_vector_from_p(inference, prefix; summary=:mean)

Extract summary vector for a parameter family stored in p[i], using priors order.
Returns the raw vector at the level stored in the model:
- non-bilateral: per-region
- bilateral: per-bilateral-group
"""
function posterior_summary_vector_from_p(inference::Dict, prefix::String; summary::Symbol=:mean)
    chain = inference["chain"]
    priors = inference["priors"]
    prior_keys = collect(keys(priors))

    param_names = ordered_param_names(prior_keys, prefix)
    idxs = [findfirst(==(pname), prior_keys) for pname in param_names]

    if summary == :mean
        return [mean(posterior_samples_from_p(chain, i)) for i in idxs]
    elseif summary == :median
        return [median(posterior_samples_from_p(chain, i)) for i in idxs]
    elseif summary == :map
        lp_mat = Array(chain[:lp])
        imax = argmax(lp_mat)
        idx = CartesianIndices(lp_mat)[imax]
        i_idx = idx[1]
        c_idx = idx[2]
        return [Array(chain[i_idx, i, c_idx])[1] for i in idxs]
    else
        error("Unknown summary = $summary")
    end
end

"""
    expand_bilateral_to_regions(values_group, model_regions)

Expand bilateral-group vector to region-level vector using build_region_groups(model_regions).
"""
function expand_bilateral_to_regions(values_group::AbstractVector, model_regions::Vector{String})
    region_group = build_region_groups(model_regions)
    M = maximum(region_group)

    if length(values_group) != M
        error("Length mismatch: got $(length(values_group)) group values, but build_region_groups(labels) gives $M groups.")
    end

    return values_group[region_group]
end

# ============================================================
# READ GENE EXPRESSION DATASET
# ============================================================

gene_data = CSV.read("data/avg_Pangea_exp.csv", DataFrame)

gene_regions = String.(gene_data[:, 1])
gene_IDs = names(gene_data)[2:end]
gene_matrix = Matrix(gene_data[:, 2:end])

# ============================================================
# READ DISEASE SPREADING MODEL
# ============================================================

simulation = "BILATERAL_DIFFGA_RETRO"
inference = load_inference("simulations/" * simulation * ".jls")

chain = inference["chain"]
priors = inference["priors"]
model_regions = String.(inference["labels"])
parameter_names = collect(keys(priors))
ode_name = inference["ode"]

println("Loaded simulation: ", simulation)
println("ODE: ", ode_name)

# ============================================================
# NONZERO REGIONS
# ============================================================

nonzero_nodes = nonzero_regions(inference["data"], eps=zero_threshold)

# ============================================================
# EXTRACT beta / gamma SUMMARIES
# ============================================================

is_bilateral = occursin("_bilateral", ode_name)
println("Bilateral model: ", is_bilateral)

beta_raw  = posterior_summary_vector_from_p(inference, "beta";  summary=PARAM_SUMMARY)
gamma_raw = posterior_summary_vector_from_p(inference, "gamma"; summary=PARAM_SUMMARY)

if PARAM_SUMMARY == :map
    lp_mat = Array(chain[:lp])
    imax = argmax(lp_mat)
    idx = CartesianIndices(lp_mat)[imax]
    println("MAP sample = $(idx[1]), chain = $(idx[2]), lp = $(lp_mat[idx[1], idx[2]])")
end

if is_bilateral
    beta_vec  = expand_bilateral_to_regions(beta_raw, model_regions)
    gamma_vec = expand_bilateral_to_regions(gamma_raw, model_regions)
else
    beta_vec  = beta_raw
    gamma_vec = gamma_raw
end

if length(beta_vec) != length(model_regions)
    error("beta vector length mismatch after expansion.")
end
if length(gamma_vec) != length(model_regions)
    error("gamma vector length mismatch after expansion.")
end

# ============================================================
# MAP GENE REGIONS TO MODEL REGIONS
# ============================================================

# maps from region in gene data to matching regions in model
# a match is considered when it's either preceded by "i" or "c"
# and then either stops or continues with a non-letter
region_map = Dict{String, Vector{String}}()

for gr in gene_regions
    gr_str = String(gr)
    L = length(gr_str)
    matches = String[]

    for (i, mr) in enumerate(model_regions)
        if skip_zero_regions && !(i in nonzero_nodes)
            continue
        end

        mr_str = String(mr)

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

# model region lookup
model_index = Dict(model_regions[i] => i for i in eachindex(model_regions))

# ============================================================
# FREQUENTIST ANALYSIS
# ============================================================

println("\n=== Running frequentist analysis using $(PARAM_SUMMARY) summary ===")

R = length(gene_regions)
G = size(gene_matrix, 2)

for (par_name, par_vec) in [("beta", beta_vec), ("gamma", gamma_vec)]

    println("  Computing correlations for $par_name ...")

    # Build regional parameter vector aligned with gene regions
    par_gene = Array{Float64}(undef, R)

    for (r, gr) in enumerate(gene_regions)
        regions = region_map[gr]

        if isempty(regions)
            par_gene[r] = NaN
            continue
        end

        idxs = [model_index[mr] for mr in regions]
        par_gene[r] = mean(par_vec[idxs])
    end

    # --------------------------------------------------------
    # Correlations
    # --------------------------------------------------------
    r_vec = Array{Float64}(undef, G)
    p_vec = Array{Float64}(undef, G)

    for g in 1:G
        expr = gene_matrix[:, g]
        mask = .!isnan.(par_gene)

        if sum(mask) < 3
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

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------
    out_df = DataFrame(
        gene   = gene_IDs,
        r      = Float32.(r_vec),
        p_un   = Float32.(p_vec),
        p_bonf = Float32.(p_bonf),
        p_fdr  = Float32.(p_fdr)
    )

    if skip_zero_regions
        out_path = "results/gene_correlation/nonzero_" * simulation * "_gene_corr_$(par_name).csv"
    else
        out_path = "results/gene_correlation/" * simulation * "_gene_corr_$(par_name).csv"
    end

    CSV.write(out_path, out_df)
    println("  Saved frequentist summary to $out_path")

    # --------------------------------------------------------
    # Count significant hits
    # --------------------------------------------------------
    valid = .!isnan.(p_vec)

    println("\nSignificant genes for $par_name (summary = $(PARAM_SUMMARY)):")
    println("  Uncorrected p < $(α)      : ", count(i -> valid[i] && p_vec[i]   < α, eachindex(p_vec)))
    println("  Bonferroni p < $(α)       : ", count(i -> valid[i] && p_bonf[i] < α, eachindex(p_bonf)))
    println("  FDR BH p < $(α)           : ", count(i -> valid[i] && p_fdr[i]  < α, eachindex(p_fdr)))

    # --------------------------------------------------------
    # Save TXT lists of significant genes
    # --------------------------------------------------------
    idx_un   = [i for i in eachindex(p_vec)   if valid[i] && p_vec[i]   < α]
    idx_fdr  = [i for i in eachindex(p_fdr)   if valid[i] && p_fdr[i]   < α]
    idx_bonf = [i for i in eachindex(p_bonf)  if valid[i] && p_bonf[i]  < α]

    genes_un   = gene_IDs[idx_un]
    genes_fdr  = gene_IDs[idx_fdr]
    genes_bonf = gene_IDs[idx_bonf]

    base = skip_zero_regions ?
        "results/gene_correlation/nonzero_" * simulation * "_siggenes_$(par_name)_$(String(PARAM_SUMMARY))" :
        "results/gene_correlation/" * simulation * "_siggenes_$(par_name)_$(String(PARAM_SUMMARY))"

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

# ============================================================
# SAVE ALL GENE NAMES
# ============================================================

open("results/gene_correlation/PANGEA_all_genes.txt", "w") do io
    for g in gene_IDs
        println(io, g)
    end
end

println("Saved list of ALL gene names to results/gene_correlation/PANGEA_all_genes.txt")