using PathoSpread, CSV, DataFrames, Statistics, ProgressMeter, Random, HypothesisTests, Plots

# Settings
# Choose how to summarise the posterior across samples:
#   :mean   -> posterior mean
#   :median -> posterior median
#   :map    -> joint MAP (mode) via max log-posterior (:lp)
const PARAM_SUMMARY = :mean   # <- change here if you want :mean or :median
# to skip zero regions or not, and how to define zero regions (maximum over time below zero_threshold)
const skip_zero_regions = false
const zero_threshold = 0.01 
# significance level
const α = 0.05

"""
    fdr_bh(pvals::AbstractVector{<:Real})

Compute Benjamini–Hochberg FDR-adjusted p-values.
Input and output vectors have the same ordering.
"""
function fdr_bh(pvals::AbstractVector{<:Real})
    N = length(pvals)
    p = Float64.(pvals)  # ensure float

    # sort p-values with their original indices
    p_sorted, idx = sort(p), sortperm(p)

    # compute BH-adjusted values in sorted order
    q_raw = p_sorted .* (N ./ (1:N))

    # enforce monotonicity from largest to smallest
    q_sorted = similar(q_raw)
    q_sorted[end] = min(q_raw[end], 1.0)
    for i in (N-1):-1:1
        q_sorted[i] = min(q_raw[i], q_sorted[i+1])
    end

    # put adjusted q-values back to original order
    q = similar(q_sorted)
    q[idx] = q_sorted

    return q
end

# Read Gene expression dataset
gene_data = CSV.read("data/avg_Pangea_exp.csv", DataFrame)

# find region and gene IDs, and convert to matrix
gene_regions = String.(gene_data[:, 1])
gene_IDs = names(gene_data)[2:end]
gene_matrix = Matrix(gene_data[:, 2:end]) 

# Read Disease Spreading Model
simulation = "DIFFG_RETRO"
inference = load_inference("simulations/"*simulation*".jls")

# Extract raw data used in inference
nonzero_nodes = nonzero_regions(inference["data"], eps=zero_threshold)   # dims: region × time × sample (usually)

# look at histogram of maximum of regions
#mean_data = mean3(inference["data"])
#mean_data = coalesce.(mean_data, 0.0)
#max_data = vec(maximum(mean_data,dims=2))
#scatter(max_data)
#hline!([0.01], color=:red, linestyle=:dash, label="threshold")
#ylims!(0,0.1)

# Extract parameters of interest
chain = inference["chain"]
priors = inference["priors"]
model_regions = inference["labels"]
parameter_names = collect(keys(priors))
# find indices of beta and gamma parameters in chain
beta_ind = findall(k -> startswith(k, "beta["), parameter_names)
gamma_ind = findall(k -> startswith(k, "gamma["), parameter_names)

# find regions common to both gene data and model
# maps from region in gene data to matching regions in model, a match is considered when it's either preceded by "i" or "c" or ends with "-X", so we include subregions and lateral regions
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
        # region name after hemisphere prefix
        if length(mr_str) < 1 + L
            continue
        end
        # extract the candidate substring *case-sensitive*
        mr_sub = mr_str[2:1+L]
        # strict match
        if mr_sub == gr_str
            # exact match ends here → accept (iPA, cPA)
            if length(mr_str) == 1 + L
                push!(matches, mr_str)
                continue
            end
            # otherwise check that next character is NOT a letter
            next_char = mr_str[2+L]
            if !isletter(next_char)
                push!(matches, mr_str)
            end
        end
    end
    region_map[gr_str] = matches
end
region_map
n_empty = count(v -> !isempty(v), values(region_map))  # returns 2
# look-up for model regions
model_index = Dict(model_regions[i] => i for i in eachindex(model_regions))


#######################################################################
# === FREQUENTIST ANALYSIS USING SUMMARY PARAMETER FIELD ==============
#######################################################################
println("\n=== Running frequentist analysis using summary parameter fields ===")


# Extract posterior arrays: iter × param × chain
beta_post  = Array(chain[:, beta_ind,  :])
gamma_post = Array(chain[:, gamma_ind, :])

using StatsBase
# Posterior mean across iterations and chains
beta_mean  = vec(mean(beta_post,  dims=1))
gamma_mean = vec(mean(gamma_post, dims=1))

# Posterior median across iterations and chains
beta_median  = vec(median(beta_post,  dims=1))
gamma_median = vec(median(gamma_post, dims=1))

# log-posterior for each sample × chain
lp_mat = Array(chain[:lp])        # <-- THIS must be defined explicitly
# Find MAP index (linear)
imax = argmax(lp_mat)
# Convert linear index to sample + chain indices
idx = CartesianIndices(lp_mat)[imax]
i_idx = idx[1]       # sample index
c_idx = idx[2]       # chain index

println("MAP sample = $(i_idx), chain = $(c_idx), lp = $(lp_mat[i_idx, c_idx])")
beta_map  = vec(Array(chain[i_idx, beta_ind, c_idx]))
gamma_map = vec(Array(chain[i_idx, gamma_ind, c_idx]))

# Pick which summary to use
function pick_summary(par_name::String)
    if PARAM_SUMMARY == :mean
        return par_name == "beta" ? beta_mean  : gamma_mean
    elseif PARAM_SUMMARY == :median
        return par_name == "beta" ? beta_median : gamma_median
    elseif PARAM_SUMMARY == :map
        return par_name == "beta" ? beta_map    : gamma_map
    else
        error("Unknown PARAM_SUMMARY = $PARAM_SUMMARY")
    end
end

R = length(gene_regions)
G = size(gene_matrix, 2)

for par_name in ["beta", "gamma"]

    println("  Using $(PARAM_SUMMARY) summary for $par_name")

    par_vec = pick_summary(par_name)

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

    #####################
    # --- Correlations --
    #####################
    println("  Computing correlations for $par_name ...")

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

    ########################
    # === Save CSV output ==
    ########################
    out_df = DataFrame(
        gene   = gene_IDs,
        r      = Float32.(r_vec),
        p_un   = Float32.(p_vec),
        p_bonf = Float32.(p_bonf),
        p_fdr  = Float32.(p_fdr)
    )

    if skip_zero_regions
        out_path = "results/gene_correlation/nonzero_DIFFG_gene_correlation_$(par_name)_$(String(PARAM_SUMMARY)).csv"
    else
        out_path = "results/gene_correlation/DIFFG_gene_correlation_$(par_name)_$(String(PARAM_SUMMARY)).csv"
    end
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
    # Indices of significant genes
    idx_un   = [i for i in eachindex(p_vec)  if valid[i] && p_vec[i]   < α]
    idx_fdr  = [i for i in eachindex(p_fdr) if valid[i] && p_fdr[i]  < α]
    idx_bonf = [i for i in eachindex(p_bonf) if valid[i] && p_bonf[i] < α]
    # Gene names for each significance class
    genes_un   = gene_IDs[idx_un]
    genes_fdr  = gene_IDs[idx_fdr]
    genes_bonf = gene_IDs[idx_bonf]
    # Output file names
    base = skip_zero_regions ?
        "results/gene_correlation/nonzero_DIFFG_siggenes_$(par_name)_$(String(PARAM_SUMMARY))" :
        "results/gene_correlation/DIFFG_siggenes_$(par_name)_$(String(PARAM_SUMMARY))"

    # Write each list to a text file (one gene name per line)
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

println("\nSaved list of ALL gene names to DIFFGA_all_genes.txt")

#######################################################################
# === END FREQUENTIST ANALYSIS =======================================
#######################################################################


# BAYESIAN ANALYSIS (PROBABLY INACCURATE APPROACH)
# Extract posterior samples for beta and gamma
#beta_posterior = Array(chain[:,beta_ind, :])
#gamma_posterior = Array(chain[:, gamma_ind, :])
#posteriors = [beta_posterior, gamma_posterior]  
#par_names = ["beta", "gamma"]   
#
## find the correlation matrix for each parameter
#for (posterior, par_name) in zip(posteriors,par_names)
#    # Compute average beta for each gene region
#    S = size(posterior, 1)
#    R = length(gene_regions)
#
#    beta_gene = Array{Float64}(undef, S, R)
#
#    for (r, gr) in enumerate(gene_regions)
#        regions = region_map[gr]
#
#        if isempty(regions)
#            beta_gene[:, r] .= NaN
#            continue
#        end
#
#        idxs = [model_index[mr] for mr in regions]  # model indices
#
#        for s in 1:S
#            beta_gene[s, r] = mean(posterior[s, idxs])
#        end
#    end
#
#
#    # Compute Pearson correlation between gene expression and beta values
#    G = size(gene_matrix, 2)
#    pearson_matrix = Array{Float64}(undef, G, S)
#    pval_matrix = Array{Float64}(undef, G, S)
#
#    @showprogress for g in 1:G
#        expr = gene_matrix[:, g]
#
#        for s in 1:S
#            beta_vec = beta_gene[s, :]
#
#            # mask invalid (NaN from unmatched regions)
#            mask = .!isnan.(beta_vec)
#
#            if sum(mask) < 3
#                pearson_matrix[g, s] = NaN
#            else
#                pearson_matrix[g, s] = cor(expr[mask], beta_vec[mask])
#                pval_matrix[g, s] = pvalue(CorrelationTest(expr[mask], beta_vec[mask]))
#            end
#        end
#    end
#
#    # Convert pearson matrix to DataFrame for saving (and convert to Float32 to save space)
#    pearson_df = DataFrame(Float32.(pearson_matrix), :auto)
#    # Give columns meaningful names (optional)
#    rename!(pearson_df, Symbol.(string.("sample_", 1:size(pearson_matrix, 2))))
#    # Give rows gene names (optional but highly recommended)
#    pearson_df.gene = gene_IDs
#    # Move the gene column to the front
#    select!(pearson_df, :gene, Not(:gene))
#    # save pearson matrix
#    CSV.write("results/gene_pearson_$(par_name).csv", pearson_df)
#    println("Saved gene pearson r-values for $par_name to results/gene_pearson_$(par_name).csv")
#
#    # Convert pearson matrix to DataFrame for saving (and convert to Float32 to save space)
#    pval_df = DataFrame(Float32.(pval_matrix), :auto)
#    # Give columns meaningful names (optional)
#    rename!(pval_df, Symbol.(string.("sample_", 1:size(pval_matrix, 2))))
#    # Give rows gene names (optional but highly recommended)
#    pval_df.gene = gene_IDs
#    # Move the gene column to the front
#    select!(pval_df, :gene, Not(:gene))
#    # save pearson matrix
#    CSV.write("results/gene_pval_$(par_name).csv", pval_df)
#    println("Saved gene pearson p-values for $par_name to results/gene_pval_$(par_name).csv")
#
#    ## === Create summary file: mean r and mean p per gene ===
#    # Compute means across samples, ignoring NaN entries
#    mean_r = [mean(skipmissing(pearson_matrix[g, :])) for g in 1:G]
#    mean_p = [mean(skipmissing(pval_matrix[g, :])) for g in 1:G]
#    mean_p_bonf = [min(p * G, 1.0) for p in mean_p]  # Bonferroni correction    
#    mean_p_fdr = fdr_bh(mean_p)  # FDR correction
#
#    summary_df = DataFrame(
#        gene = gene_IDs,
#        r = Float32.(mean_r),
#        p_bonf = Float32.(mean_p_bonf),
#        p_fdr = Float32.(mean_p_fdr),
#        p_un = Float32.(mean_p)
#    )
#
#    CSV.write("results/gene_summary_$(par_name).csv", summary_df)
#    println("Saved summary statistics for $par_name to results/gene_summary_$(par_name).csv")
#
#end
#
#
#
#for (posterior, par_name) in zip(posteriors,par_names)
#    # read the pearson and p-values 
#    pearson_matrix = Matrix(CSV.read("results/gene_pearson_$(par_name).csv", DataFrame))[:,2:end]   
#    pval_matrix = Matrix(CSV.read("results/gene_pval_$(par_name).csv", DataFrame))[:,2:end]   
#
#    ## === Create summary file: mean r and mean p per gene ===
#    G = size(pearson_matrix, 1)
#    # Compute means across samples, ignoring NaN entries
#    mean_r = [mean(pearson_matrix[g, :]) for g in 1:G]
#    mean_p = [mean(pval_matrix[g, :]) for g in 1:G]
#    mean_p_bonf = [min(p * G, 1.0) for p in mean_p]  # Bonferroni correction    
#    mean_p_fdr = fdr_bh(mean_p)  # FDR correction
#
#    summary_df = DataFrame(
#        gene = gene_IDs,
#        r = Float32.(mean_r),
#        p_bonf = Float32.(mean_p_bonf),
#        p_fdr = Float32.(mean_p_fdr),
#        p_un = Float32.(mean_p)
#    )
#
#    CSV.write("results/gene_summary_$(par_name).csv", summary_df)
#    println("Saved summary statistics for $par_name to results/gene_summary_$(par_name).csv")
#
#    # === Count significant genes at α = 0.05 ===
#    α = 0.05
#    n_sig_un   = count(<(α), summary_df.p_un)
#    n_sig_bonf = count(<(α), summary_df.p_bonf)
#    n_sig_fdr  = count(<(α), summary_df.p_fdr)
#
#    println("Significant genes for $par_name:")
#    println("  Uncorrected p-values (<0.05):      $n_sig_un")
#    println("  Bonferroni-corrected (<0.05):      $n_sig_bonf")
#    println("  FDR BH-corrected (<0.05):          $n_sig_fdr")
#end
#
#
#