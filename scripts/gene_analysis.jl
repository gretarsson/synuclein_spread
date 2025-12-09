using PathoSpread, CSV, DataFrames, Statistics, ProgressMeter, Random


# Read Gene expression dataset
gene_data = CSV.read("data/avg_Pangea_exp.csv", DataFrame)

# find region and gene IDs, and convert to matrix
gene_regions = String.(gene_data[:, 1])
gene_IDs = names(gene_data)[2:end]
gene_matrix = Matrix(gene_data[:, 2:end]) 

# Read Disease Spreading Model
simulation = "DIFFGA_RETRO"
inference = load_inference("simulations/"*simulation*".jls")

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
    for mr in model_regions
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

# Extract posterior samples for beta and gamma
beta_posterior = Array(chain[:,beta_ind, :])
gamma_posterior = Array(chain[:, gamma_ind, :])
posteriors = [beta_posterior, gamma_posterior]  
par_names = ["beta", "gamma"]   

# look-up for model regions
model_index = Dict(model_regions[i] => i for i in eachindex(model_regions))

# find the correlation matrix for each parameter
for (posterior, par_name) in zip(posteriors,par_names)
    # Compute average beta for each gene region
    S = size(posterior, 1)
    R = length(gene_regions)

    beta_gene = Array{Float64}(undef, S, R)

    for (r, gr) in enumerate(gene_regions)
        regions = region_map[gr]

        if isempty(regions)
            beta_gene[:, r] .= NaN
            continue
        end

        idxs = [model_index[mr] for mr in regions]  # model indices

        for s in 1:S
            beta_gene[s, r] = mean(posterior[s, idxs])
        end
    end


    # Compute Pearson correlation between gene expression and beta values
    G = size(gene_matrix, 2)
    pearson_matrix = Array{Float64}(undef, G, S)
    control_matrix = Array{Float64}(undef, G, S)

    @showprogress for g in 1:G
        expr = gene_matrix[:, g]

        for s in 1:S
            beta_vec = beta_gene[s, :]

            # mask invalid (NaN from unmatched regions)
            mask = .!isnan.(beta_vec)

            if sum(mask) < 3
                pearson_matrix[g, s] = NaN
            else
                pearson_matrix[g, s] = cor(expr[mask], beta_vec[mask])
                control_matrix[g, s] = cor(expr[mask], shuffle(beta_vec[mask]))
            end
        end
    end

    # Convert pearson matrix to DataFrame for saving (and convert to Float32 to save space)
    pearson_df = DataFrame(Float32.(pearson_matrix), :auto)
    # Give columns meaningful names (optional)
    rename!(pearson_df, Symbol.(string.("sample_", 1:size(pearson_matrix, 2))))
    # Give rows gene names (optional but highly recommended)
    pearson_df.gene = gene_IDs
    # Move the gene column to the front
    select!(pearson_df, :gene, Not(:gene))
    # save pearson matrix
    CSV.write("results/gene_pearson_$(par_name).csv", pearson_df)
    println("Saved gene pearson correlations for $par_name to results/gene_pearson_$(par_name).csv")
end

# read CSV
pearson_df = CSV.read("results/gene_pearson_beta.csv", DataFrame)   
pearson_matrix = Matrix(pearson_df[:, 2:end])  # exclude gene names column

# your sample
x = pearson_matrix[1, :]

# compute sample mean and std
μ = mean(x)
σ = std(x)

# normal distribution fit
dist = Normal(μ, σ)

# histogram + PDF
histogram(x, normalize=true, alpha=0.4, bins=50)  # normalized so PDF overlays properly

# PDF curve
xs = range(minimum(x), maximum(x), length=300)
plot!(xs, pdf.(dist, xs), linewidth=2)

# normality test (Shapiro–Wilk)
test = ShapiroWilkTest(x)
println("Shapiro–Wilk p-value = ", pvalue(test))
normalitys = []
for i in axes(pearson_matrix,1)
    x = pearson_matrix[i, :]
    test = ShapiroWilkTest(x)
    push!(normalitys, pvalue(test))
end 
nonnormal = count(x -> x < 0.001, normalitys)  # number of genes rejecting normality at alpha=0.001
1 - nonnormal/size(pearson_matrix,1)  # proportion of genes rejecting normality at alpha=0.001
