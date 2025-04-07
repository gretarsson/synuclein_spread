include("helpers.jl")
using CSV
using DataFrames
using PrettyTables
using Statistics
using Distributions
using Serialization
using Plots

# -------------------------------
# 1. Load Inference and Data
# -------------------------------
simulation = "total_death_simplifiedii_nodecay_N=448_threads=1_var1_olddecay_withx_notrunc"
inference_obj = deserialize("simulations/" * simulation * ".jls")
data = inference_obj["data"]  # dimensions: n_vars × n_time × n_samples
n_vars, n_time, n_samples = size(data)

# -------------------------------
# 2. Compute Growth for Each Region
# -------------------------------
# Define growth as the difference between the mean pathology at the final and initial time points.
growth = zeros(n_vars)
for i in 1:n_vars
    initial_mean = mean(skipmissing(data[i, 1, :]))
    final_mean   = mean(skipmissing(data[i, n_time, :]))
    growth[i] = final_mean - initial_mean
end

# -------------------------------
# 3. Compare Posterior and Prior for Each Region
# -------------------------------
chain = inference_obj["chain"]
priors = inference_obj["priors"]
prior_keys = collect(keys(priors))  # Expected keys: "β[i]" and "d[i]" for i = 1,…,n_vars

region_ids      = String[]
region_growth   = Float64[]

beta_post_means = Float64[]
beta_post_stds  = Float64[]
d_post_means    = Float64[]
d_post_stds     = Float64[]

beta_prior_means = Float64[]
beta_prior_stds  = Float64[]
d_prior_means    = Float64[]
d_prior_stds     = Float64[]

for i in 1:n_vars
    if growth[i] < 1e-6
        push!(region_ids, string(i))
        push!(region_growth, growth[i])
        
        # Construct parameter keys.
        beta_key = "β[$(i)]"
        d_key    = "d[$(i)]"
        
        # Find the corresponding indices.
        beta_idx = findfirst(x -> x == beta_key, prior_keys)
        d_idx    = findfirst(x -> x == d_key, prior_keys)
        
        # Extract posterior samples from the chain.
        beta_samples = chain[:, beta_idx, :]
        d_samples    = chain[:, d_idx, :]
        
        push!(beta_post_means, mean(beta_samples))
        push!(beta_post_stds, std(beta_samples))
        push!(d_post_means, mean(d_samples))
        push!(d_post_stds, std(d_samples))
        
        # Extract prior distributions.
        beta_prior = priors[beta_key]
        d_prior    = priors[d_key]
        
        push!(beta_prior_means, mean(beta_prior))
        push!(beta_prior_stds, std(beta_prior))
        push!(d_prior_means, mean(d_prior))
        push!(d_prior_stds, std(d_prior))
    end
end

# -------------------------------
# 4. Create and Save the Summary Table
# -------------------------------
results = DataFrame(
    Region          = region_ids,
    Growth          = region_growth,
    Beta_Post_Mean  = beta_post_means,
    Beta_Post_Std   = beta_post_stds,
    Beta_Prior_Mean = beta_prior_means,
    Beta_Prior_Std  = beta_prior_stds,
    d_Post_Mean     = d_post_means,
    d_Post_Std      = d_post_stds,
    d_Prior_Mean    = d_prior_means,
    d_Prior_Std     = d_prior_stds
)

PrettyTables.pretty_table(results)
#CSV.write("regions_analysis.csv", results)

# -------------------------------
# 5. VISUALIZING
# -------------------------------
# Create a Beta comparison plot.
p1 = Plots.plot(1:length(beta_post_means), beta_post_means,
    yerror = beta_post_stds,
    seriestype = :scatter,
    label = "Posterior β",
    xlabel = "Region Index",
    ylabel = "β Value",
    title = "β Parameter Comparison",
    markersize = 4);

Plots.scatter!(p1, 1:length(beta_prior_means), beta_prior_means,
    yerror = beta_prior_stds,
    label = "Prior β",
    markershape = :diamond,
    markersize = 4);

# Create a d comparison plot.
p2 = Plots.plot(1:length(d_post_means), d_post_means,
    yerror = d_post_stds,
    seriestype = :scatter,
    label = "Posterior d",
    xlabel = "Region Index",
    ylabel = "d Value",
    title = "d Parameter Comparison",
    markersize = 4);

Plots.scatter!(p2, 1:length(d_prior_means), d_prior_means,
    yerror = d_prior_stds,
    label = "Prior d",
    markershape = :diamond,
    markersize = 4);

# Combine the two plots into a single layout.
combined_plot = Plots.plot(p1, p2, layout = (1, 2), legend = :outertop);
display(combined_plot)
# Optionally, save the figure.
Plots.savefig(combined_plot, "figures/zero_regions/"*simulation*".png")
