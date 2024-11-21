using DelimitedFiles
using KernelDensity
using Plots, StatsBase
using DataFrames, StatsBase, GLM, ProgressMeter
include("helpers.jl");
#=
Gene analysis!!
We draw from the posterior and compute significance with Holm-Bonferroni for a chosen parameter.
We save the number of occurences of each significant gene in a dictionary and save it. Also including the number of samples
=#
gene_labels2 = readdlm("data/avg_Pangea_exp.csv",',')[1,2:end];  # names of genes
# pick simulation and parameter
simulation = "simulations/total_death_simplifiedii_N=448_threads=1_var1_normalpriors.jls"
parameter = "β"
file_name = "gene_significance"

# compute gene correlation with samples from the posterior
S = 1000
significants = [] 
null_significants = []
@showprogress for s in 1:S
    r2s, pvals, significant, gene_labels = gene_analysis(simulation,parameter;mode=false,show=false);
    push!(significants,significant)
    r2s, pvals, significant, gene_labels = gene_analysis(simulation,parameter;mode=false,show=false,null=true);
    push!(null_significants,significant)
end

#using Threads
using ProgressMeter

# Number of iterations
S = 1000

# Preallocate arrays for results
significants = Vector{Any}(undef, S)
null_significants = Vector{Any}(undef, S)

# Run the loop in parallel
@showprogress Threads.@threads for s in 1:S
    # Perform gene analysis
    r2s, pvals, significant, gene_labels = gene_analysis(simulation, parameter; mode=false, show=false)
    significants[s] = significant

    r2s, pvals, significant, gene_labels = gene_analysis(simulation, parameter; mode=false, show=false, null=true)
    null_significants[s] = significant
end



# create dictionary of how many times a gene was significant
counts = Dict{Int, Int}();
for vec in significants
    for num in vec
        counts[num] = get(counts, num, 0) + 1  # Increment the count
    end
end
labeled_counts = Dict(gene_labels[k] => counts[k] for k in keys(counts));

# save
serialize("simulations/"*file_name*"_"*parameter*".jls", (counts,labeled_counts,S));


# create dictionary for the null distribution
counts = Dict{Int, Int}();
for vec in null_significants
    for num in vec
        counts[num] = get(counts, num, 0) + 1  # Increment the count
    end
end
labeled_counts = Dict(gene_labels[k] => counts[k] for k in keys(counts));

serialize("simulations/"*"null"*"_"*parameter*".jls", (counts,labeled_counts,S));