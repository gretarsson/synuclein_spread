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
gene_labels = readdlm("data/avg_Pangea_exp.csv",',')[1,2:end];  # names of genes
# pick simulation and parameter
simulation = "simulations/total_death_simplifiedii_N=448_threads=4_var1_normalpriors.jls";
parameter = "d";
file_name = "gene_significance";
S = 1000  # number of iterations

# Find significant genes in each iterate from posterior
significants = Vector{Any}(undef, S);
@showprogress Threads.@threads for s in 1:S
    # Perform gene analysis
    r2s, pvals, significant, gene_labels = gene_analysis(simulation, parameter; mode=false, show=false, null=false)
    significants[s] = significant
end

# create dictionary of how many times a gene was significant
counts = Dict{Int, Int}();
for vec in significants
    for num in vec
        counts[num] = get(counts, num, 0) + 1  # Increment the count
    end
end
labeled_counts = Dict(gene_labels[k] => counts[k] for k in keys(counts));

# compute significant genes with mode
_, _, mode_significant, _ = gene_analysis(simulation,parameter;mode=true,show=false);

# save
serialize("simulations/"*file_name*"_"*parameter*".jls", (counts,labeled_counts,S,mode_significant));
