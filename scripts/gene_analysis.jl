using DelimitedFiles
using KernelDensity
using Plots
using DataFrames, StatsBase, GLM, ProgressMeter
include("helpers.jl");
#=
Gene analysis!!
We draw from the posterior and compute significance with Holm-Bonferroni for a chosen parameter.
We save the number of occurences of each significant gene in a dictionary and save it. Also including the number of samples
=#
# pick simulation and parameter
simulation = "simulations/total_death_simplifiedii_N=448_threads=1_var1_normalpriors.jls"
parameter = "β"

# compute gene correlation with samples from the posterior
S = 1000
significants = [] 
@showprogress for s in 1:S
    r2s, pvals, significant, gene_labels = gene_analysis(simulation,parameter;mode=false,show=false);
    push!(significants,significant)
end

# create dictionary of how many times a gene was significant
counts = Dict{Int, Int}();
for vec in significants
    for num in vec
        counts[num] = get(counts, num, 0) + 1  # Increment the count
    end
end
labeled_counts = Dict(gene_labels[k] => counts[k] for k in keys(counts));

serialize("simulations/gene_significance.jls", (counts,labeled_counts,S));

# Sort by counts in descending order
sorted_labels = sort(collect(labeled_counts), by=x->x[2], rev=false);

# Display the sorted labels and their counts
println("Labels sorted by counts:")
for (label, count) in sorted_labels
    println("$label: $count")
end