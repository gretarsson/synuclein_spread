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
gene_data_full = readdlm("data/avg_Pangea_exp.csv",',');
gene_labels = gene_data_full[1,2:end];
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

serialize("simulations/gene_significance_d.jls", (counts,labeled_counts,S));



# ANALYZE RESULTS
using Serialization
counts, labeled_counts, S = deserialize("simulations/gene_significance_β.jls")  
gene_data_full = readdlm("data/avg_Pangea_exp.csv",',');
gene_labels = gene_data_full[1,2:end];

# Ensure all integers are represented in counts (fill missing with 0)
all_labeled_counts = Dict(label => get(labeled_counts, label, 0) for label in gene_labels)
# Extract the counts
counts = collect(values(all_labeled_counts))

# Automatically determine number of bins (square root rule)
num_bins = ceil(Int, sqrt(length(counts)))

# Create the histogram
hist = histogram(counts, bins=num_bins, 
    xlabel="Significance counts (S=$(S))", 
    ylabel="Frequency", 
    title="Number of times gene is found significant", 
    yscale=:log10,
    legend=false)

# Display the sorted labels and their counts
sorted_labels = sort(collect(labeled_counts), by=x->x[2], rev=false);
println("Labels sorted by counts:")
for (label, count) in sorted_labels
    println("$label: $count")
end
display("Total number of significant genes: $(length(sorted_labels))")

# compute significant genes with mode
r2s, pvals, significant, gene_labels = gene_analysis(simulation,parameter;mode=true,show=false);
display("Significant genes for mode of posterior")
for signi in significant
    display(gene_labels[signi])
end

# Show the plot
display(current())
#Plots.savefig(hist,"figures/gene_analysis/histogram_d.pdf")