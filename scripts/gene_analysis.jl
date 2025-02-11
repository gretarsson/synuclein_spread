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
gene_labels = readdlm("data/avg_Pangea_exp.csv",',')
# pick simulation and parameter
simulation = "simulations/total_death_simplifiedii_N=448_threads=4_var1_normalpriors.jls";
parameter = "β";
file_name = "gene_significance";
S = 1000;  # number of iterations
null = false;

# Find significant genes in each iterate from posterior
rs = zeros(length(gene_labels),S);
pvals = zeros(length(gene_labels),S);
@showprogress Threads.@threads for s in 1:S
    # Perform gene analysis
    lm = gene_analysis(simulation, parameter; mode=false, show=false, null=null, save_plots=false)
    rs[:,s] = get_rvalue.(lm)  # a list of r values from iteration sample s
    pvals[:,s] = get_pvalue.(lm)  # a list of pvals (uncorrected) from sample s
end

# get dictionaries for r- and p-values for gene labels
labeled_rs = Dict(gene_labels[k] => rs[k,:] for k in 1:size(gene_labels)[1]);
labeled_pvals = Dict(gene_labels[k] => pvals[k,:] for k in 1:size(gene_labels)[1]);

# do significance tests (Holm-Bonferroni)
significants = [];  # a list of lists with significant genes
for s in 1:S
    significant = holm_bonferroni(pvals[:,s])
    push!(significants,significant)
end
    
# create dictionary of how many times a gene was significant
counts = Dict{Int, Int}();
for vec in significants
    for num in vec
        counts[num] = get(counts, num, 0) + 1  # Increment the count
    end
end
labeled_counts = Dict(gene_labels[k] => 0 for k in 1:size(gene_labels)[1]);
for k in keys(counts)
    labeled_counts[gene_labels[k]] = counts[k]
end

# save
serialize("simulations/"*file_name*"_"*parameter*".jls", (labeled_counts,labeled_rs,labeled_pvals,S));
