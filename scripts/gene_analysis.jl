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
simulation = "simulations/total_death_simplifiedii_N=448_threads=_var1_normalpriors.jls";
parameter = "d";
file_name = "null";
S = 10000;  # number of iterations
null = true;

# Find significant genes in each iterate from posterior
significants = Vector{Any}(undef, S);
lms = Vector{Any}(undef, S);
pvals = Vector{Any}(undef, S);
@showprogress Threads.@threads for s in 1:S
    # Perform gene analysis
    lm, pval, significant, gene_labels = gene_analysis(simulation, parameter; mode=false, show=false, null=null)
    significants[s] = significant
    lms[s] = lm
    pvals[s] = pval
end

r = get_rvalue.(lms[1])
significant = significants[1]

rs = Dict{Int,Vector{Float64}}()
pvalss = Dict{Int,Vector{Float64}}() 
for i in 1:S
    r = get_rvalue.(lms[i])  # a list of r values from iteration i
    significant = significants[i]  # a list of regions significant at iteration i
    for (k,region) in enumerate(significant)
        if haskey(rs,region)
            push!(rs[region],r[k]) 
        else
            rs[region] = [r[k]]
        end
        if haskey(pvalss,region)
            push!(pvalss[region],pvals[i][region]) 
        else
            pvalss[region] = [pvals[i][region]]
        end
    end
end
labeled_rs = Dict(gene_labels[k] => rs[k] for k in keys(rs))
labeled_pvals = Dict(gene_labels[k] => pvalss[k] for k in keys(pvalss))
for (key,item) in labeled_rs  # check if any gene has positive AND negative r values
    if abs(sum(sign.(item))) != length(item)
        println(labeled_rs[key])
    end
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
serialize("simulations/"*file_name*"_"*parameter*".jls", (counts,labeled_counts,rs,labeled_rs, pvalss, labeled_pvals, S,mode_significant,significants));
