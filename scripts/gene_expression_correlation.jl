using DelimitedFiles
using KernelDensity
using Plots
using DataFrames, StatsBase, GLM
include("helpers.jl");
#=
Here we run linear regression to find correlations
between the posteriors of the inferred model parameters
to regional gene expression data
=#
simulation = "simulations/total_death_simplifiedii_N=448_threads=1_var1_normalpriors.jls"
# read gene data
gene_data_full = readdlm("data/avg_Pangea_exp.csv",',');
gene_labels = gene_data_full[1,2:end];
gene_region_labels = identity.(gene_data_full[2:end,1])
gene_data = gene_data_full[2:end,2:end];  # region x gene
N_genes = size(gene_data)[2];

# find label indexing per the computational model / structural connectome
W_labels = readdlm("data/W_labeled.csv",',')[2:end,1];
W_label_map = dictionary_map(W_labels);

# read the inference file, and find indices for beta and decay parameters 
inference = deserialize(simulation);
chain = inference["chain"];
priors = inference["priors"];
parameter_names = collect(keys(priors))
beta_idxs = findall(key -> occursin("β[",key), parameter_names)
deca_idxs = findall(key -> occursin("d[",key), parameter_names)

# find modes of parameters
mode = posterior_mode(chain)
beta_mode = mode[beta_idxs]
deca_mode = mode[deca_idxs]

# create a dictionary from gene labels to the connectome labels, and print out number of regions not found in connectome
gene_to_struct = submap(gene_region_labels,W_labels)
regions_not_found = [];
for (keys,value) in gene_to_struct
    if isempty(value)
        push!(regions_not_found,keys)
    end
end
display("Warning: $(length(regions_not_found)) gene regions not found in connectome.")

# CORRELATION ANALYISIS
# ---------------------------------------------------------------------------------------
# create vector with parameter values in same order as genes, and average over regions in gene_to_struct[region]
para_vector = create_parameter_vector_genes(beta_mode,gene_to_struct,gene_region_labels,W_label_map)

# regions that are not found are "missing", find regions that we do have
nonmissing = findall(e -> !ismissing(e), para_vector)
para_vector = identity.(para_vector[nonmissing])
gene_matrix = identity.(gene_data[nonmissing,:])

# do multiple linear regression over genes
lms,pvals = multiple_linear_regression(para_vector,gene_matrix;labels=gene_labels,alpha=0.05,show=false);

# Holm-Bonferroni correction (less conservative than Bonferroni)
r2s = r2.(lms);
r2_inds = reverse(sortperm(r2s));
p_inds = sortperm(pvals);
significant = []
for i in p_inds
    if pvals[i] < alpha / (N_genes - (i-1))
        println("R^2: $(r2s[i]), corr p-value $(pvals[i]), gene name: $(gene_labels[i])")
        #display(Plots.scatter(gene_matrix[:,i],para_vector;title="$(gene_labels[i])"))
        push!(significant,i)
    end
end
final = (r2s,pvals,significant)
# do multiple linear regression on subset of highly significant genes
#top_gene_inds = 1 .+ p_inds[1:10]
#gene_matrix_inter = hcat(ones(size(gene_matrix)[1]), gene_matrix)
#ols = lm(gene_matrix_inter[:,[1;top_gene_inds...]],para_vector)
#r2(ols)

# NOTES & QUESTIONS
# (2) should the data be preprossed before regression? Normalize by mean? 
# (3) what is a good fit? (4) plot the most significant correlations to see whether there might be nonlinear correlation
# (4) d - b gives really interesting correlations with VERY LOW p-values and high R^2 (0.35), col6a1, gsg1l in top two
# col6a1 is related to collagen synthesis and has been correalted with dopaminergic dysfunction(!!)
# gsg1l is realted to AMAP receptors at postsynaptic synapses!!
