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
simulation = "simulations/total_death_N=448_ultimate.jls"
# read gene data
gene_data_full = readdlm("data/avg_Pangea_exp.csv",',');
gene_labels = gene_data_full[1,2:end];
gene_region_labels = identity.(gene_data_full[2:end,1])
gene_data = gene_data_full[2:end,2:end];  # region x gene
N_genes = size(gene_data)[2];

# find label indexing per the computational model / structural connectome
W_labels = readdlm("data/W_labeled.csv",',')[2:end,1];
W_label_map = dictionary_map(W_labels);

# Find estimate of posterior distributions 
model_par = "β[";
inference = deserialize(simulation);
chain = inference["chain"];
priors = inference["priors"];
model_par_idxs = [];
for i in 1:length(priors.keys)
    if occursin(model_par,priors.keys[i])
        append!(model_par_idxs,i)
    end
end
#model_par = "β[";
#inference = deserialize(simulation);
#chain = inference["chain"];
#priors = inference["priors"];
#model_par_idxs2 = [];
#for i in 1:length(priors.keys)
#    if occursin(model_par,priors.keys[i])
#        append!(model_par_idxs2,i)
#    end
#end


# find modes of each parameter
all_modes = [];
for i in eachindex(model_par_idxs)
    par_samples = vec(chain[:,model_par_idxs[i],:])
    posterior_i = KernelDensity.kde(par_samples)
    Plots.plot!(posterior_i)
    mode_i = posterior_i.x[argmax(posterior_i.density)]
    append!(all_modes,mode_i)
end
#all_modes2 = [];
#for i in eachindex(model_par_idxs2)
#    par_samples = vec(chain[:,model_par_idxs2[i],:])
#    posterior_i = KernelDensity.kde(par_samples)
#    Plots.plot!(posterior_i)
#    mode_i = posterior_i.x[argmax(posterior_i.density)]
#    append!(all_modes2,mode_i)
#end
#all_modes = all_modes ./ all_modes2

# create a map, indexing the regions in gene expression data and relating them to regions in the structural connectome
gene_to_struct = Dict(); 
for label in gene_region_labels
    sublabels = []
    for W_label in W_labels
        #if occursin(label,W_label)
        if lowercase(label) == lowercase(W_label[2:end]) || occursin(lowercase(label)*"-",lowercase(W_label))
            push!(sublabels,W_label)
        end
    end
    gene_to_struct[label] = sublabels
end
# find gene regions with no counterpart in connectome and display how many there are
regions_not_found = [];
for (keys,value) in gene_to_struct
    if isempty(value)
        push!(regions_not_found,keys)
    end
end
display("Warning: $(length(regions_not_found)) gene regions not found in connectome.")

# CORRELATION ANALYISIS
# extract gene expression vector for index i
    # find corresponding vector over regions for simulated pathology
para_vector = Vector{Union{Float64,Missing}}(missing,length(gene_region_labels))
for (k,gene_region) in enumerate(gene_region_labels)
    struct_regions = gene_to_struct[gene_region]
    model_pars = []
    if !isempty(struct_regions)
        for struct_region in struct_regions
            struct_index = W_label_map[struct_region]
            model_pars_i = all_modes[struct_index] 
            push!(model_pars, model_pars_i)
        end
        display(struct_regions)
        model_pars_subregion_avg = mean(model_pars)
        para_vector[k] = model_pars_subregion_avg
    end
end


# model_par_matrix now contains the optimal model parameters in same format as gene expression data
# as mentioned, some gene regions are missing from the connectome. These come up as rows of missing type in model_par_matrix
# find index of rows that are not missing
missing_rows = []
for non_region in regions_not_found
    non_region_index = findall(gene_region_labels .== non_region)[1]
    push!(missing_rows,non_region_index)
end
nonmissing = [i for i in 1:size(gene_data)[1]]
nonmissing = filter!(e->e∉missing_rows,nonmissing)

# find the parameter and gene matrix without missing regions
para_vector = identity.(para_vector[nonmissing])
gene_matrix = identity.(gene_data[nonmissing,:])

# do simple linear regression
lms = []
pvals = []
for gene_index in axes(gene_matrix,2)
    df_gene = DataFrame(X=gene_matrix[:,gene_index], Y=para_vector)
    ols = lm(@formula(Y ~ X),df_gene)
    coeff_p = coeftable(ols).cols[4][2]
    if coeff_p < 1e-5
        println("R^2: $(r2(ols)), p-value $(coeff_p), gene name: $(gene_labels[gene_index])")
    end
    push!(lms,ols)
    push!(pvals,coeff_p)
end
r2s = r2.(lms)
r2_inds = reverse(sortperm(r2s))
p_inds = sortperm(pvals)

# do multiple linear regression on subset of highly significant genes
top_gene_inds = 1 .+ p_inds[1:10]
gene_matrix_inter = hcat(ones(size(gene_matrix)[1]), gene_matrix)
ols = lm(gene_matrix_inter[:,[1;top_gene_inds...]],para_vector)
r2(ols)

gene_labels[p_inds[1:10]]
# TODO (1) clean this messy code up (2) should the data be preprossed before regression? Normalize by mean? 
# (3) what is a good fit?
