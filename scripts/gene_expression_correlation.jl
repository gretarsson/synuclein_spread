using DelimitedFiles
using KernelDensity
using Plots
include("helpers.jl");
#=
Here we run linear regression to find correlations
between the posteriors of the inferred model parameters
to regional gene expression data
=#
simulation = "simulations/total_aggregation2_N=448_ultimate.jls"
# read gene data
gene_data_full = readdlm("data/avg_Pangea_exp.csv",',');
gene_labels = gene_data_full[1,2:end];
gene_region_labels = gene_data_full[2:end,1];
gene_data = gene_data_full[2:end,2:end];  # region x gene

# find label indexing per the computational model / structural connectome
W_labels = readdlm("data/W_labeled.csv",',')[2:end,1];
W_label_map = dictionary_map(W_labels);

# Find estimate of posterior distributions 
model_par = "β";
inference = deserialize(simulation);
chain = inference["chain"];
priors = inference["priors"];
model_par_idxs = [];
for i in 1:length(priors.keys)
    if occursin(model_par,priors.keys[i])
        append!(model_par_idxs,i)
    end
end

# find modes of each parameter
all_modes = [];
for i in eachindex(model_par_idxs)
    par_samples = vec(chain[:,model_par_idxs[i],:])
    posterior_i = KernelDensity.kde(par_samples)
    Plots.plot!(posterior_i)
    mode_i = posterior_i.x[argmax(posterior_i.density)]
    append!(all_modes,mode_i)
end

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
model_par_matrix = Array{Union{Float64,Missing}}(missing,size(gene_data))
# iterate through each gene and find optimal model parameters in each region
for gene_index in axes(gene_data,2)
    gene_expression = gene_data[:,gene_index]
    # find corresponding vector over regions for simulated pathology
    model_pars_gene_i = Vector{Union{Float64,Missing}}(missing,length(gene_region_labels))
    for (k,gene_region) in enumerate(gene_region_labels)
        struct_regions = gene_to_struct[gene_region]
        model_pars = []
        if !isempty(struct_regions)
            for struct_region in struct_regions
                struct_index = W_label_map[struct_region]
                model_pars_i = all_modes[struct_index] 
                push!(model_pars, model_pars_i)
            end
            model_pars_subregion_avg = mean(model_pars)
            model_pars_gene_i[k] = model_pars_subregion_avg
        end
    end
    # set optimal model parameter regional vector as column in model_par_matrix
    model_par_matrix[:,gene_index] .= model_pars_gene_i
end


# model_par_matrix now contains the optimal model parameters in same format as gene expression data
# as mentioned, some gene regions are missing from the connectome. These come up as rows of missing type in model_par_matrix
# We remove these regions from both gene_data and model_par_matrix








