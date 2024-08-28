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
gene_data_full = readdlm("data/avg_Pangea_exp.csv",',')
gene_labels = gene_data_full[1,2:end]
region_labels = gene_data_full[2:end,1]
gene_data = gene_data_full[2:end,2:end]  # region x gene

# find label indexing per the computational model / structural connectome
W_labels = readdlm("data/W_labeled.csv",',')[2:end,1]
label_map = dictionary_map(W_labels)

# Find estimate of posterior distributions 
model_par = "β"
inference = deserialize(simulation)
chain = inference["chain"]
priors = inference["priors"]
model_par_idxs = []
priors.keys
for i in 1:length(priors.keys)
    if occursin(model_par,priors.keys[i])
        append!(model_par_idxs,i)
    end
end

# find modes of each parameter
all_modes = []
for i in eachindex(model_par_idxs)
    par_samples = vec(chain[:,model_par_idxs[i],:])
    posterior_i = KernelDensity.kde(par_samples)
    Plots.plot!(posterior_i)
    mode_i = posterior_i.x[argmax(posterior_i.density)]
    append!(all_modes,mode_i)
end

# find the modes of the regions with gene expression data and store them in same order
for label in region_labels
    for W_label in W_labels
        if occursin(label,W_label)
            display(W_label)
        end
    end
end
# TODO for each gene expression region, there is multiple subregions(?) in the comp. model region.
# find out what this means