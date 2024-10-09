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
simulation = "simulations/total_death_N=448_threads=4_var1_normalpriors.jls"
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
model_par = "d[";
inference = deserialize(simulation);
chain = inference["chain"];
priors = inference["priors"];
model_par_idxs2 = [];
for i in 1:length(priors.keys)
    if occursin(model_par,priors.keys[i])
        append!(model_par_idxs2,i)
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
all_modes2 = [];
for i in eachindex(model_par_idxs2)
    par_samples = vec(chain[:,model_par_idxs2[i],:])
    posterior_i = KernelDensity.kde(par_samples)
    Plots.plot!(posterior_i)
    mode_i = posterior_i.x[argmax(posterior_i.density)]
    append!(all_modes2,mode_i)
end

# rename modes
modes_capacity = all_modes
modes_decay = all_modes2

# load data
timepoints = vec(readdlm("data/timepoints.csv", ','));
data = deserialize("data/total_path_3D.jls");
data = Array(reshape(mean3(data),(size(data)[1],size(data)[2],1)));
data = data[:,:,1]
N = size(data)[1]
zero_regions = []
for i in 1:N
    if maximum(skipmissing(data[i,:])) < 0.01
        push!(zero_regions,i)
    end
end

colors = ["blue" for _ in 1:N]
for index in zero_regions
    colors[index] = "red"
end

# the zero regions are the ones that seem to have random decay. 
# maybe a stricter prior on the decay will fix this, and just set the decay to zero.
Plots.scatter(all_modes,all_modes2; color=colors, alpha=0.7, ylabel="d", xlabel="\\beta", legend=false)  # TODO investigate correlations in optimal parameters
