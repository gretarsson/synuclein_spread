using DelimitedFiles
using KernelDensity
using Plots
using DataFrames, StatsBase, GLM, Plots, Statistics, CSV
include("helpers.jl");
#=
Here we look at correlations in the posterior distribution between parameters
=#
folder_name = "death_simplifiedii"
simulation = "simulations/total_death_simplifiedii_N=448_threads=1_var1_normalpriors.jls"
# create directory to save figures in

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
variance = []
for i in eachindex(model_par_idxs)
    par_samples = vec(chain[:,model_par_idxs[i],:])
    posterior_i = KernelDensity.kde(par_samples)
    Plots.plot!(posterior_i)
    mode_i = posterior_i.x[argmax(posterior_i.density)]
    push!(variance,var(par_samples))
    append!(all_modes,mode_i)
end
all_modes2 = [];
variance2 = [];
for i in eachindex(model_par_idxs2)
    par_samples = vec(chain[:,model_par_idxs2[i],:])
    posterior_i = KernelDensity.kde(par_samples)
    Plots.plot!(posterior_i)
    mode_i = posterior_i.x[argmax(posterior_i.density)]
    push!(variance2,var(par_samples))
    append!(all_modes2,mode_i)
end

# rename modes
modes_capacity = all_modes
modes_decay = all_modes2
var_capacity = variance
var_decay = variance2

# put modes and capacity in a CSV
parameter_results = DataFrame(hcat(W_labels,modes_capacity,var_capacity,modes_decay,var_decay),["region","capacity mode","capacity variance","decay mode","decay variance"])
CSV.write("simulations/optimal_parameters_nonlinear_model.csv",parameter_results)

# plot the optimal parameters modes but coloured by variance
Plots.scatter(modes_capacity, modes_decay, marker_z=var_capacity, color=:viridis)
Plots.scatter(modes_decay, var_decay, xlims=(-2,2))