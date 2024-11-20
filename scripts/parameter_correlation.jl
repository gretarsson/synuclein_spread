using DelimitedFiles
using KernelDensity
using Plots
using DataFrames, StatsBase, GLM, Plots, Statistics
include("helpers.jl");
#=
Here we look at correlations in the posterior distribution between parameters
=#
folder_name = "death_simplifiedii"
simulation = "simulations/total_death_simplifiedii_N=448_threads=1_var1_normalpriors.jls"

# create directory to save figures in
save_path = "figures/posterior_correlation/"*folder_name
try
    mkdir(save_path) 
catch
end
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
p = Plots.scatter(all_modes,all_modes2; color=colors, alpha=0.7, ylabel="d", xlabel="\\beta", legend=false);
Plots.savefig(p, "figures/posterior_correlation/"*folder_name*"/all_regions_modes.png")

# okay that is cool what about samplings from the posterior and looking at correlations therein
# find indices of beta and decay parameters in chain
parameter_names = collect(keys(priors))
beta_idxs = findall(key -> occursin("β[",key), parameter_names)
deca_idxs = findall(key -> occursin("d[",key), parameter_names)

# sample from the posterior
S = 1000;
posterior_samples = sample(chain, S; replace=false);

# look at beta and decay samples, and store them in arrays (S x regions)
betas = Array(posterior_samples[:,beta_idxs,1])
decas = Array(posterior_samples[:,deca_idxs,1])
for i in 1:N
    samples1 = betas[:,i]
    samples2 = decas[:,i]

    # Calculate Pearson correlation
    pearson_corr = cor(samples1, samples2)

    # Prepare data for linear regression
    X = hcat(ones(length(samples1)), samples1)  # Add a column of ones for the intercept
    y = samples2

    # Perform linear regression
    model = lm(X, y)

    # Get slope and intercept from the model
    intercept = coef(model)[1]
    slope = coef(model)[2]

    # Generate points for the line
    x_fit = range(minimum(samples1), maximum(samples1), length=100)  # x-values for line
    y_fit = intercept .+ slope .* x_fit  # Corresponding y-values

    # Plot the ordered pairs with your custom labels
    p = Plots.scatter(samples1, samples2, 
        xlabel="β", 
        ylabel="d", 
        title="Region $(i)",
        label=nothing,
        legend=true);

    # Add the line of best fit
    maxi = max(maximum(samples1),maximum(samples2))
    mini = min(minimum(samples1),minimum(samples2))
    Plots.plot!(x_fit, y_fit, 
                label="Pearson r = $(round(pearson_corr, digits=2)), slope = $(round(slope,digits=2)), intercept = $(round(intercept,digits=2))", 
                color=:red, xlims=(mini,maxi), ylims=(mini,maxi));

    # Save the plot
    Plots.savefig(p,"figures/posterior_correlation/" * folder_name * "/beta_decay_region_$(i).png")
end
