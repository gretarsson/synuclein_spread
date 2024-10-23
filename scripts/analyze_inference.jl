#=
here we create a folder of analysis plots of interence results
=#
include("helpers.jl");

# simulation to analyze
simulation = "total_death_N=40_threads=1_var1_binomial_mean_noseed";

# plot 
inference_obj = deserialize("simulations/"*simulation*".jls")
plot_inference(inference,"figures/"*simulation;plotscale=log10)  
