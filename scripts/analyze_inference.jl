#=
here we create a folder of analysis plots of interence results
=#
include("helpers.jl");

# simulation to analyze
simulation = "total_death_N=40_threads=1_var1_constrain_seed";

# plot 
inference_obj = deserialize("simulations/"*simulation*".jls")
plot_inference(inference_obj,"figures/"*simulation;plotscale=log10)  
