#=
here we create a folder of analysis plots of interence results
=#
using Serialization
include("helpers.jl");

# simulation to analyze
simulation = "total_death_N=448_threads=1_var1_truncated_normal";

# plot 
inference_obj = deserialize("simulations/"*simulation*".jls")
plot_inference(inference_obj,"figures/"*simulation;plotscale=log10)  
