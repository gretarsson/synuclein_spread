#=
here we create a folder of analysis plots of interence results
=#
include("helpers.jl");

# simulation to analyze
simulation = "total_aggregation_N=448_threads=1_var1_CORRECT";

# plot 
inference_obj = deserialize("simulations/"*simulation*".jls")
inference_obj["data_indices"]

plot_inference(inference_obj,"figures/"*simulation;plotscale=log10)  
