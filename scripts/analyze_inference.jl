#=
here we create a folder of analysis plots of interence results
=#
include("helpers.jl");

# simulation to analyze
simulation = "total_aggregation_N=40_threads=1_var1_loglikelihood";

# plot 
inference_obj = deserialize("simulations/"*simulation*".jls")
inference_obj["data"] = exp.(inference_obj["data"])

plot_inference(inference_obj,"figures/"*simulation;plotscale=log10)  
