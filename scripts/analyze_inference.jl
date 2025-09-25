#=
here we create a folder of analysis plots of interence results
=#
include("helpers.jl");
include("helpers_plots.jl");

# simulation to analyze
#simulation = "total_DIFFGAM_bilateral_N=40_threads=1_var1_NEWRHS";
simulation = "DIFFGA_RETRO";

# plot 
inference_obj = deserialize("simulations/"*simulation*".jl")
inference_obj
#display(inference_obj["chain"])
#inference_obj["chain"] = inference_obj["chain"][:,:,1]

plot_inference(inference_obj,"figures/"*simulation)  
