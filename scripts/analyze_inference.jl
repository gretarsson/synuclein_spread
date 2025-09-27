#=
here we create a folder of analysis plots of interence results
=#
include("helpers.jl");
include("helpers_plots.jl");

# simulation to analyze
simulation = "DIFFGA_RETRO";

# read file 
inference_obj = deserialize("simulations/"*simulation*".jl")

# look at chains
#display(inference_obj["chain"][:,:,[1,2,3,4]])
#inference_obj["chain"] = inference_obj["chain"][:,:,[1,3,4]]
#serialize("simulations/" * simulation * ".jl", inference_obj)

# plot
plot_inference(inference_obj,"figures/"*simulation)  
