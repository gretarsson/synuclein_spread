#=
here we create a folder of analysis plots of interence results
=#
include("helpers.jl");


# simulation to analyze
simulation = "total_diffusion2_N=174_ratio";

# plot 
inference = deserialize("simulations/"*simulation*".jls")
plot_inference(inference,"figures/"*simulation)
