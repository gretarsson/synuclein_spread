#=
here we create a folder of analysis plots of interence results
=#
include("helpers.jl");


# simulation to analyze
simulation = "test";

# plot 
inference = deserialize("simulations/"*simulation*".jls")
plot_inference(inference,"figures/"*simulation;plotscale=log10)
