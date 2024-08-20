#=
here we create a folder of analysis plots of interence results
=#
include("helpers.jl");


# simulation to analyze
simulation = "total_death_N=95_datadict";
simulation2 = "total_death_N=95";


# plot 
inference = deserialize("simulations/"*simulation*".jls")
inference2 = deserialize("simulations/"*simulation2*".jls")
inference["data"] = inference2["data"]


plot_inference(inference,"figures/"*simulation;plotscale=log10)
