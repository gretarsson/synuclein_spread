#=
here we create a folder of analysis plots of interence results
=#
include("helpers.jl");


# simulation to analyze
simulation = "total_death_N=448_threads=4_var1_normalpriors";

# plot 
inference = deserialize("simulations/"*simulation*".jls")
plot_inference(inference,"figures/"*simulation;plotscale=log10)  
