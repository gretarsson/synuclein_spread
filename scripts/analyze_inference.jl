#=
here we create a folder of analysis plots of interence results
=#
include("helpers.jl");

# simulation to analyze
simulation = "total_death_simplifiedii_nodecay_N=448_threads=1_var1_olddecay_withx_notrunc";

# plot 
inference_obj = deserialize("simulations/"*simulation*".jls")

plot_inference(inference_obj,"figures/"*simulation;plotscale=log10)  
