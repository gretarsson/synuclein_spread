#=
here we create a folder of analysis plots of interence results
=#
using Serialization
include("helpers.jl");

# simulation to analyze
simulation = "total_death_N=448_threads=1_var1_sis_inspired_logpriors_truncatedlikelihood";

# plot 
inference_obj = deserialize("simulations/"*simulation*".jls")
#W = -inference_obj["L"][1]
#for i in 1:size(W)[1]
#    W[i,i] = 0
#end
#W = (W,size(W)[1])
#inference_obj["L"] = W

plot_inference(inference_obj,"figures/"*simulation;plotscale=log10)  
