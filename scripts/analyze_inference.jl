#=
here we create a folder of analysis plots of interence results
=#
include("helpers.jl");

# simulation to analyze
simulation = "total_death_N=40_threads=1_var1_sis_inspired_meandata_TNormal";

# plot 
inference_obj = deserialize("simulations/"*simulation*".jls")
#W = -inference_obj["L"][1]
#for i in 1:size(W)[1]
#    W[i,i] = 0
#end
#W = (W,size(W)[1])
#inference_obj["L"] = W


plot_inference(inference_obj,"figures/"*simulation;plotscale=log10,N_samples=100,show_variance=true)  
