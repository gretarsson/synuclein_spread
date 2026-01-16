#=
here we create a folder of analysis plots of interence results
=#
using PathoSpread, Statistics, DataFrames, CSV
gene_data = CSV.read("data/avg_Pangea_exp.csv", DataFrame)


simulation = "DIFFGA_RETRO"
display("Plotting simulations: $simulation")

# read file 
inference_obj = load_inference("simulations/"*simulation*".jls")
inference_obj["chain"]

L = inference_obj["L"][1]
W = -L
for i in axes(W,1)
    W[i,i] = 0
end

corrs = [0. for i in axes(W,1)]
for i in axes(W,1)
    corrs[i] = cor(W[74,:], W[i,:])
end
println(sortperm(corrs, rev=true))


loglik = inference_obj["loglik_mat"]
mean(loglik, dims=1)

# look at chains
display(inference_obj["chain"])
new_chain2 = inference_obj["chain"][:,:,[2]]
new_chain_new = chainscat(new_chain, new_chain2)



inference_obj["chain"] = new_chain_new
save_inference("simulations/" * simulation * ".jls", inference_obj)

# plot
setup_plot_theme!()  # set plotting settings
display("Plotting inference results...")
plot_inference(inference_obj,"figures/inferences/"*simulation, plot_priors_posteriors=true)  
display("Plots saved to figures/inferences/"*simulation)
display("---------------------------------------------------")

# plot with training data
#setup_plot_theme!()  # set plotting settings
#data_full, timepoints_full = PathoSpread.process_pathology("data/total_path.csv", W_csv="data/W_labeled_filtered.csv")
#plot_inference(inference_obj,"figures/"*simulation; full_data=data_full, full_timepoints=timepoints_full)  
#
