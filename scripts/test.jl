#=
here we create a folder of analysis plots of interence results
=#
using PathoSpread, Statistics

simulation = "DIFFGA_BIDIR"
display("Plotting simulations: $simulation")

# read file 
inference_obj = load_inference("simulations/"*simulation*".jls")
inference_obj["chain"]

loglik = inference_obj["loglik_mat"]
mean(loglik, dims=1)

# look at chains
display(inference_obj["chain"])
new_chain = inference_obj["chain"][:,:,[1,2,3,4]]
inference_obj["chain"] = new_chain
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
