#=
here we create a folder of analysis plots of interence results
=#
using PathoSpread

# simulation to analyze
simulation = "DIFFGAM_BIDIR";

# read file 
inference_obj = load_inference("simulations/"*simulation*".jls")


# look at chains
inference_obj["chain"]
new_chain = inference_obj["chain"][:,:,[1,3]]
inference_obj["chain"] = new_chain
save_inference("simulations/" * simulation * "_CUT.jls", inference_obj)

# plot
setup_plot_theme!()  # set plotting settings
plot_inference(inference_obj,"figures/inferences/"*simulation)  

# plot with training data
#setup_plot_theme!()  # set plotting settings
#data_full, timepoints_full = PathoSpread.process_pathology("data/total_path.csv", W_csv="data/W_labeled_filtered.csv")
#plot_inference(inference_obj,"figures/"*simulation; full_data=data_full, full_timepoints=timepoints_full)  
#