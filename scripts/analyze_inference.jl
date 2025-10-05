#=
here we create a folder of analysis plots of interence results
=#
using PathoSpread

# simulation to analyze
simulation = "DIFFG_TEST";

# read file 
inference_obj = load_inference("simulations/"*simulation*".jls")

# look at chains
#display(inference_obj["chain"][:,:,[1,2,3]])
#inference_obj["chain"] = inference_obj["chain"][:,:,[1,2,3]]
#save_inference("simulations/" * simulation * ".jl", inference_obj)

# plot
setup_plot_theme!()  # set plotting settings
plot_inference(inference_obj,"figures/"*simulation)  

# plot with training data
#data_full, timepoints_full = PathoSpread.process_pathology("data/total_path.csv", W_csv="data/W_labeled_filtered.csv")
#plot_inference(inference_obj,"figures/"*simulation; full_data=data_full, full_timepoints=timepoints_full)  
