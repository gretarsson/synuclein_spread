#=
here we create a folder of analysis plots of interence results
=#
using PathoSpread

# simulation to analyze
simulation = "DIFFGA_RETRO";

# read file 
inference_obj = load_inference("simulations/"*simulation*".jls")

# look at chains
#display(inference_obj["chain"][:,:,:])
#save_inference("simulations/" * simulation * ".jl", inference_obj)

# plot
setup_plot_theme!()  # set plotting settings
plot_inference(inference_obj,"figures/"*simulation)  

# plot with training data
#setup_plot_theme!()  # set plotting settings
#data_full, timepoints_full = PathoSpread.process_pathology("data/total_path.csv", W_csv="data/W_labeled_filtered.csv")
#plot_inference(inference_obj,"figures/"*simulation; full_data=data_full, full_timepoints=timepoints_full)  
#PathoSpread.plot_retrodiction(inference_obj; save_path="figures/"*simulation*"_retrodiction", N_samples=100)
