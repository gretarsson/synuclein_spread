#=
here we create a folder of analysis plots of interence results
=#
using PathoSpread

# simulation to analyze
simulation = "DIFFGAM_BIDIR";

# read file 
inference_obj = load_inference("simulations/"*simulation*".jls")

# look at chains
# 1,4
#display(inference_obj["chain"][:,:,[1,2,3,4]])
#save_inference("simulations/" * simulation * "_CUT.jl", inference_obj)

# plot
#setup_plot_theme!()  # set plotting settings
#plot_inference(inference_obj,"figures/"*simulation)  

# plot with training data
setup_plot_theme!()  # set plotting settings
data_full, timepoints_full = PathoSpread.process_pathology("data/total_path.csv", W_csv="data/W_labeled_filtered.csv")
data_full
mean_data = PathoSpread.mean3(data_full)
mean_data = Array(reshape(mean_data, size(mean_data,1), size(mean_data,2), 1))

plot_inference(inference_obj,"figures/"*simulation; full_data=data_full, full_timepoints=timepoints_full)  
