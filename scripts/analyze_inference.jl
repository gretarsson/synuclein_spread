#=
here we create a folder of analysis plots of interence results
=#
using PathoSpread

# simulation to analyze
simulation = "DIFFGA_RETRO_T-1";

# read file 
inference_obj = load_inference("simulations/"*simulation*".jls")

# look at chains
#display(inference_obj["chain"][:,:,[1,2,3]])
#inference_obj["chain"] = inference_obj["chain"][:,:,[1,2,3]]
#save_inference("simulations/" * simulation * ".jl", inference_obj)

# get full data (needed for held out time points)
data_full, timepoints_full = PathoSpread.process_pathology("data/total_path.csv", W_csv="data/W_labeled_filtered.csv")

# plot
#setup_plot_theme!()  # set plotting settings
#plot_inference(inference_obj,"figures/"*simulation)  
#plot_inference(inference_obj,"figures/"*simulation; full_data=data_full, full_timepoints=timepoints_full)  


scores = PathoSpread.compute_heldout_scores(inference_obj;
    data_full = data_full,          # R×T_full or R×T_full×K
    timepoints_full = timepoints_full,
    S = 400
)
@show scores.elpd_mean, scores.crps_mean, scores.n_points