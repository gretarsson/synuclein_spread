#=
here we create a folder of analysis plots of interence results
=#
using PathoSpread


simulations = ["igs_DIFF_EUCL", "igs_DIFF_ANTERO", "igs_DIFF_RETRO", "igs_DIFF_BIDIR",
               "DIFFG_EUCL", "DIFFG_ANTERO", "DIFFG_RETRO", "DIFFG_BIDIR",
               "DIFFGA_EUCL", "DIFFGA_ANTERO_CUT", "DIFFGA_RETRO", "DIFFGA_BIDIR",
]
#simulations = ["DIFFGAM_RETRO"]


for simulation in simulations
    # simulation to analyze
    display("Plotting simulations: $simulation")

    # read file 
    inference_obj = load_inference("simulations/"*simulation*".jls")

    # look at chains
    display(inference_obj["chain"])
    #new_chain = inference_obj["chain"][:,:,[2,3,4]]
    #inference_obj["chain"] = new_chain
    #save_inference("simulations/" * simulation * "_CUT.jls", inference_obj)

    # plot
    setup_plot_theme!()  # set plotting settings
    display("Plotting inference results...")
    plot_inference(inference_obj,"figures/inferences/"*simulation)  
    display("Plots saved to figures/inferences/"*simulation)
    display("---------------------------------------------------")

    # plot with training data
    #setup_plot_theme!()  # set plotting settings
    #data_full, timepoints_full = PathoSpread.process_pathology("data/total_path.csv", W_csv="data/W_labeled_filtered.csv")
    #plot_inference(inference_obj,"figures/"*simulation; full_data=data_full, full_timepoints=timepoints_full)  
    #
end