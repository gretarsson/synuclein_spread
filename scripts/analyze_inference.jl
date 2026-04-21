#=
here we create a folder of analysis plots of interence results
=#
using PathoSpread
plot_priors_and_posteriors = true


simulations = ["igs_DIFF_EUCL", "igs_DIFF_ANTERO", "igs_DIFF_RETRO", "igs_DIFF_BIDIR",
               "DIFFG_EUCL", "DIFFG_ANTERO", "DIFFG_RETRO", "DIFFG_BIDIR",
               "DIFFGA_EUCL", "DIFFGA_ANTERO_CUT", "DIFFGA_RETRO", "DIFFGA_BIDIR",
]
simulations = ["DIFFG_global", "DIFFGA_global", "DIFFG_alpha_C2", "DIFFGA_alpha_C3"]
#simulations = ["u0_DIFF_RETRO", "u0_DIFFG_RETRO", "u0_DIFFGA_RETRO", "u0_DIFFGA_EUCL", "u0_DIFFGA_ANTERO", "u0_DIFFGA_BIDIR"]
simulations = ["u0s_DIFF_EUCL", "u0s_DIFF_ANTERO", "u0s_DIFF_BIDIR"]
simulations = ["u0percNoS_DIFFG_BIDIR", "u0percNoS_DIFFG_RETRO", "u0percNoS_DIFFG_EUCL"]
simulations = ["hippo_DIFFGA_RETRO_posterior_prior"]

#simulations = ["BILATERAL_DIFFGA_RETRO", "BILATERAL_DIFFGA_RETRO"]
#simulations = ["DIFFGA_T1", "DIFFGA_T2", "DIFFG_T1", "DIFFG_T2"]

#simulations = ["FLUSH_DIFF_RETRO", "FLUSH_DIFFG_RETRO", "FLUSH_DIFFGA_RETRO"]
#simulations = ["DIFFG_RETRO"]
#simulations = ["u0_DIFF_RETRO", "DIFFG_RETRO", "DIFFGA_RETRO"]
#simulations = ["new_hippo_DIFF_RETRO", "new_hippo_DIFFG_RETRO", "new_hippo_DIFFGA_RETRO"]
#simulations = ["global_hippo_DIFFGA_RETRO_posterior_prior"]
simulations = ["syn_mapt_DIFFGA_RETRO", "syn_app_DIFFGA_RETRO"]


#simulations = ["igs_DIFF_RETRO", "DIFFG_RETRO", "DIFFGA_RETRO"]
#simulations = ["DIFFGA_T1", "DIFFG_T1", "DIFFGA_T2", "DIFFG_T2"]

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
    #setup_plot_theme!()  # set plotting settings
    #display("Plotting inference results...")
    #plot_inference(inference_obj,"figures/inferences/"*simulation; plot_priors_posteriors=plot_priors_and_posteriors)  
    #display("Plots saved to figures/inferences/"*simulation)
    #display("---------------------------------------------------")

    # plot with training data
    setup_plot_theme!()  # set plotting settings
    data_full, timepoints_full = PathoSpread.process_pathology("data/total_path.csv", W_csv="data/W_labeled_filtered.csv")
    plot_inference(inference_obj,"figures/inferences_copathology/"*simulation; full_data=data_full, full_timepoints=timepoints_full, plot_priors_posteriors=false)  
    #
end