using PathoSpread


# load inference object
inference = load_inference("simulations/DIFFGAM_RETRO.jls")

# save posterior modes
save_inference_MAP_csv(inference; path="simulations/optimal_parameters/posterior_mode_DIFFGAM.csv")
