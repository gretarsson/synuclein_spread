using PathoSpread

inference = load_inference("simulations/DIFFGA_RETRO.jls")
priors = posterior_to_priors(inference; widen=2.0)