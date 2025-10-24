using PathoSpread

inference = load_inference("simulations/DIFFGA_RETRO.jls")
priors = posterior_to_priors(inference; widen=2.0)

println(keys(priors))
old_priors = get_priors("DIFFGA",412)
