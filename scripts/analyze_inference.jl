using Serialization
using Distributions, Turing  # required to deserialize priors dictionary and MCMCChains object 
using MCMCChains
include("helpers.jl");

# load inference simulation 
simulation = "total_diffusion2_N=40";
inference = deserialize("simulations/"*simulation*".jls")

# plot pred. vs obsv., chains, posterior distributions, and retrodiction
save_path = "figures/"*simulation;
pred_obsv = predicted_observed(inference; save_path=save_path);
chain_figs = plot_chains(inference, save_path=save_path*"/chains");
prior_figs = plot_priors(inference; save_path=save_path*"/priors");
posterior_figs = plot_posteriors(inference, save_path=save_path*"/posteriors");
retrodiction_figs = plot_retrodiction(inference; save_path=save_path*"/retrodiction");
prior_and_posterior_figs = plot_prior_and_posterior(inference; save_path=save_path*"/prior_and_posterior");

inference["chain"]