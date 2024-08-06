using Serialization
using Distributions, Turing  # required to deserialize priors dictionary and MCMCChains object 

# load inference simulation 
inference = deserialize("simulations/total_diffusion2_N=40.jls")

# plot average vs predicted
