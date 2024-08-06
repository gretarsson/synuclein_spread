using Serialization
include("helpers.jl");
#=
Infer parameters of ODE using Bayesian framework
=#

# DIFFUSION, RETRO- AND ANTEROGRADE
# list of priors to include 
priors = OrderedDict( "σ" => LogNormal(0,1), "ρₐ" => truncated(Normal(0,0.1), lower=0.), "ρᵣ" => truncated(Normal(0,0.1), lower=0.), "seed" => truncated(Normal(1.,0.1),lower=0) );
inference = infer(diffusion2, 
                  priors,
                  "data/avg_total_path.csv",
                  "data/timepoints.csv", 
                  "data/W_labeled.csv"; 
                  n_threads=1,
                  retro_and_antero=true,
                  threshold=0.15,
                  abstol=1e-6,
                  reltol=1e-3,
                  benchmark=false
                  )
serialize("simulations/total_diffusion2_N=40.jls", inference)





#priors2 = [LogNormal(0,1),truncated(Normal(0,0.01), lower=0.), truncated(Normal(0,0.01), lower=0.), truncated(Normal(0.,0.25), lower=0.)]
# AGGREGATION
# priors for aggregation
#N = 3
#σ = LogNormal(0,1)
#ρ = truncated(Normal(0,0.01),lower=0.)
#α = truncated(Normal(0,2.5),lower=0.)
#β = [truncated(Normal(0.0,0.25), lower=0., upper=1.) for _ in 1:N]
#u0_seed = truncated(Normal(0.0,0.01),lower=0., upper=1.)
#priors = [σ, ρ, α, β..., u0_seed]
#priors_names = [σ, ρ, α, β..., u0_seed]
#priors = OrderedDict(priors)

#
#chain = infer(
#    aggregation, 
#    priors,
#    "data/avg_total_path.csv",
#    "data/timepoints.csv", 
#    "data/W_labeled.csv"; 
#    threshold=0.15,
#    sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)),
#    adtype = AutoReverseDiff(),
#    abstol=1e-6,
#    reltol=1e-3
#    )