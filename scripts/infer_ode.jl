using Serialization
include("helpers.jl");
#=
Infer parameters of ODE using Bayesian framework
=#
# DIFFUSION, RETRO- AND ANTEROGRADE
#priors = OrderedDict( "σ" => LogNormal(0,1), "ρ" => truncated(Normal(0,0.1), lower=0.), "seed" => truncated(Normal(1.,0.1),lower=0) );
priors2 = OrderedDict( "σ" => truncated(Normal(0,1), lower=0), "ρₐ" => truncated(Normal(0,0.1), lower=0.), "ρᵣ" => truncated(Normal(0,0.1), lower=0.), "seed" => truncated(Normal(1.,0.1),lower=0) );

inference = infer(diffusion2, 
                  priors2,
                  "data/avg_total_path.csv",
                  "data/timepoints.csv", 
                  "data/W_labeled.csv"; 
                  n_threads=1,
                  retro_and_antero=false,
                  threshold=0.15,
                  abstol=1e-6,
                  reltol=1e-3,
                  benchmark=false
                  )

# save inference result
serialize("simulations/total_diffusion2_N=40.jls", inference)
