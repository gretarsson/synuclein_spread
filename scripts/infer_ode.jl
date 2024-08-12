using Serialization
include("helpers.jl");
#=
Infer parameters of ODE using Bayesian framework
=#

# DIFFUSION, RETRO- AND ANTEROGRADE
#thresholds = [0.15, 0.05, 0.01, 0.0];
thresholds = [0.15, 0.05, 0.01, 0.0];
#thresholds = [0.0];
Ns = Dict(0.15 => 40, 0.05 => 95, 0.01 => 174, 0.0 => 366);
for i in eachindex(thresholds)
    N = Ns[thresholds[i]]
    println("Inferring with N=$(N)")
    seed_m = 0.1*N
    seed_v = 0.1*seed_m
    priors2 = OrderedDict( "σ" => InverseGamma(2,3), "ρ" => truncated(Normal(0,1), lower=0), "ρᵣ" => truncated(Normal(1,0.1), lower=0), "seed" => truncated(Normal(seed_m,seed_v),lower=0) );  
    u0 = [0. for _ in 1:N]
    pred_idxs = [i for i in 1:N]

    # aggregation prior
    priors_agg2 =OrderedDict( "σ" => InverseGamma(2,3), "ρ" => truncated(Normal(0,1), lower=0), "ρᵣ" => truncated(Normal(1,0.1), lower=0), "α" => truncated(Normal(0,1),lower=0)); 
    for i in 1:N
        priors_agg2["β[$(i)]"] = truncated(Normal(0,1),lower=0)
    end
    priors_agg2["seed"] = truncated(Normal(0,1),lower=0)

    inference = infer(aggregation2, 
                    priors_agg2,
                    "data/avg_total_path.csv",
                    "data/timepoints.csv", 
                    "data/W_labeled.csv"; 
                    u0=u0,
                    pred_idxs=pred_idxs,
                    n_threads=4,
                    threshold=thresholds[i],
                    W_factor=100.,
                    bayesian_seed=true,
                    seed_value=1.,
                    transform_observable=true,
                    alg=Tsit5(),
                    abstol=1e-6,
                    reltol=1e-3,
                    adtype=AutoReverseDiff(compile=true),
                    sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)),
                    #adtype=AutoForwardDiff(),
                    #sensealg=InterpolatingAdjoint(autojacvec=true),
                    benchmark=false,
                    benchmark_ad=[:reversediff,:reversediff_compiled, :forwarddiff],
                    test_typestable=false
                    )

    # save inference result
    serialize("simulations/total_aggregation2_N=$(N)_ratio.jls", inference)
end
