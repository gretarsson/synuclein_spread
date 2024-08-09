using Serialization
include("helpers.jl");
#=
Infer parameters of ODE using Bayesian framework
=#

# DIFFUSION, RETRO- AND ANTEROGRADE
thresholds = [0.15, 0.05, 0.01, 0.0];
#thresholds = [0.15];
Ns = Dict(0.15 => 40, 0.05 => 95, 0.01 => 174, 0.0 => 366);
for i in eachindex(thresholds)
    N = Ns[thresholds[i]]
    println("Inferring with N=$(N)")
    priors4 = OrderedDict( "σ" => InverseGamma(2,3), "ρa" => truncated(Normal(0,0.01), lower=0.), "ρr" => truncated(Normal(0,0.01), lower=0.), "γ" => truncated(Normal(0,1), lower=0.), "seed" => truncated(Normal(0.,1.),lower=0) );
    priors3 = OrderedDict( "σ" => InverseGamma(2,3), "ρ" => truncated(Normal(0,0.1), lower=0.), "γ" => truncated(Normal(0,1), lower=0.), "seed" => truncated(Normal(0.,1.),lower=0) );
    priors2 = OrderedDict( "σ" => InverseGamma(2,3), "ρₐ" => truncated(Normal(0,0.01), lower=0.), "ρᵣ" => truncated(Normal(0,0.01), lower=0.), "seed" => truncated(Normal(0.,1.),lower=0) );
    priors = OrderedDict( "σ" => LogNormal(0,1), "ρ" => truncated(Normal(0,0.01), lower=0.), "seed" => truncated(Normal(0.,0.1), lower=0) );
    u034 = [0. for _ in 1:(2*N)]
    pred_idxs34 = [N+i for i in 1:N]
    u0 = [0. for _ in 1:(1*N)]
    pred_idxs = [i for i in 1:N]

    inference = infer(diffusion2, 
                    priors2,
                    "data/avg_total_path.csv",
                    "data/timepoints.csv", 
                    "data/W_labeled.csv"; 
                    u0=u0,
                    pred_idxs=pred_idxs,
                    n_threads=4,
                    threshold=thresholds[i],
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
    serialize("simulations/total_diffusion2_N=$(N).jls", inference)
end
