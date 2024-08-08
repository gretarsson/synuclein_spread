using Serialization
include("helpers.jl");
#=
Infer parameters of ODE using Bayesian framework
=#

# DIFFUSION, RETRO- AND ANTEROGRADE
thresholds = [0.05];
Ns = Dict(0.15 => 40, 0.05 => 95, 0.01 => 174, 0.0 => 366);
for i in eachindex(thresholds)
    println("Inferring N=$(Ns[thresholds[i]])")
    priors2 = OrderedDict( "σ" => LogNormal(0,1), "ρₐ" => truncated(Normal(0,0.01), lower=0.), "ρᵣ" => truncated(Normal(0,0.01), lower=0.), "seed" => truncated(Normal(1.,0.1),lower=0) );
    priors = OrderedDict( "σ" => LogNormal(0,1), "ρ" => truncated(Normal(0,0.01), lower=0.), "seed" => truncated(Normal(1.,0.1),lower=0) );
    u0 = [0. for _ in 1:Ns[thresholds[i]]]

    inference = infer(diffusion, 
                    priors,
                    "data/avg_total_path.csv",
                    "data/timepoints.csv", 
                    "data/W_labeled.csv"; 
                    u0=u0,
                    n_threads=1,
                    threshold=thresholds[i],
                    bayesian_seed=false,
                    seed_value=1.,
                    alg=Tsit5(),
                    abstol=1e-6,
                    reltol=1e-3,
                    #adtype=AutoReverseDiff(compile=true),
                    adtype=AutoForwardDiff(),
                    #sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)),
                    sensealg=InterpolatingAdjoint(autojacvec=true),
                    benchmark=false,
                    benchmark_ad=[:reversediff,:reversediff_compiled, :forwarddiff],
                    test_typestable=false
                    )

    # save inference result
    serialize("simulations/test.jls", inference)
end
