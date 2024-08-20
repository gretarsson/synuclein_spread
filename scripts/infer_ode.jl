using Serialization
include("helpers.jl");
#=
Infer parameters of ODE using Bayesian framework
=#
# read data
timepoints = vec(readdlm("data/timepoints.csv", ','))::Vector{Float64};
data = deserialize("data/avg_total_path.jls")
idxs = [79:89...];


# DIFFUSION, RETRO- AND ANTEROGRADE
thresholds = [0.05];
Ns = Dict(0.15 => 40, 0.05 => 95, 0.01 => 174, 0.0 => 366);
for i in eachindex(thresholds)
    #N = Ns[thresholds[i]]
    N = length(idxs)
    println("Inferring with N=$(N)")
    #seed_m = 0.1*N
    #seed_v = 0.1*seed_m
    seed_m = 10
    seed_v = 2.5
    u0 = [0. for _ in 1:N]
    sol_idxs = [i for i in 1:N]
    sol_idxs_death = [i for i in 1:N]
    u0_death = [0. for _ in 1:(2*N)]

    # aggregation prior
    #priors_diff = OrderedDict( "σ" => InverseGamma(2,3), "ρ" => truncated(Normal(0,1), lower=0), "ρᵣ" => truncated(Normal(0,0.5), lower=0), "seed" => truncated(Normal(seed_m,seed_v),lower=0) );  
    priors_diff = OrderedDict(  "ρ" => truncated(Normal(0,1), lower=0), "ρᵣ" => truncated(Normal(1,0.25), lower=0), "σ" => InverseGamma(2,3), "seed" => truncated(Normal(seed_m,seed_v),lower=0));  
    #priors_diff = OrderedDict(  "ρ" => truncated(Normal(0,1), lower=0), "ρᵣ" => truncated(Normal(1.0,0.25), lower=0), "σ" => InverseGamma(2,3), "seed" => truncated(Normal(seed_m,seed_v),lower=0), "c" => truncated(Normal(0.0,1),lower=0));  
    #priors_diff4 = OrderedDict( "ρ" => truncated(Normal(0,1), lower=0), "ρᵣ" => truncated(Normal(1.,0.25), lower=0), "γ" => truncated(Normal(0,1),lower=0), "σ" => InverseGamma(2,3), "seed" => truncated(Normal(seed_m,seed_v),lower=0) );  
    priors_agg =OrderedDict{Any,Any}( "ρ" => truncated(Normal(0,1), lower=0), "ρᵣ" => truncated(Normal(1,0.25), lower=0), "α" => truncated(Normal(0,1),lower=0)); 
    #priors_agg =OrderedDict( "σ" => LogNormal(0,1), "ρ" => truncated(Normal(0,1), lower=0), "α" => truncated(Normal(0,1),lower=0)); 
    #for i in 1:N
    #    priors_agg["α[$(i)]"] = truncated(Normal(0,1),lower=0)
    #end
    for i in 1:N
        priors_agg["β[$(i)]"] = truncated(Normal(0,1),lower=0)
    end
    for i in 1:N
        priors_agg["d[$(i)]"] = truncated(Normal(0,1),lower=0)
    end
    #for i in 1:N
    #    priors_agg["γ[$(i)]"] = truncated(Normal(0,1),lower=0)
    #end
    priors_agg["γ"] = truncated(Normal(0,1),lower=0)
    priors_agg["σ"] = InverseGamma(2,3)
    priors_agg["seed"] = truncated(Normal(0,1),lower=0)

    # parameter refactorization
    #factors_agg = [1/100, 1., [10. for _ in 1:N]..., [1/10. for _ in 1:N]...]
    factors_death = [1/100, 1., 1., [1 for _ in 1:N]..., [1. for _ in 1:N]..., 1.]

    inference = infer(death_superlocal2, 
                    priors_agg,
                    data,
                    timepoints, 
                    "data/W_labeled.csv"; 
                    factors=factors_death,
                    u0=u0_death,
                    idxs=idxs,
                    sol_idxs=sol_idxs_death,
                    n_threads=1,
                    threshold=thresholds[i],
                    bayesian_seed=true,
                    seed_value=1.,
                    transform_observable=true,
                    alg=Tsit5(),
                    #alg=AutoTsit5(Rosenbrock23()),
                    abstol=1e-6,
                    reltol=1e-3,
                    adtype=AutoReverseDiff(),  # without compile much faster for aggregation
                    sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)),
                    #adtype=AutoForwardDiff(),
                    #sensealg=InterpolatingAdjoint(autojacvec=true),
                    benchmark=false,
                    benchmark_ad=[:reversediff,:reversediff_compiled, :forwarddiff],
                    test_typestable=false,
                    remove_nans=true
                    )

    # save inference result
    serialize("simulations/test_datadict.jls", inference)
end
