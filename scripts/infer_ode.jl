using Serialization
include("helpers.jl");
#=
Infer parameters of ODE using Bayesian framework
=#
# read data
timepoints = vec(readdlm("data/timepoints.csv", ','));
data = deserialize("data/total_path_3D.jls");
#_, idxs = read_data("data/avg_total_path.csv", remove_nans=true, threshold=0.15);
#idxs = findall(idxs);


# DIFFUSION, RETRO- AND ANTEROGRADE
#N = length(idxs)
N = size(data)[1];
sol_idxs_death = [i for i in 1:N]
u0_death = [0. for _ in 1:(2*N)]

# INFORM PRIORS
# find maxima and end-point for each region for a sample, and remove that sample from data
sample_n = 4;  # sample to inform prior
maxima = Vector{Float64}(undef,N);
endpoints = Vector{Float64}(undef,N);
for region in axes(data,1)
    region_timeseries = data[region,:,sample_n]
    nonmissing = findall(region_timeseries .!== missing)
    region_timeseries = identity.(region_timeseries[nonmissing])
    if isempty(region_timeseries)
        maxima[region] = 0
        endpoints[region] = 0
    else
        maxima[region] = maximum(region_timeseries)
        if ismissing(region_timeseries[end])
            endpoints[region] = 0
        else
            endpoints[region] = region_timeseries[end]
        end
    end
end
#data = data[:,:,[1:3..., 5:end...]]  # TODO remove sample programmatically

# aggregation prior
priors_agg =OrderedDict{Any,Any}( "ρ" => truncated(Normal(0,1), lower=0), "ρᵣ" => truncated(Normal(1,0.25), lower=0), "α" => truncated(Normal(0,1),lower=0)); 
for i in 1:N
    priors_agg["β[$(i)]"] = truncated(Normal(0,1),lower=0)
    #priors_agg["β[$(i)]"] = truncated(Normal(maxima[i],0.05),lower=0)
end
for i in 1:N
    #priors_agg["d[$(i)]"] = truncated(Normal( maxima[i] - endpoints[i] , 0.05),lower=0)
    priors_agg["d[$(i)]"] = truncated(Normal( 0 , 1),lower=0)
end
priors_agg["γ"] = truncated(Normal(0,1),lower=0)
priors_agg["σ"] = InverseGamma(2,3)
priors_agg["seed"] = truncated(Normal(0,0.1),lower=0)

# parameter refactorization
factors_death = [1/100, 1., 1., [1 for _ in 1:N]..., [1. for _ in 1:N]..., 1.]

inference = infer(death_superlocal2, 
                priors_agg,
                data,
                timepoints, 
                "data/W_labeled.csv"; 
                factors=factors_death,
                u0=u0_death,
                #idxs=idxs,
                sol_idxs=sol_idxs_death,
                n_threads=1,
                bayesian_seed=true,
                seed_value=1.,
                transform_observable=true,
                alg=Tsit5(),
                abstol=1e-6,
                reltol=1e-3,
                adtype=AutoReverseDiff(),  # without compile much faster for aggregation
                sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)),
                benchmark=false,
                benchmark_ad=[:reversediff,:reversediff_compiled, :forwarddiff],
                test_typestable=false
                )

# save inference result
serialize("simulations/total_death_N=$(N).jls", inference)
