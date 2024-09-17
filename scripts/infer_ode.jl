using Serialization
include("helpers.jl");
#=
Infer parameters of ODE using Bayesian framework
NOTE:
the diffusion-only model does not do well when
trained on all the data. It does much better when trained
on t=1,3,6,9 skipping the first four timepoints.
=#
# pick ode
ode = death2;
n_threads = 1;

# read data
timepoints = vec(readdlm("data/timepoints.csv", ','));
data = deserialize("data/total_path_3D.jls");
#data = data[:,5:end,:];
#timepoints = timepoints[5:end];
#data = Array(reshape(mean3(data),(size(data)[1],size(data)[2],1)));
#_, idxs = read_data("data/avg_total_path.csv", remove_nans=true, threshold=0.15);
#idxs = findall(idxs);


# DIFFUSION, RETRO- AND ANTEROGRADE
#N = length(idxs);
N = size(data)[1];
u0 = [0. for _ in 1:2*N];

# INFORM PRIORS
#data2, maxima2, endpoints2 = inform_priors(data,4)

# DEFINE PRIORS
priors =OrderedDict{Any,Any}( "ρ" => LogNormal(0,1), "ρᵣ" =>  LogNormal(0,1)); 
priors["α"] = LogNormal(0,1);
for i in 1:N
    priors["β[$(i)]"] = truncated(Normal(0,1), 0, Inf);
end
for i in 1:N
    priors["d[$(i)]"] = truncated(Normal(0,1), 0, Inf);
end
priors["γ"] = LogNormal(0,1);
#priors["σ"] = filldist(LogNormal(0,1),N);  # regional variance
priors["σ"] = LogNormal(0,1);  # global variance
priors["seed"] = truncated(Normal(0,0.1), 0, Inf);
# diffusion seed prior
#seed_m = round(0.05*N,digits=2)
#seed_v = round(0.1*seed_m,digits=2)
#priors["seed"] = truncated(Normal(seed_m,seed_v),lower=0)

# parameter refactorization
factors = [1, 1., 1., [1 for _ in 1:N]..., [1 for _ in 1:N]..., 1];

# INFER
inference = infer(ode, 
                priors,
                data,
                timepoints, 
                "data/W_labeled.csv"; 
                factors=factors,
                u0=u0,
                #idxs=idxs,
                n_threads=n_threads,
                bayesian_seed=true,
                seed_value=0.01,
                transform_observable=true,
                alg=Tsit5(),
                abstol=1e-6,
                reltol=1e-3,
                adtype=AutoReverseDiff(),  # without compile much faster for aggregation and death
                sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)),
                benchmark=true,
                benchmark_ad=[:reversediff],
                test_typestable=false
                )

# SAVE
serialize("simulations/total_$(ode)_N=$(N)_threads=$(n_threads).jls", inference)
