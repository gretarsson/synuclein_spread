using Turing
using Distributed
addprocs(0)

# instantiate and precompile environment in all processes
@everywhere begin
    using Pkg; Pkg.activate(".")
    Pkg.instantiate(); Pkg.precompile()
end

@everywhere begin 
using Turing, ParallelDataTransfer
end

@everywhere begin
include("helpers.jl")
#include("tanhnormal.jl")
end


# -----------------------------------
#=
Infer parameters of ODE using Bayesian framework
NOTE:
the diffusion-only model does not do well when
trained on all the data. It does much better when trained
on t=1,3,6,9 skipping the first four timepoints.
=#
# pick ode
include("tanhnormal.jl")
ode = sir;
n_threads = 1;

# read data
timepoints = vec(readdlm("data/timepoints.csv", ','));
data = deserialize("data/total_path_3D.jls");
#data = data ./ maximum(skipmissing(data)) 
#for i in eachindex(data)
#    if !ismissing(data[i]) && data[i] == 0
#        data[i] = missing
#    end
#end
#data = atanh.(data)
#data = data .+ 0.01
#data = data ./ (maximum(skipmissing(data))+0.1)
#data = data[:,5:end,:];
#timepoints = timepoints[5:end];
#data = Array(reshape(mean3(data),(size(data)[1],size(data)[2],1)));
_, idxs = read_data("data/avg_total_path.csv", remove_nans=true, threshold=0.15);
idxs = findall(idxs);


# DIFFUSION, RETRO- AND ANTEROGRADE
N = length(idxs);
#N = size(data)[1];
display("N = $(N)")
u0 = [0. for _ in 1:(2*N)];

# DEFINE PRIORS
priors = OrderedDict{Any,Any}( )
for i in 1:N
    priors["τ[$(i)]"] = truncated(Normal(0,1),lower=0);
end
for i in 1:N
    priors["γ[$(i)]"] = truncated(Normal(0,1),lower=0);
end
for i in 1:N
    priors["θ[$(i)]"] = truncated(Normal(0,1),lower=0);
end
priors["ϵ"] = truncated(Normal(0,1),lower=0);
priors["σ"] = LogNormal(0,1);
priors["seed"] = truncated(Normal(0,0.1),lower=0,upper=1);
#
# parameter refactorization
#factors = [[1. for _ in 1:N]..., [1 for _ in 1:N]...,[1 for _ in 1:N]..., 1]
factors = [[1. for _ in 1:N]..., [1 for _ in 1:N]..., [1 for _ in 1:N]..., 1]


# INFER
inference = infer(ode, 
                priors,
                data,
                timepoints, 
                "data/W_labeled.csv"; 
                factors=factors,
                u0=u0,
                idxs=idxs,
                n_threads=n_threads,
                bayesian_seed=true,
                seed_value=0.01,
                transform_observable=false,
                alg=Tsit5(),
                abstol=1e-6,
                reltol=1e-3,
                adtype=AutoReverseDiff(),  # without compile much faster for aggregation and death
                sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)),
                benchmark=false,
                benchmark_ad=[:reversediff],
                test_typestable=false
                )

# SAVE 
serialize("simulations/total_$(ode)_N=$(N)_threads=$(n_threads)_var$(length(priors["σ"]))_local_infection_recovery_death_transpose.jls", inference)
Distributed.interrupt()  # kill workers from previous run (killing REPL does not do this)
