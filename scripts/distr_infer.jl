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
ode = death;
n_threads = 1;

# read data
timepoints = vec(readdlm("data/timepoints.csv", ','));
data = deserialize("data/total_path_3D.jls");
data = data ./ 2
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
data = Array(reshape(mean3(data),(size(data)[1],size(data)[2],1)));
_, idxs = read_data("data/avg_total_path.csv", remove_nans=true, threshold=0.15);
idxs = findall(idxs);


# DIFFUSION, RETRO- AND ANTEROGRADE
N = length(idxs);
#N = size(data)[1];
display("N = $(N)")
u0 = [0. for _ in 1:(2*N)];

# INFORM PRIORS
#data2, maxima2, endpoints2 = inform_priors(data,4)

# DEFINE PRIORS
#priors = OrderedDict{Any,Any}( "ρ" => LogNormal(0,1), "ρᵣ" =>  LogNormal(0,1)); 
#priors = OrderedDict{Any,Any}( "ρ" => LogNormal(0,1) ); 
#priors = OrderedDict{Any,Any}( "ρ" => LogNormal(0,1) ); 
#priors["α"] = LogNormal(0,1);
priors = OrderedDict{Any,Any}( "ρ" => truncated(Normal(0,0.1),lower=0)); 
priors["α"] = truncated(Normal(0,0.1),lower=0);
for i in 1:N
    priors["β[$(i)]"] = truncated(Normal(0,1),lower=0);
end
for i in 1:N
    priors["d[$(i)]"] = truncated(Normal(0,1), lower=0);
end
priors["γ"] = truncated(Normal(0,0.1),lower=0);
#priors["γ"] = LogNormal(0,1);
priors["σ"] = LogNormal(0,1);
#priors["σ"] = truncated(Normal(0,0.01),lower=0);  # regional variance
#priors["σ"] = filldist(LogNormal(0,1),N); 
#priors["σ"] = filldist(InverseGamma(2,3),N); # global variance
#priors["σ"] = InverseGamma(2,3); # global variance
#priors["σ"] = truncated(Normal(0,0.01),lower=0); # global variance
priors["seed"] = truncated(Normal(0,0.1),lower=0);
#priors["seed"] = Uniform(0,0.1);
#priors["seed"] = LogNormal(0,1);
# diffusion seed prior
#seed_m = round(0.05*N,digits=2)
#seed_v = round(0.1*seed_m,digits=2)
#priors["seed"] = truncated(Normal(seed_m,seed_v),0,Inf)
#
# parameter refactorization
factors = [1., 1., [1 for _ in 1:N]..., [1 for _ in 1:N]..., 1.];  # death
#factors = [1., 1., [1 for _ in 1:N]...];  # aggregation
#factors = [1.]  # diffusion


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
                bayesian_seed=false,
                seed_value=0.01,
                transform_observable=true,
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
serialize("simulations/total_$(ode)_N=$(N)_threads=$(n_threads)_var$(length(priors["σ"]))_poisson_normal.jls", inference)
Distributed.interrupt()  # kill workers from previous run (killing REPL does not do this)
