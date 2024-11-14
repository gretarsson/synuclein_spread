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
end


# -----------------------------------
#=
Infer parameters of ODE using Bayesian framework
=#
# pick ode
ode = death;
n_threads = 1;

# read data
timepoints = vec(readdlm("data/timepoints.csv", ','));
data = deserialize("data/total_path_3D.jls");
data = Array(reshape(mean3(data),(size(data)[1],size(data)[2],1)));
_, idxs = read_data("data/avg_total_path.csv", remove_nans=true, threshold=0.15);
idxs = findall(idxs);


# DIFFUSION, RETRO- AND ANTEROGRADE
N = length(idxs);
#N = size(data)[1];
display("N = $(N)")
u0 = [0. for _ in 1:(2*N)];

# DEFINE PRIORS
priors = OrderedDict{Any,Any}( "ρ" => LogNormal(0,1) ); 
priors["α"] = LogNormal(0,1);
#priors = OrderedDict{Any,Any}( "ρ" => truncated(Normal(0,0.1),lower=0)); 
#priors["α"] = truncated(Normal(0,0.1),lower=0);
for i in 1:N
    priors["β[$(i)]"] = truncated(Normal(0,1),lower=0);
end
for i in 1:N
    priors["d[$(i)]"] = truncated(Normal(0,0.1),lower=0);
    #priors["d[$(i)]"] = LogNormal(0,1);
end
priors["σ"] = LogNormal(0,1);
priors["seed"] = truncated(Normal(0,0.01),lower=0);

# parameter refactorization
factors = [1., 1., [1 for _ in 1:N]..., [1 for _ in 1:N]...,];  # death
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
                bayesian_seed=true,
                seed_value=0.01,
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
serialize("simulations/total_$(ode)_N=$(N)_threads=$(n_threads)_var$(length(priors["σ"]))_sis_inspired_meandata_TNormal.jls", inference)
Distributed.interrupt()  # kill workers from previous run (killing REPL does not do this)
