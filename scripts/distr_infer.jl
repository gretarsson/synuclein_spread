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
ode = death_simplifiedii;
n_threads = 1;

# read data
timepoints = vec(readdlm("data/timepoints.csv", ','));
data = deserialize("data/total_path_3D.jls");

# DIFFUSION, RETRO- AND ANTEROGRADE
#N = length(idxs);
N = size(data)[1];
display("N = $(N)")
u0 = [0. for _ in 1:(2*N)];

# DEFINE PRIORS
priors = OrderedDict{Any,Any}( "ρ" => truncated(Normal(0,0.1),lower=0) ); 
priors["α"] = truncated(Normal(0,0.1),lower=0);
for i in 1:N
    priors["β[$(i)]"] = truncated(Normal(0,1),lower=0);
end
for i in 1:N
    priors["d[$(i)]"] = truncated(Normal(0,1),upper=0);
end
priors["γ"] = truncated(Normal(0,0.1),lower=0);
priors["σ"] = LogNormal(0,1);
priors["seed"] = truncated(Normal(0,0.1),lower=0);

# parameter refactorization
factors = [1., 1., [1 for _ in 1:N]..., [1 for _ in 1:N]..., 1.];  # death

# INFER
inference = infer(ode, 
                priors,
                data,
                timepoints, 
                "data/W_labeled.csv"; 
                factors=factors,
                u0=u0,
                n_threads=n_threads,
                bayesian_seed=true,
                seed_value=100,
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
serialize("simulations/total_$(ode)_N=$(N)_threads=$(n_threads)_var$(length(priors["σ"]))_truncated_decay.jls", inference)
Distributed.interrupt()  # kill workers from previous run (killing REPL does not do this)
