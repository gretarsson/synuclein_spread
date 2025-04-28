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
# PICK ODE
ode = DIFFGAM;
n_threads = 1;

# READ DATA
timepoints = vec(readdlm("data/timepoints.csv", ','));
data = deserialize("data/total_path_3D.jls");
Lr,N,labels = read_W("data/W_labeled_filtered.csv", direction=:retro);
La,_,_ = read_W("data/W_labeled_filtered.csv", direction=:antero);
Ltuple = (Lr,N)  # order is (L,N) or (Lr, La, N)
seed = findfirst(==("iCP"), labels);

# SET PRIORS
K = N  # number of regional parameters
display("N = $(N)")
u0 = [0. for _ in 1:(2*N)];  # adaptation
#u0 = [0. for _ in 1:(N)];  # without adaptation

# DEFINE PRIORS
priors = OrderedDict{Any,Any}( "ρ" => truncated(Normal(0,0.1),lower=0) ); 
#priors = OrderedDict{Any,Any}( "ρr" => truncated(Normal(0,0.1),lower=0), "ρa" => truncated(Normal(0,0.1),lower=0) ); 
priors["α"] = truncated(Normal(0,0.1),lower=0);
for i in 1:K
    #priors["β[$(i)]"] = Normal(0,1);
    priors["β[$(i)]"] = truncated(Normal(0,1),lower=0);
end
for i in 1:K
    #priors["d[$(i)]"] = Normal(0,1);
    priors["d[$(i)]"] = truncated(Normal(0,1),lower=0);
end
priors["γ"] = truncated(Normal(0,1),lower=0)
priors["λ"] = Normal(0,1)
#priors["σ"] = LogNormal(0,1);
priors["σ"] = filldist(LogNormal(0,1),N);
priors["seed"] = truncated(Normal(0,0.1),lower=0);
#
# parameter refactorization
factors = [1., 1., [1 for _ in 1:K]..., [1 for _ in 1:K]..., 1., 1.];  # death


# INFER
inference = infer(ode, 
                priors,
                data,
                timepoints, 
                Ltuple; 
                factors=factors,
                u0=u0,
                #idxs=idxs,
                n_threads=n_threads,
                bayesian_seed=true,
                seed=seed,
                alg=Tsit5(),
                abstol=1e-6,
                reltol=1e-3,
                adtype=AutoReverseDiff(),  # without compile much faster for aggregation and death
                sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)),
                benchmark=false,
                benchmark_ad=[:reversediff],
                labels=labels,
                #M=M,
                test_typestable=false
                )

# SAVE 
serialize("simulations/total_$(ode)_N=$(N)_threads=$(n_threads)_var$(length(priors["σ"])).jls", inference)
Distributed.interrupt()  # kill workers from previous run (killing REPL does not do this)
