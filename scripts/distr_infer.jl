#using Turing
using Distributed
addprocs(0)
# instantiate and precompile environment in all processes
@everywhere begin
    using Pkg; Pkg.activate(".")
    Pkg.instantiate(); Pkg.precompile()
end
@everywhere begin 
    using Turing, ParallelDataTransfer
    include("helpers.jl")
    include("model_priors.jl")
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

# LOAD CONNECTOME AND MAKE LAPLACIAN
Lr,N,labels = read_W("data/W_labeled.csv", direction=:retro);
La,_,_ = read_W("data/W_labeled.csv", direction=:antero);
Ltuple = (Lr,N)  # order is (L,N) or (Lr, La, N). The latter is used for bidirectional spread

# SET SEED AND INITIAL CONDITIONS
seed = findfirst(==("iCP"), labels);  
u0 = [0. for _ in 1:(2*N)];  

# SET PRIORS (variance and seed have to be last, in that order)
priors = get_priors(ode,N)
priors["σ"] = filldist(LogNormal(0,0.1),N);
priors["seed"] = truncated(Normal(0,0.1),lower=0);
σ

# INFER
inference = infer(ode, 
                priors,
                data,
                timepoints, 
                Ltuple; 
                u0=u0,
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
