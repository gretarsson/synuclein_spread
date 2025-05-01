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
_, thr_idxs = read_data("data/avg_total_path.csv", remove_nans=true, threshold=0.15);
timepoints = vec(readdlm("data/timepoints.csv", ','));
data = deserialize("data/total_path_3D.jls")[thr_idxs,:,:];


# LOAD CONNECTOME AND MAKE LAPLACIAN
Lr,N,labels = read_W("data/W_labeled.csv", direction=:retro, idxs=thr_idxs );
La,_,_ = read_W("data/W_labeled.csv", direction=:antero);
#Lr = random_laplacian(N)
#Lr = Lr ./ maximum(Lr[Lr .!= 0])
Ltuple = (Lr,N)  # order is (L,N) or (Lr, La, N). The latter is used for bidirectional spread
display("N = $(N)")

# SET SEED AND INITIAL CONDITIONS
seed = findfirst(==("iCP"), labels);  
#seed = 1;
u0 = [0. for _ in 1:(2*N)];  

# SET PRIORS (variance and seed have to be last, in that order)
priors = get_priors(ode,N)
priors["σ"] = LogNormal(0,1);
priors["seed"] = truncated(Normal(0,0.1),lower=0);

region_groups = build_region_groups(labels)

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
serialize("simulations/total_$(ode)_N=$(N)_threads=$(n_threads)_var$(length(priors["σ"]))_NEWRHS.jls", inference)
Distributed.interrupt()  # kill workers from previous run (killing REPL does not do this)
