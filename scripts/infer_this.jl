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
    include("odes.jl")
end

# import ODEs
using .ODEs: odes


# -----------------------------------
#=
Infer parameters of ODE using Bayesian framework
=#
# PICK ODE
#ode = DIFFGAM_bilateral;
#ode = odes[ARGS[1]]
ode = "DIFFGAM"

n_threads = 1;

# flag if bilateral
bilateral = endswith(string(ode), "_bilateral")

# READ DATA
_, thr_idxs = read_data("data/avg_total_path.csv", remove_nans=true, threshold=0.15);
timepoints = vec(readdlm("data/timepoints.csv", ','));
data = deserialize("data/total_path_3D.jls")[thr_idxs,:,:];


# LOAD CONNECTOME AND MAKE LAPLACIAN
Lr,N,labels = read_W("data/W_labeled.csv", direction=:retro, idxs=thr_idxs );
La,_,_ = read_W("data/W_labeled.csv", direction=:antero);
Ltuple = (Lr,N)  # order is (L,N) or (Lr, La, N). The latter is used for bidirectional spread
display("N = $(N)")

# SET SEED AND INITIAL CONDITIONS
seed = findfirst(==("iCP"), labels);  

# SET PRIORS (variance and seed have to be last, in that order)
region_group = build_region_groups(labels)  # prepare bilateral parameters
K = bilateral ? maximum(region_group) : N
priors = get_priors(ode,K)
priors["σ"] = LogNormal(0,1);
priors["seed"] = truncated(Normal(0,0.1),lower=0);

# DEFINE ODE PROBLEM
factors = ones(length(get_priors(ode,K)))
u0 = [0. for _ in 1:(2*N)];  
prob = make_ode_problem(odes[ode];
    labels     = labels,
    Ltuple     = Ltuple,
    factors    = factors,
    u0         = u0,
    timepoints = timepoints,
)

# INFER
inference = infer(prob, 
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
                ode_name=ode,
                test_typestable=false
                )

# SAVE 
serialize("simulations/total_$(ode)_N=$(N)_threads=$(n_threads)_var$(length(priors["σ"]))_NEWRHS.jls", inference)
Distributed.interrupt()  # kill workers from previous run (killing REPL does not do this)
