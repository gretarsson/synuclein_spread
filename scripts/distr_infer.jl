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
ode = fastslow;
n_threads = 1;

# read data
timepoints = vec(readdlm("data/timepoints.csv", ','));
data = deserialize("data/total_path_3D.jls");
#data = data[:,1:(end-3),:] 
#timepoints = timepoints[1:(end-3)]
_, thr_idxs = read_data("data/avg_total_path.csv", remove_nans=true, threshold=0.15);
idxs = findall(thr_idxs);


# get bilateral idxs
#W_file = "data/W_labeled.csv"
#W_labelled = readdlm(W_file,',')
#labels = W_labelled[1,2:end]
#bi_idxs = only_bilateral(labels)
#N = length(labels)
#M = Int(length(bi_idxs) / 2)
#nobi_idxs = setdiff(1:N, bi_idxs)  # Indices of regions without twins
#idxs = vcat(bi_idxs, nobi_idxs)

# FEWER VARIABLES WITH BILATERAL PARAMETERS
#idxs = thresholded_bilateral_idxs(thr_idxs,bi_idxs)
#M = Int(length(idxs)/2)
#for i in 1:M
#    i = Int(i)
#    display(labels[new_idxs[i]])
#    display(labels[new_idxs[i+40]])
#end
#nobi_idxs = []


# DIFFUSION, RETRO- AND ANTEROGRADE
#N = size(data)[1];
N = length(idxs)  # when using test subset of data
#K = M + length(nobi_idxs)  # number of unique regional parameters
K = N
#M = N  # without bilateral
display("N = $(N)")
u0 = [0. for _ in 1:(2*N)];

# DEFINE PRIORS
priors = OrderedDict{Any,Any}( "ρ" => truncated(Normal(0,0.1),lower=0) ); 
priors["α"] = truncated(Normal(0,0.1),lower=0);
#for i in 1:M
#    priors["α[$(i)]"] = truncated(Normal(0,1),lower=0);
#end
for i in 1:K
    priors["β[$(i)]"] = Normal(0,1);
end
for i in 1:K
    #priors["d[$(i)]"] = Normal(0,0.1);
    priors["d[$(i)]"] = Normal(0,0.1);
end
#for i in 1:M
#    priors["γ[$(i)]"] = truncated(Normal(0,0.1),lower=0);
#end
priors["γ"] = truncated(Normal(0,0.1),lower=0);
priors["b"] = Normal(0,1);
priors["σ"] = LogNormal(0,1);
priors["seed"] = truncated(Normal(0,0.1),lower=0);
#
# parameter refactorization
#factors = [1., [1 for _ in 1:M]..., [1 for _ in 1:M]..., [1 for _ in 1:M]..., [1 for _ in 1:M]...];  # death
#factors = [1., 1., [1 for _ in 1:K]..., [1 for _ in 1:K]..., 1];  # death
factors = [1., 1., [1 for _ in 1:K]..., [1 for _ in 1:K]..., 1, 1];  # death


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
                retro=true,
                seed_value=100,
                alg=Tsit5(),
                abstol=1e-6,
                reltol=1e-3,
                adtype=AutoReverseDiff(),  # without compile much faster for aggregation and death
                sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)),
                benchmark=false,
                benchmark_ad=[:reversediff],
                #M=M,
                test_typestable=false
                )

# SAVE 
serialize("simulations/total_$(ode)_N=$(N)_threads=$(n_threads)_var$(length(priors["σ"]))_smallgamma.jls", inference)
Distributed.interrupt()  # kill workers from previous run (killing REPL does not do this)
