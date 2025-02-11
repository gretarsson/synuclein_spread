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
ode = death_simplifiedii_bilateral;
n_threads = 1;

# read data
timepoints = vec(readdlm("data/timepoints.csv", ','));
data = deserialize("data/total_path_3D.jls");
#_, idxs = read_data("data/avg_total_path.csv", remove_nans=true, threshold=0.15);
#idxs = findall(idxs);



# get bilateral idxs
W_file = "data/W_labeled.csv"
W_labelled = readdlm(W_file,',')
labels = W_labelled[1,2:end]
idxs = only_bilateral(labels)
#
#seed_idx = findall(s -> contains(s,"iCP"),labels)
#labels2 = labels[bi_idxs]
#for i in 1:222
#    display("$(labels2[i]) + $(labels2[i+222]) ")
#end
#idxs = [[bi_idxs[i] for i in 78:88]...,[bi_idxs[i] for i in (78+222):(88+222)]...]
#labels2 = labels[idxs]
#for i in 1:Int(length(labels2)/2)
#    display("$(labels2[i]) + $(labels2[i+Int(length(labels2)/2)]) ")
#end





W_file = "data/W_labeled.csv"
W_labelled = readdlm(W_file,',')

if isempty(idxs)
    idxs = [i for i in 1:(size(W_labelled)[1] - 1)]
end
labels = W_labelled[1,2:end]
W = W_labelled[2:end,2:end]
W = W ./ maximum( W[ W .> 0 ] )  # normalize connecivity by its maximum
L = Matrix(transpose(laplacian_out(W; self_loops=false, retro=true)))  # transpose of Laplacian (so we can write LT * x, instead of x^T * L)


# DIFFUSION, RETRO- AND ANTEROGRADE
N = length(idxs);
M = Int(length(idxs)/2)
#N = size(data)[1];
display("N = $(N)")
u0 = [0. for _ in 1:(2*N)];

# DEFINE PRIORS
priors = OrderedDict{Any,Any}( "ρ" => truncated(Normal(0,0.1),lower=0) ); 
priors["α"] = truncated(Normal(0,0.1),lower=0);
for i in 1:M
    priors["β[$(i)]"] = truncated(Normal(0,1),lower=0);
end
for i in 1:M
    priors["d[$(i)]"] = Normal(0,1);
end
priors["γ"] = truncated(Normal(0,0.1),lower=0);
priors["σ"] = LogNormal(0,1);
priors["seed"] = truncated(Normal(0,0.1),lower=0);
#
# parameter refactorization
factors = [1., 1., [1 for _ in 1:M]..., [1 for _ in 1:M]..., 1];  # death


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
serialize("simulations/total_$(ode)_N=$(N)_threads=$(n_threads)_var$(length(priors["σ"])).jls", inference)
Distributed.interrupt()  # kill workers from previous run (killing REPL does not do this)
