using ArgParse

# use ArgParse to define CLI arguments
function build_parser()
    s = ArgParseSettings()

    @add_arg_table s begin
        "ode"
            arg_type = String
            help = "string naming the ode"
        "w_file"
            arg_type = String
            help = "CSV file of structural connectivity"
        "data_file"
            arg_type = String
            help = "3D pathology data file"
        "--time_file"
            arg_type = String
            default = nothing
            help   = "file of timepoints csv file"
        "--n_chains"
            arg_type = Int
            default = 1
            help = "how many MCMC chains to run with distributed computing"
        "--output", "-o"
            arg_type = String
            default  = "results.txt"
            help     = "Output filename"
    end

    return s
end
# parse arguments
parsed = parse_args(build_parser())

# read ARGS
ode = parsed["ode"]
w_file = parsed["w_file"]
data_file = parsed["data_file"]
time_file = parsed["time_file"]
n_chains = parsed["n_chains"]

#using Turing
using Distributed
addprocs(n_chains-1)
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

# PRINT ARGS
println("→ ODE:        $ode")
println("→ Pathology data:   $data_file")
println("→ Structural data:      $w_file")
println("→ Timepoints:   $time_file")
intln("→ #Chains:    $n_chains")
#println("→ Output:     $out_file")


# -----------------------------------
#=
Infer parameters of ODE using Bayesian framework
=#
# flag if bilateral
bilateral = endswith(ode, "_bilateral")

# READ DATA
_, thr_idxs = read_data("data/avg_total_path.csv", remove_nans=true, threshold=0.15);
data = deserialize(data_file)[thr_idxs,:,:];
if isnothing(time_file)
    timepoints = Float64.(1:size(data)[2])
else
    timepoints = vec(readdlm(time_file, ','));
end

# LOAD CONNECTOME AND MAKE LAPLACIAN
Lr,N,labels = read_W(w_file, direction=:retro, idxs=thr_idxs );
La,_,_ = read_W(w_file, direction=:antero);
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
                n_chains=n_chains,
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
                ode_name=string(ode),
                test_typestable=false
                )

# SAVE 
serialize("simulations/total_$(ode)_N=$(N)_threads=$(n_threads)_var$(length(priors["σ"]))_NEWRHS.jls", inference)
Distributed.interrupt()  # kill workers from previous run (killing REPL does not do this)
