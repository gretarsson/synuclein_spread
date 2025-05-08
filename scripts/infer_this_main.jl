#!/usr/bin/env julia --project=.
using ArgParse
using Distributed
include("Data_processing.jl")
include("ODE_dimensions.jl")
using .Data_processing: process_pathology


# use ArgParse to define CLI arguments
function build_parser()
    s = ArgParseSettings()

    @add_arg_table s begin
        # POSITIONAL ARGUMENTS
        "ode"
            arg_type = String
            help = "string naming the ode"
        "w_file"
            arg_type = String
            help = "CSV file of structural connectivity"
        "data_file"
            arg_type = String
            help = "3D pathology data file"
        # OPTIONAL ARGUMENTS
        "--time_file"
            arg_type = String
            default = nothing
            help   = "file of timepoints csv file"
        "--n_chains"
            arg_type = Int
            default = 1
            help = "how many MCMC chains to run with distributed computing"
        "--seed_label"
            arg_type = String
            default  = "iCP"
            help     = "The label of the seeded region"
        "--infer_seed"
            arg_type = Bool
            default  = true
            help     = "If true, infer the value of the seeded region at time zero"
        "--out_file"
            arg_type = String
            default = nothing
            help = "where to save the inference object"
        "--test"
            action = :store_true
            help = "if set, use test subset (this is hardcoded don't use)"
    end

    return s
end

function main(parsed)
    # read ARGS
    ode = parsed["ode"]
    w_file = parsed["w_file"]
    data_file = parsed["data_file"]
    time_file = parsed["time_file"]
    n_chains = parsed["n_chains"]
    seed_label = parsed["seed_label"]
    infer_seed = parsed["infer_seed"]
    out_file = parsed["out_file"]
    test = parsed["test"]

    # PRINT ARGS
    println("→ ODE:        $ode")
    println("→ Structural data:      $w_file")
    println("→ Pathology data:   $data_file")
    println("→ Timepoints:   $time_file")
    println("→ Chains:    $n_chains")
    println("→ Seed label:    $seed_label")
    println("→ Infer seed:    $infer_seed")
    println("→ Output:     $out_file")
    if test
        println("→ Test:     $test")
    end

    # -----------------------------------
    #=
    Infer parameters of ODE using Bayesian framework
    =#
    # flag if bilateral and bidirectional
    bilateral = endswith(ode, "_bilateral")
    bidirectional = endswith(ode, "_bidirectional")

    # READ DATA
    #data = deserialize(data_file);
    data = process_pathology(data_file; W_csv=w_file)
    Ntotal = size(data)[1]
    if test 
        _, idxs = read_data("data/avg_total_path.csv", remove_nans=true, threshold=0.15);
    else
        idxs = trues(Ntotal)
    end 
    data = process_pathology(data_file; W_csv=w_file)[idxs,:,:];
    if isnothing(time_file)
        timepoints = Float64.(1:size(data)[2])
    else
        timepoints = vec(readdlm(time_file, ','));
    end

    # LOAD CONNECTOME AND MAKE LAPLACIAN
    Lr,N,labels = read_W(w_file, direction=:retro, idxs=idxs);
    La,_,_ = read_W(w_file, direction=:antero, idxs=idxs);
    if bidirectional
        Ltuple = (Lr,La,N)  # order is (L,N) or (Lr, La, N). The latter is used for bidirectional spread
    else
        Ltuple = (Lr,N)
    end

    # SET SEED AND INITIAL CONDITIONS
    seed = findfirst(==(seed_label), labels);  

    # SET PRIORS (variance and seed have to be last, in that order)
    region_group = build_region_groups(labels)  # prepare bilateral parameters
    K = bilateral ? maximum(region_group) : N
    priors = get_priors(ode,K)
    priors["σ"] = LogNormal(0,1);
    priors["seed"] = truncated(Normal(0,0.1),lower=0);

    # DEFINE ODE PROBLEM
    factors = ones(length(get_priors(ode,K)))
    u0 = zeros(ODE_dimensions[ode](N))
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
                    bayesian_seed=infer_seed,
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
    if isnothing(out_file)
        serialize("simulations/total_$(ode)_N=$(N)_threads=$(n_threads)_var$(length(priors["σ"]))_NEW.jls", inference)
    else
        serialize(out_file, inference)
    end
end


if abspath(PROGRAM_FILE) == @__FILE__
    # parse arguments
    parsed = parse_args(build_parser())
    n_chains = parsed["n_chains"]

    # 1) spin up your workers
    addprocs(n_chains)

    # 2) on *each* worker, load Pkg and prepare Env
    @everywhere using Pkg
    @everywhere Pkg.activate(".")
    @everywhere Pkg.instantiate()
    @everywhere Pkg.precompile()

    # 3) on each worker, bring in your modeling code
    @everywhere using Turing
    @everywhere using ParallelDataTransfer
    @everywhere include("helpers.jl")
    @everywhere include("model_priors.jl")

    try
        main(parsed)
    finally
        rmprocs(workers())  # kill workers, closing Julia REPL does not do this
    end
    
end