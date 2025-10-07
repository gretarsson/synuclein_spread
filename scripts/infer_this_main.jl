#!/usr/bin/env julia --project=.
# =============================================================================
# Run Bayesian inference of prion-like spreading models on networks
#
# MAIN IDEA
# ---------
# This script is the command-line entry point for simulating and inferring
# parameters of ODE-based models of pathology spread on brain networks.
#
# WHAT YOU NEED TO RUN IT
# -----------------------
# 1. A choice of ODE model     → defined in "odes.jl"
# 2. Priors for that ODE       → defined in "model_priors.jl" (matched automatically)
# 3. A structural connectome   → adjacency matrix CSV with region labels
# 4. Pathology measurements    → pathology CSV with sample IDs, timepoints, regions
#
# NOTE: You do not specify priors on the command line.
#       The script looks up the priors associated with the chosen ODE
#       in "model_priors.jl".
#
# EXAMPLE USAGE
# -------------
#   julia --project=. scripts/infer_this_main.jl DIFF data/W_labeled.csv data/total_path.csv
#
# Here:
#   - "DIFF" is the ODE model (must match a model defined in "odes.jl")
#   - Priors for "DIFF" are taken from "model_priors.jl"
#   - "data/W_labeled.csv" is the weighted adjacency matrix (with row & column headers)
#   - "data/total_path.csv" is the pathology data file
#
# FILE FORMATS (details)
# ----------------------
# * W-file (structural connectivity):
#   - CSV with row and column headers naming the same set of regions
#   - Column 1 = row labels (region names)
#   - Columns 2:end = weights of adjacency matrix
#   - Column headers 2:end define canonical region order
#
# * Pathology data:
#   - Column 1 = sample/experiment ID (e.g., mouse ID)
#   - Column 2 = time points, with header "mpi" (months post-inoculation)
#   - Columns 3:end = pathology values per region (headers must match W-file)
#   - Missing values allowed (e.g., "NA")
#
# NOTES
# -----
# - The script orchestrates everything: loading data, building the ODE problem,
#   setting priors, running inference, and saving results.
# - The ODEs and priors themselves are not defined here, but in "odes.jl" and
#   "model_priors.jl" respectively.
# - Model name suffixes:
#       * "_bilateral"     → group parameters across left/right homologous regions
#       * "_bidirectional" → use both retrograde and anterograde Laplacians
# =============================================================================
# print to show we start
println("Starting inference script...")
flush(stdout)

using ArgParse
using Distributed

# NEW
using PathoSpread
# OLD
#include("Data_processing.jl")
#include("ODE_dimensions.jl")
#using .Data_processing: process_pathology

# for algorithm solver choice, sensitivity analysis
using DifferentialEquations
using SciMLSensitivity
using Turing


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
            help = "CSV pathology data file"
        # OPTIONAL ARGUMENTS
        "--n_chains"
            arg_type = Int
            default = 1
            help = "how many MCMC chains to run with distributed computing"
        "--retrograde"
            arg_type = Bool
            default = true
            help = "If true then retrograde transport is used, if false anterograde transport is used"
        "--seed_label"
            arg_type = String
            default  = "iCP"
            help     = "The label of the seeded region"
        "--seed_index"
            arg_type = Int
            default  = 0
            help     = "Index of the seeded region (1-based). Overrides --seed_label if > 0"
        "--infer_seed"
            arg_type = Bool
            default  = true
            help     = "If true, infer the value of the seeded region at time zero"
        "--target_acceptance"
            arg_type = Float64
            default = 0.65
            help = "The target acceptance ratio for the NUTS sampler"
        "--out_file"
            arg_type = String
            default = nothing
            help = "where to save the inference object"
        "--test"
            action = :store_true
            help = "if set, use test subset (this is hardcoded don't use)"
        "--holdout_last"
            arg_type = Int
            default  = 0
            help     = "Number of last timepoints to remove from data/timepoints before inference (0 = keep all)"
        "--shuffle"
            action = :store_true
            help = "If set, randomly permute the weights of the adjacency matrix before building the Laplacian (null model control)"
    end

    return s
end

function main(parsed)
    # read ARGS
    ode = parsed["ode"]
    w_file = parsed["w_file"]
    data_file = parsed["data_file"]
    n_chains = parsed["n_chains"]
    retrograde = parsed["retrograde"]
    seed_label = parsed["seed_label"]
    infer_seed = parsed["infer_seed"]
    target_acceptance = parsed["target_acceptance"]
    out_file = parsed["out_file"]
    test = parsed["test"]
    holdout_last = parsed["holdout_last"]
    shuffle = parsed["shuffle"]

    # PRINT ARGS
    println("→ ODE:        $ode")
    println("→ Structural data:      $w_file")
    println("→ Pathology data:   $data_file")
    println("→ Chains:    $n_chains")
    println("→ Retrograde:    $retrograde")
    println("→ Seed label:    $seed_label")
    println("→ Infer seed:    $infer_seed")
    println("→ Hold out last timepoints: $holdout_last")
    println("→ Shuffle network weights: $shuffle")
    println("→ Target acceptance:    $target_acceptance")
    println("→ Output:     $out_file")
    if test
        println("→ Test:     $test")
    end
    flush(stdout)

    # -----------------------------------
    #=
    Infer parameters of ODE using Bayesian framework
    =#
    # flag if bilateral and bidirectional
    bilateral = endswith(ode, "_bilateral")
    bidirectional = endswith(ode, "_bidirectional")

    # READ PATHOLOGY AND STRUCTURAL DATA
    if test 
        # HARD-CODED TEST, N=40 
        _, idxs = read_data("data/avg_total_path.csv", remove_nans=true, threshold=0.15);
        data, timepoints = process_pathology(data_file; W_csv="data/W_labeled.csv");
        data = data[idxs,:,:];
        Lr,N,labels = read_W("data/W_labeled.csv", direction=:retro, idxs=idxs);
        La,_,_ = read_W("data/W_labeled.csv", direction=:antero, idxs=idxs);
    else
        # PATHOLOGY DATA
        data, timepoints = process_pathology(data_file; W_csv=w_file);
        # STRUCTURAL DATA
        Lr,N,labels = read_W(w_file, direction=:retro, shuffle=shuffle);
        La,_,_ = read_W(w_file, direction=:antero, shuffle=shuffle);
        # REMOVE LAST TIMEPOINTS IF SPECIFIED
        if holdout_last < 0
            error("holdout_last must be ≥ 0, got $holdout_last")
        elseif holdout_last > 0
            T = length(timepoints)
            @assert holdout_last < T "holdout_last ($holdout_last) must be < number of timepoints ($T)."
            data        = data[:, 1:(T - holdout_last), :]
            timepoints  = timepoints[1:(T - holdout_last)]
        end
    end 

    # ADD EXTRA LAPLACIAN IF BIDIRECTIONAL TRANSPORT
    if bidirectional
        Ltuple = (Lr,La,N)  # order is (L,N) or (Lr, La, N). The latter is used for bidirectional spread
    else
        if retrograde
            Ltuple = (Lr,N)
        else
            Ltuple = (La,N)
        end
    end

    # SET SEED AND INITIAL CONDITIONS
    #seed = findfirst(==(seed_label), labels);  # OLD
    if parsed["seed_index"] > 0  # NEW
        seed = parsed["seed_index"]
    else
        seed = findfirst(==(seed_label), labels)
    end
    

    # SET PRIORS (variance and seed have to be last, in that order)
    region_group = build_region_groups(labels)  # prepare bilateral parameters
    K = bilateral ? maximum(region_group) : N
    priors = get_priors(ode,K)
    priors["σ"] = LogNormal(0,1);
    priors["seed"] = truncated(Normal(0,0.1),lower=0);

    # DEFINE ODE PROBLEM
    factors = ones(length(get_priors(ode,K)))
    u0 = zeros(ode_dimensions[ode](N))
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
                    alg=Tsit5(),  # from DifferentialEquations
                    abstol=1e-6,
                    reltol=1e-3,
                    adtype=AutoReverseDiff(),  # without compile much faster, from SciMLSensitivity
                    sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)),  # from Turing/SciMLSensitivity
                    target_acceptance=target_acceptance,
                    benchmark=false,
                    benchmark_ad=[:reversediff],
                    labels=labels,
                    ode_name=string(ode),
                    test_typestable=false
                    )
    println("Finished sampling")
    flush(stdout)

    # SAVE 
    if isnothing(out_file)
        out_file = "simulations/$(ode)_N=$(N)_threads=$(n_chains).jls"
        save_inference(out_file, inference)
    else
        save_inference(out_file, inference)
    end
    println("Saved inference dictionary at $(out_file)")
    flush(stdout)
end



# ---------------------------------------------------------------------------
# Entry point:
#   - Parse args
#   - Spawn workers (one per chain)
#   - Activate & prepare the project environment on each worker
#   - Load modeling code on each worker
#   - Run main(), then clean up workers
# ---------------------------------------------------------------------------
if abspath(PROGRAM_FILE) == @__FILE__

    # parse arguments
    parsed = parse_args(build_parser())
    n_chains = parsed["n_chains"]

    # 1) spin up your workers
    addprocs(n_chains)

    # 2) on *each* worker, load Pkg and prepare Env
    @everywhere using Pkg
    @everywhere Pkg.activate(".")
    #@everywhere Pkg.instantiate()  # supposedely not needed ChatGPT
    #@everywhere Pkg.precompile()  # same here

    # 3) on each worker, bring in your modeling code
    @everywhere using Turing
    @everywhere using ParallelDataTransfer
    # NEW
    @everywhere using PathoSpread
    # OLD
    #@everywhere include("helpers.jl")
    #@everywhere include("model_priors.jl")

    try
        main(parsed)
    finally
        rmprocs(workers())  # kill workers, closing Julia REPL does not do this
    end
    
end