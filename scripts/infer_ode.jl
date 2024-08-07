using Serialization
include("helpers.jl");
#=
Infer parameters of ODE using Bayesian framework
=#

# DIFFUSION, RETRO- AND ANTEROGRADE
thresholds = [0.15, 0.05, 0.01, 0.0];
Ns = [40, 95, 174, 366];
for i in eachindex(thresholds)
    println("Inferring N=$(Ns[i])")
    priors2 = OrderedDict( "σ" => LogNormal(0,1), "ρₐ" => truncated(Normal(0,0.01), lower=0.), "ρᵣ" => truncated(Normal(0,0.01), lower=0.), "seed" => truncated(Normal(1.,0.1),lower=0) );

    inference = infer(diffusion2, 
                    priors2,
                    "data/avg_total_path.csv",
                    "data/timepoints.csv", 
                    "data/W_labeled.csv"; 
                    n_threads=4,
                    threshold=thresholds[i],
                    abstol=1e-6,
                    reltol=1e-3,
                    adtype=AutoReverseDiff(compile=true),
                    sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)),
                    benchmark=false,
                    benchmark_ad=[:reversediff,:reversediff_compiled]
                    )

    # save inference result
    serialize("simulations/total_diffusion2_N=$(Ns[i]).jls", inference)
end
