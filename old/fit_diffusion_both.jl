using Turing
using DelimitedFiles
using StatsPlots
using DifferentialEquations
using Distributions
using TuringBenchmarking  
using ReverseDiff
using SciMLSensitivity
using LinearAlgebra
using Serialization
using CairoMakie
using ParetoSmooth
# add helper functions
include("helpers.jl")

# Set name for files to be saved in figures/ and simulations/
simulation_code = "both_total_diffusion_N=40"
threshold = 0.00

#=
read pathology data
=#
data, idxs = read_data("data/avg_total_path.csv", remove_nans=true, threshold=threshold)
timepoints = vec(readdlm("data/timepoints.csv", ','))::Vector{Float64}
plt = StatsPlots.plot()
for i in axes(data,1)
    StatsPlots.plot!(plt,timepoints, data[i,:], legend=false)
end
plt
N = size(data)[1]

#=
Read structural data 
=#
Wa = readdlm("data/W.csv",',')[idxs,idxs]
LTa = transpose(laplacian_out(Wa; self_loops=false))  # transpose of Laplacian (so we can write LT * x, instead of x^T * L)
Wr = transpose(Wa)
LTr = transpose(laplacian_out(Wr; self_loops=false))  # transpose of Laplacian (so we can write LT * x, instead of x^T * L)
labels = readdlm("data/W_labeled.csv",',')[1,2:end][idxs]
seed = findall(x->x=="iCP",labels)[1]  # find index of seed region

#=
Define, simulate, and plot model
=#
#function both_diffusion(du,u,p,t;Lr=LTr,La=LTa)
function both_diffusion(du,u,p,t)
    pa = p[1]
    pr = p[2]


    du .= -pr*LTr*u - pa*LTa*u   # this gives fast gradient computation
    #du .= -(pr*LTr + pa*LTa)*u  # this gives superslow gradient computation 
end
# ODE settings
alg = Tsit5()
tspan = (0.0,9.0)
# ICs
u0 = [0. for i in 1:N]  # initial conditions
u0[seed] = 1 # seed, past seed=15 some regions go beyond 1.
# Parameters
pa = 0.01
pr = 0.02
p = [pa, pr]
# setting up, solve, and plot
#sensealg = ForwardDiffSensitivity()
prob = ODEProblem(both_diffusion, u0, tspan, p; alg=alg)
sol = solve(prob,alg; abstol=1e-10, reltol=1e-3)
StatsPlots.plot(sol; legend=false, ylim=(0,1))


# -----------------------------------------------------------------------------------------------------------------
#=
Bayesian estimation of model parameters
=#
# Bayesian model input for priors and ODE IC
priors = Dict( 
            "σ" => LogNormal(0,1), 
            "ρa" => truncated(Normal(0,0.01), lower=0.), 
            "ρr" => truncated(Normal(0,0.01), lower=0.), 
            "u0[$(seed)]" => truncated(Normal(1.,0.1), lower=0.) 
            )
@model function bayesian_model(data, prob; timepoints=timepoints::Vector{Float64}, seed=seed::Int)
    # Priors and initial conditions 
    u0 = [0. for _ in 1:N]
    #σ ~ priors["σ"]
    #ρa ~ priors["ρa"]  
    #ρr ~ priors["ρr"]  
    #u0[seed] ~ priors["u0[$(seed)]"]  
    σ ~ LogNormal(0,1)
    p1 ~ truncated(Normal(0,0.01), lower=0.)  
    p2 ~ truncated(Normal(0,0.01), lower=0.)  
    u0[seed] ~ truncated(Normal(1,0.1), lower=0.)  

    # Simulate diffusion model 
    p = [p1, p2]
    #sensealg = ForwardDiffSensitivity() 
    sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))   # best sensealg for diffusion2 (with reversediff)
    #sensealg = ForwardDiffSensitivity()  
    alg = Tsit5()
    predicted = solve(prob, alg; u0=u0, p=p, saveat=timepoints, sensealg=sensealg, abstol=1e-10, reltol=1e-10, trajectories=N)
    #predicted = solve(prob, alg; u0=u0, p=p, saveat=timepoints, abstol=1e-10, reltol=1e-10)

    # Observations.
    for i in axes(predicted,1), j in axes(predicted,2)  # this is quiker than mvnormal over rows
        data[i,j] ~ Normal(predicted[i,j], σ^2)
    end
    #predicted = vec(predicted)
    #data ~ MvNormal(predicted, σ^2*I)

    return nothing
end

# define Turing model
model = bayesian_model(data, prob)
#@code_warntype model.f(
#    model,
#    Turing.VarInfo(model),
#    Turing.SamplingContext(
#        Random.GLOBAL_RNG, Turing.SampleFromPrior(), Turing.DefaultContext(),
#    ),
#    model.args...,
#)

# benchmarking 
suite = TuringBenchmarking.make_turing_suite(model;adbackends=[:reversediff])
run(suite)

# Sample to approximate posterior, and save
chain = sample(model, NUTS(;adtype=AutoForwardDiff()), 1000; progress=true)
#chain = sample(model, NUTS(;adtype=AutoForwardDiff()), MCMCThreads(), 1000, 4; progress=true)

# plot posterior distributions and retrodiction
save_folder = "figures/"*simulation_code
plot_chains(chain, save_folder*"/chains"; priors=priors)
plot_retrodiction(data=data,prob=prob,chain=chain,timepoints=timepoints,path=save_folder*"/retrodiction",seed=seed, seed_bayesian=true, u0=u0)

# compute elpd (expected log predictive density)
elpd = compute_psis_loo(model,chain)

# save
inference = Dict("chain" => chain, "priors" => priors, "model" => model, "elpd" => elpd, "data_threshold" => threshold, "data" => data, "retro" => 2)
serialize("simulations/"*simulation_code*".jls", inference)
