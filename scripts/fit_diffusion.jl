using Turing
using DelimitedFiles
using StatsPlots
using DifferentialEquations
using Distributions
using TuringBenchmarking  # only version 0.5.1 works for Mac
using ReverseDiff
using SciMLSensitivity
using LinearAlgebra
using Serialization
using CairoMakie
using ParetoSmooth
# add helper functions
include("helpers.jl")

# Set name for files to be saved in figures/ and simulations/
simulation_code = "total_diffusion_N=40"
threshold = 0.16

#=
read pathology data
=#
data, idxs = read_data("data/avg_total_path.csv", remove_nans=true, threshold=threshold)
timepoints = vec(readdlm("data/timepoints.csv", ','))
plt = StatsPlots.plot()
for i in axes(data,1)
    StatsPlots.plot!(plt,timepoints, data[i,:], legend=false)
end
plt
N = size(data)[1]

#=
Read structural data 
=#
W = readdlm("data/W.csv",',')[idxs,idxs]
LT = transpose(laplacian_out(W))  # transpose of Laplacian (so we can write LT * x, instead of x^T * L)
labels = readdlm("data/W_labeled.csv",',')[1,2:end][idxs]
seed = findall(x->x=="iCP",labels)[1]  # find index of seed region

#=
Define, simulate, and plot model
=#
function diffusion(du,u,p,t;L=LT)
    ρ = p[1]

    du .= -ρ*L*u 
end
# ODE settings
alg = Rosenbrock23()
tspan = (0.0,9.0)
# ICs
u0 = [0. for i in 1:N]  # initial conditions
u0[seed] = rand(Uniform(0,1))  # seed, past seed=15 some regions go beyond 1.
# Parameters
ρ = rand(truncated(Normal(0,2.5),lower=0.))
p = [ρ]
# setting up, solve, and plot
sensealg = ForwardDiffSensitivity()
prob = ODEProblem(diffusion, u0, tspan, p; alg=alg)
sol = solve(prob,alg; abstol=1e-6, reltol=1e-3)
StatsPlots.plot(sol; legend=false, ylim=(0,1))


# -----------------------------------------------------------------------------------------------------------------
#=
Bayesian estimation of model parameters
=#
# Bayesian model input for priors and ODE IC
priors = Dict( 
            "σ" => LogNormal(0,1), 
            "ρ" => truncated(Normal(0,1.), lower=0.), 
            "u0[$(seed)]" => Uniform(0,1) 
            )
@model function bayesian_model(data, prob; alg=alg, timepoints=timepoints, seed=seed, priors=priors)
    # Priors and initial conditions 
    u0 = [0. for _ in 1:N]
    σ ~ priors["σ"]
    ρ ~ priors["ρ"]  
    u0[seed] ~ priors["u0[$(seed)]"]  

    # Simulate diffusion model 
    p = [ρ]
    sensealg = ForwardDiffSensitivity() 
    predicted = solve(prob, alg; u0=u0, p=p, saveat=timepoints, sensealg=sensealg, abstol=1e-6, reltol=1e-3)

    # Observations.
    for i in axes(predicted,1), j in axes(predicted,2)
        data[i,j] ~ Normal(predicted[i,j], σ^2)
    end

    return nothing
end

# define Turing model
model = bayesian_model(data, prob)

# benchmarking 
suite = TuringBenchmarking.make_turing_suite(model;adbackends=[:forwarddiff,:reversediff])
run(suite)

# Sample to approximate posterior, and save
chain = sample(model, NUTS(;adtype=AutoForwardDiff()), MCMCThreads(), 1000, 4; progress=true)

# plot posterior distributions and retrodiction
save_folder = "figures/"*simulation_code
plot_chains(chain, save_folder*"/chains"; priors=priors)
plot_retrodiction(data=data,prob=prob,chain=chain,timepoints=timepoints,path=save_folder*"/retrodiction",seed=seed, seed_bayesian=true, u0=u0)

# compute elpd (expected log predictive density)
elpd = compute_psis_loo(model,chain)

# save
inference = Dict("chain" => chain, "priors" => priors, "model" => model, "elpd" => elpd, "threshold" => threshold)
serialize("simulations/"*simulation_code*".jls", inference)
