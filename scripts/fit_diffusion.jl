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
simulation_code = "total_diffusion"

#=
read pathology data
=#
data, idxs = read_data("data/avg_total_path.csv", remove_nans=true, threshold=0.16)
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
alg = Tsit5()
tspan = (0.0,9.0)
# ICs
u0 = [0. for i in 1:N]  # initial conditions
u0[seed] = rand(Uniform(0,1))  # seed, past seed=15 some regions go beyond 1.
# Parameters
ρ = rand(truncated(Normal(0,2.5),lower=0.))
p = [ρ]
# setting up, solve, and plot
prob = ODEProblem(diffusion, u0, tspan, p; alg=alg)
sol = solve(prob,alg; abstol=1e-9, reltol=1e-9)
StatsPlots.plot(sol; legend=false, ylim=(0,1))


# -----------------------------------------------------------------------------------------------------------------
#=
Bayesian estimation of model parameters
=#
# Bayesian model input for priors and ODE IC
priors = Dict( 
            "σ" => InverseGamma(2,3), 
            "ρ" => Uniform(0,1), 
            "seed" => Normal(0.5,0.1) 
            )
@model function fitlv(data, prob; alg=alg, timepoints=timepoints, seed=seed, priors=priors)
    # Priors and initial conditions 
    u0 = [0. for _ in 1:N]
    σ ~ priors["σ"]
    ρ ~ priors["ρ"]  # (0.,2.5)
    u0[seed] ~ priors["seed"]  

    # Simulate diffusion model 
    p = [ρ]
    sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)) 
    predicted = solve(prob, alg; u0=u0, p=p, saveat=timepoints, sensealg=sensealg, abstol=1e-6, reltol=1e-3)

    # Observations.
    for i in axes(predicted,1)
        data[i, :] ~ MvNormal(predicted[i,:], σ^2 * I)
    end

    return nothing
end

# define Turing model
model = fitlv(data, prob)

# benchmarking 
suite = TuringBenchmarking.make_turing_suite(model;adbackends=[:forwarddiff,:reversediff])
run(suite)

# Sample to approximate posterior, and save
#chain = sample(model, NUTS(;adtype=AutoReverseDiff()), MCMCThreads(), 1000, 2; progress=true)
chain = sample(model, NUTS(), MCMCThreads(), 1000, 2; progress=false)
inference = Dict("chain" => chain, "priors" => priors, "model" => model)
serialize("simulations/"*simulation_code*".jls", inference)

# load inference results
inference = deserialize("simulations/"*simulation_code*".jls")
chain = inference["chain"]

# plot posterior distributions and retrodiction
save_folder = "figures/"*simulation_code
plot_chains(chain, save_folder*"/chains")
plot_retrodiction(data=data,prob=prob,chain=chain,timepoints=timepoints,path=save_folder*"/retrodiction",seed=seed)

# compute leave-one-out cross validation
loo = psis_loo(model, chain)
