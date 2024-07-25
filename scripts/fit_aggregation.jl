using Turing
using DifferentialEquations
using OrdinaryDiffEq
using BenchmarkTools
using DelimitedFiles

# Load StatsPlots for visualizations and diagnostics.
using CairoMakie
using Colors
import StatsPlots
Makie.inline!(true)

using LinearAlgebra
using ReverseDiff
using Zygote
using SciMLSensitivity
using TuringBenchmarking  # only version 0.5.1 works for Mac

# Set a seed for reproducibility.
using Random
Random.seed!(16);

include("helpers.jl")

#=
read pathology data
=#
data = read_data("data/avg_total_path.csv", remove_nans=true, threshold=0.)
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
LT = transpose(laplacian_out(W))
# find seed iCP
labels = readdlm("data/W_labeled.csv",',')[1,2:end][idxs]
seed = findall(x->x=="iCP",labels)[1]

#=
Define, simulate, and plot model
=#
function aggregation(du,u,p,t;L=LT)
    ρ = p[1]
    α = p[2]
    β = p[3:end]

    du .= -ρ*L*u .+ α .* u .* (β .- u)  
end
# ODE settings
alg = Tsit5()
tspan = (0.0,9.0)
# ICs
u0 = [0. for i in 1:N]  # initial conditions
u0[seed] = 1e-5  # seed
# Parameters
ρ = rand(truncated(Normal(0,2.5),lower=0.))
α = rand(Normal(50,10))
β = [rand(truncated(Normal(data[i,end],0.1), lower=0.)) for i in 1:N]
p = [ρ, α, β...]
# setting up, solve, and plot
prob = ODEProblem(aggregation, u0, tspan, p; alg=alg)
sol = solve(prob,alg; abstol=1e-9, reltol=1e-6, saveat=timepoints)
StatsPlots.plot(sol; legend=false, ylim=(0,1))


# -----------------------------------------------------------------------------------------------------------------
#=
Bayesian estimation of model parameters
=#
# Bayesian model input for priors and ODE IC
β0 = [data[i,end] for i in 1:N]  # β prior average
u0 = [0. for _ in 1:N]
u0[seed] = 1e-5

@model function fitlv(data, prob; alg=alg, timepoints=timepoints, u0=u0, β0=β0)
    # Prior on model parameters
    σ ~ InverseGamma(2, 3)
    ρ ~ truncated(Normal(0., 2.5); lower=0.)  # (0.,2.5)
    α ~ truncated(Normal(50, 10); lower=0.)  # (0.,2.5)
    β ~ arraydist([truncated(Normal(β0[i], 0.1); lower=0.) for i in 1:N])  # vector

    # Simulate diffusion model 
    p = [ρ,α,β...]
    sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)) 
    predicted = solve(prob, alg; u0=u0, p=p, saveat=timepoints, sensealg=sensealg, abstol=1e-9, reltol=1e-6)

    # Observations.
    for i in axes(predicted,1)
        data[i, :] ~ MvNormal(predicted[i,:], σ^2 * I)
    end

    return nothing
end

# define Turing model
model = fitlv(data, prob)

# benchmarking 
benchmark_model(
    model;
    # Check correctness of computations
    check=true,
        # Automatic differentiation backends to check and benchmark
        adbackends=[:reversediff]
)

# Sample to approximate posterior
chain = sample(model, NUTS(;adtype=AutoReverseDiff()), 1000; progress=true)

# plot posterior distributions and retrodiction
plot_chains(chain, "figures/aggregation_inference/aggregation_data/chains")
plot_retrodiction(data=data,prob=prob,chain=chain,timepoints=timepoints,path="figures/aggregation_inference/aggregation_data/retrodiction")