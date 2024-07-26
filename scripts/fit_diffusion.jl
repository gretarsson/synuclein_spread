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
save_folder = "figures/diffusion_inference/diffusion_data"

#=
read pathology data
=#
data, idxs = read_data("data/avg_total_path.csv", remove_nans=true, threshold=0.)
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
function diffusion(du,u,p,t;L=LT)
    ρ = p[1]

    du .= -ρ*L*u 
end
# ODE settings
alg = Tsit5()
tspan = (0.0,9.0)
# ICs
u0 = [0. for i in 1:N]  # initial conditions
u0[seed] = rand(Uniform(0,1))  # seed
# Parameters
ρ = rand(truncated(Normal(0,2.5),lower=0.))
p = [ρ]
# setting up, solve, and plot
sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)) 
prob = ODEProblem(diffusion, u0, tspan, p; alg=alg)
sol = solve(prob,alg; abstol=1e-9, reltol=1e-9, sensealg=sensealg)
StatsPlots.plot(sol; legend=false)


# -----------------------------------------------------------------------------------------------------------------
#=
Bayesian estimation of model parameters
=#
# Bayesian model input for priors and ODE IC
@model function fitlv(data, prob; alg=alg, timepoints=timepoints, seed=seed)
    # Prior on model parameters
    u0 = [0. for _ in 1:N]
    σ ~ InverseGamma(2, 3)
    ρ ~ Uniform(0, 1)  # (0.,2.5)
    u0[seed] ~ Uniform(0.,1.)  

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
plot_chains(chain, save_folder*"/chains")
plot_retrodiction(data=data,prob=prob,chain=chain,timepoints=timepoints,path=save_folder*"/retrodiction")