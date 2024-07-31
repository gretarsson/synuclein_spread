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
using KernelDensity
using Interpolations
using Distributions
using StatsPlots
# add helper functions
include("helpers.jl")

# Set name for files to be saved in figures/ and simulations/
simulation_code = "total_death_N=40_wo_beta"
data_threshold = 0.16

#=
read pathology data
=#
data, idxs = read_data("data/avg_total_path.csv", remove_nans=true, threshold=data_threshold)
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
function dad(du,u,p,t;L=LT)  # diffusion-aggregation-death
    # unpack variables
    x = u[1:N]  # protein concentration / pathology
    y = u[N+1:end]  # proportion of cells alive
    # unpack parameters
    ρ = p[1]
    α = p[2:(N+1)]
    γ = p[(N+2):(2*N+1)]
    κ = p[(2*N+2):(3*N+1)]
    # rhs
    du[1:N] .= -ρ*L*x .+ α .* x .* (y .- x)  
    du[N+1:2*N] .= -1 ./ γ .* (y .- (1 .- κ.*x))  
end
# ODE settings
alg = Tsit5()
tspan = (0.0,9.0)
# ICs
x0 = [0. for i in 1:N]  # initial conditions
x0[seed] = rand(truncated(Normal(0.,0.05), lower=0.))  
y0 = [1. for i in 1:N]
u0 = [x0...,y0...]
# Parameters
ρ = rand(truncated(Normal(0.,0.1),lower=0.))
α = [rand(truncated(Normal(5,1), lower=0.)) for i in 1:N]
γ = [rand(truncated(Normal(0,0.25), lower=0.)) for i in 1:N]
κ = [rand(truncated(Normal((1-data[i,end])/data[i,end],0.1), lower=0.)) for i in 1:N]
p = [ρ, α..., γ..., κ...]
# setting up, solve, and plot
prob = ODEProblem(dad, u0, tspan, p; alg=alg)
sol = solve(prob,alg; abstol=1e-9, reltol=1e-6)
StatsPlots.plot(sol; legend=false, ylim=(0,1), idxs=1:N)

# -----------------------------------------------------------------------------------------------------------------
#=
Bayesian estimation of model parameters
=#
# Bayesian model input for priors and ODE IC
priors = Dict( 
            "σ" => InverseGamma(2,3), 
            "ρ" => truncated(Normal(0,0.1),lower=0.), 
            "seed" => truncated(Normal(0.0,0.05),lower=0.),
            "α" => arraydist([truncated(Normal(5., 1); lower=0.) for i in 1:N]),  
            "γ" => arraydist([truncated(Normal(0.,0.25), lower=0.) for i in 1:N]),
            "κ" => arraydist([truncated(Normal((1-data[i,end])/data[i,end], 0.1); lower=0.) for i in 1:N]),  
            )
@model function bayesian_model(data, prob; alg=alg, timepoints=timepoints, seed=seed, priors=priors, N=N)
    # Priors and initial conditions 
    x0 = [0. for _ in 1:N]
    y0 = [1. for _ in 1:N]
    σ ~ priors["σ"]
    ρ ~ priors["ρ"]  
    α ~ priors["α"]
    γ ~ priors["γ"]
    κ ~ priors["κ"]
    x0[seed] ~ priors["seed"]  

    # Simulate diffusion model 
    p = [ρ,α...,γ...,κ...]
    u0 = [x0...,y0...]
    sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)) 
    predicted = solve(prob, alg; u0=u0, p=p, saveat=timepoints, sensealg=sensealg, abstol=1e-9, reltol=1e-6)

    # Observations.
    for i in 1:N
        data[i, :] ~ MvNormal(predicted[i,:], σ^2 * I)
    end

    return nothing
end

# define Turing model
model = bayesian_model(data, prob)

# benchmarking 
suite = TuringBenchmarking.make_turing_suite(model;adbackends=[:forwarddiff,:reversediff])
run(suite)

# Sample to approximate posterior
chain = sample(model, NUTS(;adtype=AutoReverseDiff()), 1000; progress=true)

# plot posterior distributions and retrodiction
save_folder = "figures/"*simulation_code
plot_chains(chain, save_folder*"/chains")
plot_retrodiction(data=data,prob=prob,chain=chain,timepoints=timepoints,path=save_folder*"/retrodiction",seed=seed,seed_bayesian=true, u0=u0)

# hypothesis testing
prior_interest = priors["γ"]
posterior_interest = KernelDensity.kde(vec(chain[:γ]), boundary=(0,40))
savage_dickey_density = pdf(posterior_interest,0.) / pdf(prior_interest, 0.)
println("Probability of model: $(1 - savage_dickey_density / (savage_dickey_density+1))")

# save chain, model, priors, data threshold, and hypothesis test 
inference = Dict("chain" => chain, "priors" => priors, "model" => model, "data_threshold" => data_threshold, "savage_dickey_density" => savage_dickey_density)
serialize("simulations/"*simulation_code*".jls", inference)
