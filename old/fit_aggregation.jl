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
include("../scripts/helpers.jl")

# Set name for files to be saved in figures/ and simulations/
simulation_code = "total_aggregation_N=40"
data_threshold = 0.15

#=
read pathology data
=#
data, idxs = read_data("data/avg_total_path.csv", remove_nans=false, threshold=data_threshold)
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
W = W ./ maximum( W[W.>0] )
LT = transpose(laplacian_out(W))  # transpose of Laplacian (so we can write LT * x, instead of x^T * L)
LT = laplacian_out(W)
labels = readdlm("data/W_labeled.csv",',')[1,2:end][idxs]
seed = findall(x->x=="iCP",labels)[1]  # find index of seed region

#=
Define, simulate, and plot model
=#
function aggregation2(du,u,p,t;L=LT)
    ρ = p[1]
    α = p[2]
    β = p[3:(N+2)]

    du .= -ρ*L*u .+ α .* u .* (β .- u) .* (u .- 1e-6)  
end
function stochastic(du,u,p,t)
    du .= p[end] .* u
end
# ODE settings
alg = EM()
#alg = Tsit5()
tspan = (0.0,9.0)
# ICs
u0 = [0.0 for i in 1:N]  # initial conditions
u0[seed] = 0.01  # seed

# Parameters
ρ = 1
α = 50
#α = [rand(truncated(Normal(10.,0.5))) for _ in 1:N]
β = [1. for i in 1:N]
σ = 0.01
#p = [ρ, α..., β...]
p = [ρ, α, β..., σ]
# setting up, solve, and plot
#prob = ODEProblem(aggregation2, u0, tspan, p; alg=alg)
prob = SDEProblem(aggregation2, stochastic, u0, tspan, p; alg=alg)
sol = solve(prob,alg,dt=1e-3; abstol=1e-10, reltol=1e-10)
StatsPlots.plot(sol; legend=false)

# testing a split function
function f1(du,u,p,t;L=LT)
    ρ = p[1]

    du .= -ρ*L*u   
end
function f2(du,u,p,t)
    α = p[2]
    β = p[3:end]

    du .= α .* u .* (β .- u)  
end
laplacian = DiffEqArrayOperator(-p[1]*Matrix(LT))
split = SplitFunction(laplacian,f2)
# ODE settings
alg = LawsonEuler(krylov=false)
tspan = (0.0,9.0)
# ICs
u0 = [0. for i in 1:N]  # initial conditions
u0[seed] = 0.1  # seed
# Parameters
ρ = 0.001
α = 3
#α = [rand(truncated(Normal(10.,0.5))) for _ in 1:N]
β = [1. for i in 1:N]
#p = [ρ, α..., β...]
p = [ρ, α, β...]
split_prob = SplitODEProblem(split,u0,tspan,p)
sol = solve(split_prob,alg; abstol=1e-9, reltol=1e-9, dt=1e-3)
StatsPlots.plot(sol; legend=false)



# -----------------------------------------------------------------------------------------------------------------
#=
Bayesian estimation of model parameters
=#
# Bayesian model input for priors and ODE IC
priors = Dict( 
            "σ" => LogNormal(0.,1.), 
            "ρ" => truncated(Normal(0,0.01),lower=0.), 
            "u0[$(seed)]" => truncated(Normal(0.0,0.01),lower=0., upper=1.),
            "α" => truncated(Normal(0.,2.5),lower=0.),
            #"α" => arraydist([truncated(Normal(10.,5.), lower=0.) for _ in 1:N]),
            "β" => arraydist([truncated(Normal(0.0,0.25), lower=0., upper=1.) for _ in 1:N])
            )
@model function bayesian_model(data, prob; alg=alg, timepoints=timepoints, seed=seed, priors=priors)
    # Priors and initial conditions 
    u0 = [0. for _ in 1:N]
    σ ~ priors["σ"]
    ρ ~ priors["ρ"]  # (0.,2.5)
    α ~ priors["α"]
    β ~ priors["β"]
    u0[seed] ~ priors["u0[$(seed)]"]  

    # Simulate diffusion model 
    #p = [ρ,α...,β...]
    p = [ρ,α,β...]
    sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)) 
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
suite = TuringBenchmarking.make_turing_suite(model;adbackends=[:reversediff])
run(suite)

# Sample to approximate posterior, and save
chain = sample(model, NUTS(0.65;adtype=AutoReverseDiff()), 1000; progress=true)
#chain = sample(model, NUTS(;adtype=AutoReverseDiff()), MCMCThreads(), 1000, 4; progress=true)

# plot posterior distributions and retrodiction
save_folder = "figures/"*simulation_code
plot_chains(chain, save_folder*"/chains"; priors=priors)
plot_retrodiction(data=data,prob=prob,chain=chain,timepoints=timepoints,path=save_folder*"/retrodiction",seed=seed, seed_bayesian=true, u0=u0)

# compute elpd (expected log predictive density)
elpd = compute_psis_loo(model,chain)
waic = elpd.estimates[2,1] - elpd.estimates[3,1]  # naive elpd - p_eff

# hypothesis testing
prior_alpha = priors["α"]
posterior_alpha = KernelDensity.kde(vec(chain[:α]), boundary=(0,40))
savage_dickey_density = pdf(posterior_alpha,0.) / pdf(prior_alpha, 0.)
println("Probability of aggregation model: $(1 - savage_dickey_density / (savage_dickey_density+1))")

# save  
inference = Dict("chain" => chain, "priors" => priors, "elpd" => elpd, "data_threshold" => data_threshold, "savage_dickey_density" => savage_dickey_density, "data" => data, "waic" => waic)
serialize("simulations/"*simulation_code*".jls", inference)
