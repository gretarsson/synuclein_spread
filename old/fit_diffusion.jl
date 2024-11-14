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
include("../scripts/helpers.jl")

# Set name for files to be saved in figures/ and simulations/
simulation_code = "total_diffusion_N=40"
threshold = 0.0
retro = true

#=
read pathology data
=#
data, idxs = read_data("data/avg_total_path.csv", remove_nans=false, threshold=threshold)
timepoints = vec(readdlm("data/timepoints.csv", ','))
plt = StatsPlots.plot(;ylabel="total pathology", xlabel="time (month)")
for i in axes(data,1)
    StatsPlots.plot!(plt,timepoints, data[i,:], legend=false)
end
plt
N = size(data)[1]
save("figures/total_path/all_$(N)_regions.png",plt)

#=
Read structural data 
=#
W = readdlm("data/W.csv",',')[idxs,idxs]
#W = W ./ maximum(W[W .> 0])
L = Matrix(transpose(laplacian_out(W; self_loops=true, retro=true)))  # transpose of Laplacian (so we can write LT * x, instead of x^T * L)
#L = Matrix(laplacian_out(W; self_loops=false, retro=retro))  # transpose of Laplacian (so we can write LT * x, instead of x^T * L)
labels = readdlm("data/W_labeled.csv",',')[1,2:end][idxs]
seed = findall(x->x=="iCP",labels)[1]  # find index of seed region

#=
Define, simulate, and plot model
=#
function diffusion2(du,u,p,t;L=L)
    ρ = p[1]

    du .= -ρ*L*u 
end
# ODE settings
alg = Tsit5()
tspan = (0.0,9.0)
# ICs
u0 = [0. for i in 1:N]  # initial conditions
u0[seed] = 100 # seed, past seed=15 some regions go beyond 1.
# Parameters
ρ = 1
p = [ρ]
# setting up, solve, and plot
prob = ODEProblem(diffusion2, u0, tspan, p; alg=alg)
sol = solve(prob,alg; abstol=1e-6, reltol=1e-3, maxiters=100)
size(sol)
StatsPlots.plot(sol; legend=false, ylim=(0,1.))

# -----------------------------------------------------------------------------------------------------------------
#=
Bayesian estimation of model parameters
=#
# Bayesian model input for priors and ODE IC
priors = Dict( 
            "σ" => LogNormal(0,1), 
            "ρ" => truncated(Normal(0,0.1), lower=0.), 
            "u0[$(seed)]" => truncated(Normal(0.,2.5), lower=0.) 
            )
@model function bayesian_model(data, prob; alg=alg, timepoints=timepoints, seed=seed, priors=priors, L=L)
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
chain = sample(model, NUTS(;adtype=AutoForwardDiff()), 1000; progress=true)
#chain = sample(model, NUTS(;adtype=AutoForwardDiff()), MCMCThreads(), 1000, 4; progress=true)

# save chain
inference = Dict("chain" => chain, "priors" => priors, "data_threshold" => threshold, "data" => data, "retro" => retro)
serialize("simulations/"*simulation_code*".jls", inference)

# plot posterior distributions and retrodiction
save_folder = "figures/"*simulation_code
plot_chains(chain, save_folder*"/chains"; priors=priors)
plot_retrodiction(data=data,prob=prob,chain=chain,timepoints=timepoints,path=save_folder*"/retrodiction",seed=seed, seed_bayesian=true, u0=u0)

# compute elpd (expected log predictive density)
elpd = compute_psis_loo(model,chain)
waic = elpd.estimates[2,1] - elpd.estimates[3,1]  # naive elpd - p_eff

# save
inference = Dict("chain" => chain, "priors" => priors, "elpd" => elpd, "data_threshold" => threshold, "data" => data, "retro" => retro)
serialize("simulations/"*simulation_code*".jls", inference)
