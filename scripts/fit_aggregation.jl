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
simulation_code = "total_aggregation_N=366"
data_threshold = 0.00

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
α = rand(Normal(0,2.5))
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
priors = Dict( 
            "σ" => InverseGamma(2,3), 
            "ρ" => truncated(Normal(0,0.1),lower=0.), 
            "seed" => truncated(Normal(0.0,0.1),lower=0.),
            "α" => truncated(Normal(0.,2.5), lower=0.),
            "β" => arraydist([truncated(Normal(data[i,end], 0.05); lower=0.) for i in 1:N])  # vector
            )
@model function fitlv(data, prob; alg=alg, timepoints=timepoints, seed=seed, priors=priors)
    # Priors and initial conditions 
    u0 = [0. for _ in 1:N]
    u0[seed] = 1e-5
    σ ~ priors["σ"]
    ρ ~ priors["ρ"]  # (0.,2.5)
    α ~ priors["α"]
    β ~ priors["β"]
    u0[seed] ~ priors["seed"]  

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
suite = TuringBenchmarking.make_turing_suite(model;adbackends=[:forwarddiff,:reversediff])
run(suite)

# Sample to approximate posterior
chain = sample(model, NUTS(;adtype=AutoReverseDiff()), 1000; progress=true)

# plot posterior distributions and retrodiction
save_folder = "figures/"*simulation_code
plot_chains(chain, save_folder*"/chains")
plot_retrodiction(data=data,prob=prob,chain=chain,timepoints=timepoints,path=save_folder*"/retrodiction",seed=seed,seed_bayesian=true)

# hypothesis testing
prior_alpha = priors["α"]
posterior_alpha = KernelDensity.kde(vec(chain[:α]), boundary=(0,40))
savage_dickey_density = pdf(posterior_alpha,0.) / pdf(prior_alpha, 0.)
println("Probability of aggregation model: $(1 - savage_dickey_density / (savage_dickey_density+1))")

# save chain, model, priors, data threshold, and hypothesis test 
inference = Dict("chain" => chain, "priors" => priors, "model" => model, "data_threshold" => data_threshold, "savage_dickey_density" => savage_dickey_density)
serialize("simulations/"*simulation_code*".jls", inference)
