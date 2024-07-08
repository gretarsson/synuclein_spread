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
Read data 
=#
W = readdlm("data/W.csv",',')
L = laplacian_out(W)
LT = transpose(L)  # to use col vecs and Lx, we use the transpose
N = size(LT,1)

# find seed iCP
labels = readdlm("data/W_labeled.csv",',')[1,:]
seed = findall(x->x=="iCP",labels)[1]
#=
Define, simulate, and plot model
=#

# Define network diffusion model.
function diffusion(du,u,p,t;L=LT)
    ρ = p[1]
    du .= -ρ*L*u  
end

# Define initial-value problem.
alg = Tsit5()
u0 = [0. for i in 1:N]  # initial conditions
u0[80] = 10  # seed
p = [0.075]
tspan = (0.0,9.0)
prob = ODEProblem(diffusion, u0, tspan, p)
sol = solve(prob,alg)
StatsPlots.plot(sol; legend=false, ylim=(0,1))

#=
read data
=#
file_data = "data/avg_total_path.csv"
file_time = "data/timepoints.csv"
data = readdlm(file_data, ',')
timepoints = vec(readdlm(file_time, ','))
plt = StatsPlots.plot()
for i in 1:N
    StatsPlots.plot!(plt,timepoints, data[i,:], legend=false)
end
plt

#=
Bayesian estimation of model parameters
=#
# priors (some data points are NaN, set them to zero at t=0)
u0_prior_avg = data[:,1]
for i in 1:N
    if isnan(u0_prior_avg[i])
        u0_prior_avg[i] = 0.
    end
end
# create map for indices that are not NaN
proper_idxs = [[] for _ in 1:length(timepoints)]
for j in axes(data,2), i in axes(data,1)
    if !isnan(data[i,j])
        append!(proper_idxs[j], i)
    end
end
# std of IC prior
u0_prior_std = [0.1 for i in 1:N]
u0_prior_avg = [0. for i in 1:N]
u0_prior_avg[seed] = 10.
u0_prior_std[seed] = 1.

@model function fitlv(data, prob; alg=alg, u0_prior_avg=u0_prior_avg, u0_prior_std=u0_prior_std, timepoints=timepoints, proper_idxs=proper_idxs)
    # Prior distributions.
    σ ~ InverseGamma(2, 3)
    ρ ~ truncated(Normal(1.0, 0.1); lower=0.)
    #u0 ~ arraydist([truncated(Normal(u0_prior_avg[i], u0_prior_std[i]); lower=0) for i in 1:N])

    # Simulate diffusion model 
    p = [ρ]
    sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)) 
    predicted = solve(prob, alg; u0=u0_prior_avg, p=p, saveat=timepoints, sensealg=sensealg)

    # Observations.
    for j in axes(data,2)  # 1.2s / 2.1s 
        for i in proper_idxs[j]  # using pre-defined indices is quicker
            data[i,j] ~ Normal(predicted[i,j], σ^2)
        end
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
    adbackends=[:forwarddiff, :reversediff]
)

# Sample to approximate posterior
chain = sample(model, NUTS(;adtype=AutoReverseDiff()), 1000; progress=true)
chain_plot = StatsPlots.plot(chain)
save("figures/diffusion_inference/diffusion_data/chain_plot.png",chain_plot)

# plot individual posterior
N_pars = size(chain)[2]
vars = chain.info.varname_to_symbol
i = 1
for (key,value) in vars
    println(key)
    chain_i = Chains(chain[:,i,:], [value])
    chain_plot_i = StatsPlots.plot(chain_i)
    save("figures/diffusion_inference/diffusion_data/chain_$(i).png",chain_plot_i)
    i += 1
end

#=
Data retrodiciton
=#
fs = Any[NaN for i in 1:N]
axs = Any[NaN for i in 1:N]
for i in 1:N
    f = Figure()
    ax = Axis(f[1,1], title="Region $(i)", ylabel="Portion of cells infected", xlabel="time (months)", xticks=0:9, limits=(0,9.1,nothing,nothing))
    fs[i] = f
    axs[i] = ax
end
posterior_samples = sample(chain, 300; replace=false)
for sample in eachrow(Array(posterior_samples))
    ρ = sample[2]
    #u0 = sample[3:end]
    sol_p = solve(prob, alg; p=ρ, u0=u0_prior_avg, saveat=0.1)
    #sol_p = solve(prob, alg; p=ρ, saveat=0.1)
    for i in 1:N
        lines!(axs[i],sol_p.t, sol_p[i,:]; alpha=0.3, color=:grey)
    end
end

# Plot simulation and noisy observations.
for i in 1:N
    scatter!(axs[i], timepoints, data[i,:], colormap=:tab10)
    save("figures/diffusion_inference/diffusion_data/retrodiction_region_$(i).png", fs[i])
end



