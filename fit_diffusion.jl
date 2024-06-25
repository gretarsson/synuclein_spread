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
using TuringBenchmarking
using ReverseDiff
using Zygote
using SciMLSensitivity

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
alg_stiff = TRBDF2()
max_iter = Int(10^8)
u0 = [1. for i in 1:N]  # initial conditions
u0[1] = 1  # seed
p = [0.075]
tspan = (0.0,9.0)
prob = ODEProblem(diffusion, u0, tspan, p)

# Plot simulation.
#@btime solve(prob,alg, abstol=1e-4, reltol=1e-2)
plot(solve(prob, alg; saveat=timepoints), legend=false)

#=
read data
=#
file_data = "C:/Users/cga32/Desktop/synuclein_spread/data/avg_total_path.csv"
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
# priors
u0_prior_avg = data[:,1]
u0_prior_std = [0.05 for i in 1:N]

@model function fitlv(data, prob; alg=alg, timestep=timestep, u0_prior_avg=u0_prior_avg, u0_prior_std=u0_prior_std, timepoints=timepoints)
    # Prior distributions.
    σ ~ InverseGamma(2, 3)
    ρ ~ truncated(Normal(1.0, 0.1); lower=0., upper=Inf)
    u0 ~ MvNormal(u0_prior_avg, u0_prior_std)

    # Simulate diffusion model 
    p = [ρ]
    predicted = solve(prob, alg; u0=u0, p=p, saveat=timepoints, abstol=1e-4, reltol=1e-2)

    # Observations.
    for j in axes(data,2), i in axes(data,1)
        data[i,j] ~ Normal(predicted[i,j], σ^2)
    end
    #for i in 1:length(predicted)
    #    data[:,i] ~ MvNormal(predicted[i], σ^2 * I)
    #end

    return nothing
end

model = fitlv(odedata, prob)

# benchmarking 
#benchmark_model(
#    model;
#    # Check correctness of computations
#    check=true,
#    # Automatic differentiation backends to check and benchmark
#    adbackends=[:forwarddiff, :reversediff, :reversediff_compiled, :zygote]
#)
# with priors on ICs (450 parameters), AutoReverseDiff(true) is much better than AutoForwardDiff

# Sample to approximate posterior
chain = sample(model, NUTS(;adtype=AutoReverseDiff()), 1000; progress=true)
chain_plot = StatsPlots.plot(chain)
save("figures/diffusion_inference/diffusion_toy_struct/chain_plot.png",chain_plot)

# plot individual posterior
N_pars = size(chain)[2]
vars = chain.info.varname_to_symbol
i = 1
for (key,value) in vars
    println(key)
    chain_i = Chains(chain[:,i,:], [value])
    chain_plot_i = StatsPlots.plot(chain_i)
    save("figures/diffusion_inference/diffusion_toy_struct/chain_$(i).png",chain_plot_i)
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
    u0 = sample[3:end]
    sol_p = solve(prob, alg; p=ρ, u0=u0, saveat=0.1)
    #sol_p = solve(prob, alg; p=ρ, saveat=0.1)
    for i in 1:N
        lines!(axs[i],sol_p.t, sol_p[i,:]; alpha=0.3, color=:grey)
    end
end

# Plot simulation and noisy observations.
ground_truth = solve(prob, alg; saveat=0.1)  # ground truth
for i in 1:N
    lines!(axs[i], ground_truth.t, ground_truth[i,:]; linewidth=1, alpha=1., colormap=:tab10)
    scatter!(axs[i], sol.t, odedata[i,:], colormap=:tab10)
    save("figures/diffusion_inference/diffusion_toy_struct/retrodiction_region_$(i).png", fs[i])
end



