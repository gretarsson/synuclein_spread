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
file_data = "data/avg_total_path.csv"
file_time = "data/timepoints.csv"
data = readdlm(file_data, ',')
N = size(data)[1]
timepoints = vec(readdlm(file_time, ','))
plt = StatsPlots.plot()
for i in 1:N
    StatsPlots.plot!(plt,timepoints, data[i,:], legend=false)
end
plt

# prune data for NaN
# create map of rows with no NaNs (ignoring regions with any NaN)
nonnan_idxs = nonnan_rows(data)
larger_idxs = larger_rows(data,-0.1)
idxs = nonnan_idxs .* larger_idxs
data = data[idxs,:]
N = size(data)[1]
println(sum(idxs))
plt = StatsPlots.plot()
StatsPlots.plot(timepoints, transpose(data), legend=false, ylim=(0,1))

#=
Read structural data 
=#
W = readdlm("data/W.csv",',')
W = W[idxs,idxs]
L = laplacian_out(W)
LT = transpose(L)  # to use col vecs and Lx, we use the transpose
N = size(LT,1)

# find seed iCP
labels = readdlm("data/W_labeled.csv",',')[1,2:end]
labels = labels[idxs]
seed = findall(x->x=="iCP",labels)[1]
#=
Define, simulate, and plot model
=#

# Define network diffusion model.
function aggregation(du,u,p,t;L=LT)
    ρ = p[1]
    α = p[2]
    β = p[3:end]

    du .= -ρ*L*u .+ α .* u .* (β .- u)  
end

# Define initial-value problem.
alg = Tsit5()
u0 = [0. for i in 1:N]  # initial conditions
u0[seed] = 1e-5  # seed
ρ = 0.01
α = 25.
β = [data[i,end] for i in 1:N]
p = [ρ, α, β...]
tspan = (0.0,9.0)
prob = ODEProblem(aggregation, u0, tspan, p)
sol = solve(prob,alg; abstol=1e-9, reltol=1e-6)
StatsPlots.plot(sol; legend=false, ylim=(0,1))


#=
Bayesian estimation of model parameters
=#
β0 = [data[i,end] for i in 1:N]
u0_seed = data[seed,1]

#u0 = [0. for _ in 1:N]
@model function fitlv(data, prob; alg=alg, timepoints=timepoints, seed=seed)
    # Prior on model parameters
    σ ~ InverseGamma(2, 3)
    ρ ~ truncated(Normal(0., 2.5); lower=0.)  # (0.,2.5)
    α ~ truncated(Normal(0., 2.5); lower=0.)  # (0.,2.5)
    β ~ arraydist([truncated(Normal(β0[i], 0.5); lower=0.) for i in 1:N])  # vector
    
    # Prior on initial conditions 
    u0 = [0. for _ in 1:N]
    u0[seed] = 1e-5  # (seed) 

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
#chain_plot = StatsPlots.plot(chain)
#save("figures/aggregation_inference/aggregation_data/chain_plot.png",chain_plot)

# plot individual posterior
N_pars = size(chain)[2]
vars = chain.info.varname_to_symbol
i = 1
for (key,value) in vars
    chain_i = Chains(chain[:,i,:], [value])
    chain_plot_i = StatsPlots.plot(chain_i)
    save("figures/aggregation_inference/aggregation_data/chains/chain_$(key).png",chain_plot_i)
    i += 1
end

#=
Data retrodiciton
=#
fs = Any[NaN for i in 1:N]
axs = Any[NaN for i in 1:N]
for i in 1:N
    f = Figure()
    ax = Axis(f[1,1], title="Region $(i)", ylabel="Portion of cells infected", xlabel="time (months)", xticks=0:9, limits=(0,9.1,0.,1.))
    fs[i] = f
    axs[i] = ax
end
posterior_samples = sample(chain, 300; replace=false)
for sample in eachrow(Array(posterior_samples))
    # samples
    ρ = sample[2]
    α = sample[3]
    β = sample[4:end]
    # IC
    u0 = [0. for _ in 1:N]
    u0[seed] = 1e-5
    # solve
    sol_p = solve(prob, alg; p=[ρ,α,β...], u0=u0, saveat=0.1)
    for i in 1:N
        lines!(axs[i],sol_p.t, sol_p[i,:]; alpha=0.3, color=:grey)
    end
end

# Plot simulation and noisy observations.
for i in 1:N
    scatter!(axs[i], timepoints, data[i,:]; colormap=:tab10)
    save("figures/aggregation_inference/aggregation_data/retrodiction/retrodiction_region_$(i).png", fs[i])
end
