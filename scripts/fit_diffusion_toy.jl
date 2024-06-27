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

# Set a seed for reproducibility.
using Random
Random.seed!(16);

#=
Read data 
=#
#L = readdlm("data/L_out.csv",',')
#LT = transpose(L)  # to use col vecs and Lx, we use the transpose
#N = size(L,1)

# create random Laplacian
N = 5
W = rand([0,1], N, N)
#W = W*W'
for i in 1:N
    W[i,i] = 0
end
D = zeros(N,N)
for i in 1:N
    D[i,i] = sum(W[i,:])
end
L = D - W
LT = transpose(L)


#=
Define, simulate, and plot model
=#

# Define network diffusion model.
function diffusion(du,u,p,t;L=LT)
    ρ = p
    du .= -ρ*L*u  # u is a row vector
end

# Define initial-value problem.
alg = TRBDF2()
u0 = [0.0 for i in 1:N]  # initial conditions
u0[1] = 1  # seed
p = 1
tspan = (0.0,9.0)
prob = ODEProblem(diffusion, u0, tspan, p)

# Plot simulation.
plot(solve(prob, alg), legend=false)


#=
create synthetic data from simulation
=#
timestep = 1.
variance = 0.02
sol = solve(prob, alg; saveat=timestep)
odedata = Array(sol) + variance * randn(size(Array(sol)))

# Plot simulation and noisy observations.
plt = plot(sol; alpha=0.7)
for i in 1:N
    scatter!(sol.t, odedata[i,:])
end
plt

#=
Bayesian estimation of model parameters
=#
# priors
u0_prior_avg = odedata[:,1]
u0_prior_std = [0.1 for i in 1:N]

@model function fitlv(data, prob)
    # Prior distributions.
    σ ~ InverseGamma(2, 3)
    ρ ~ Normal(0, 0.5)
    u0 ~ MvNormal(u0_prior_avg, u0_prior_std)

    # Simulate diffusion model 
    p = ρ
    predicted = solve(prob, alg; u0=u0, p=p, saveat=timestep)

    # Observations.
    for i in 1:length(predicted)
        data[:, i] ~ MvNormal(predicted[i], σ^2 * I)
    end

    return nothing
end

model = fitlv(odedata, prob)

# Sample 3 independent chains with forward-mode automatic differentiation (the default).
chain = sample(model, NUTS(), MCMCSerial(), 1000, 3; progress=false)
chain_plot = StatsPlots.plot(chain)
save("figures/diffusion_inference/diffusion_toy/chain_plot.png",chain_plot)


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
    for i in 1:N
        lines!(axs[i],sol_p.t, sol_p[i,:]; alpha=0.3, color=:grey)
    end
end

# Plot simulation and noisy observations.
ground_truth = solve(prob, alg; saveat=0.1)  # ground truth
for i in 1:N
    lines!(axs[i], ground_truth.t, ground_truth[i,:]; linewidth=1, alpha=1., colormap=:tab10)
    scatter!(axs[i], sol.t, odedata[i,:], colormap=:tab10)
    save("figures/diffusion_inference/diffusion_toy/retrodiction_region_$(i).png", fs[i])
end



