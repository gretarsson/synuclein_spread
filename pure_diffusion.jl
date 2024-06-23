using DifferentialEquations
using DelimitedFiles
using Plots
using SparseArrays
using BenchmarkTools
include("helpers.jl")

# create random Laplacian
N = 10
L_rand = random_laplacian(N)

# read strutural connectome
file_W = "C:/Users/cga32/Desktop/synuclein_spread/data/W.csv"
W = readdlm(file_W, ',')
N = size(W, 1)
W = threshold_matrix(W,0.9)
L = laplacian_out(W)
LT = transpose(L)
LT_sparse = sparse(LT)

# Define network diffusion model.
function diffusion(du,u,p,t;L=LT_sparse)
    ρ = p
    du .= -ρ*L*u  
end

# settings for ODE
u0 = [0.0 for i in 1:N]  # initial conditions
u0[1] = N  # seed
p = 0.025
tspan = (0.0,9.0)

# run ODE
prob = ODEProblem(diffusion,u0, tspan,p)
@btime solve(prob, Tsit5())
sol = solve(prob, Tsit5())
Plots.plot(sol)



