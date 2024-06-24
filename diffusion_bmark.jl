using DifferentialEquations
using DelimitedFiles
using Plots
using SparseArrays
using BenchmarkTools
using Random
Random.seed!(16)
include("helpers.jl")
#=
Here we benchmark the numerical solving
of a connectome diffusion model with respect
to thresholding of the connectome etc...
=#

threshold = 0.75
file_W = "C:/Users/cga32/Desktop/synuclein_spread/data/W.csv"
W = readdlm(file_W, ',')
N = size(W, 1)
Wt = threshold_matrix(W,threshold)

L = laplacian_out(W)
LT = transpose(L)

Lt = laplacian_out(Wt)
LTt = transpose(Lt)
LTt_sparse = sparse(LTt)

nzs = Any[[] for _ in 1:N]
for i in 1:N
    nzs[i] = findall(x->x!=0,LTt[i,:])
    println(length(nzs[i]))
end


# Define network diffusion model.
function diffusion(du,u,p,t;L=LTt)
    ρ = p
    du .= -ρ*L*u  
end
function diffusion_sparse(du,u,p,t;L=LTt_sparse)
    ρ = p
    du .= -ρ*L*u  
end
function diffusion_custom(du,u,p,t;L=LTt,nzs=nzs)
    ρ = p
    for i in eachindex(u)
        nz = nzs[i]
        for j in eachindex(nz)
            du[i] += du[i] + L[i,j]*u[j]  
        end
    end
    du .= -ρ .* du
end

# settings for ODE
u0 = [1. for i in 1:N]  # initial conditions
p = 0.075
tspan = (0.0,9.0)

# run ODE
prob = ODEProblem(diffusion,u0, tspan,p)
prob_sparse = ODEProblem(diffusion_sparse,u0, tspan,p)
prob_custom = ODEProblem(diffusion_sparse,u0, tspan,p)
b1 = @benchmark solve(prob, Tsit5(); saveat=0.1)
b2 = @benchmark solve(prob_sparse, Tsit5(); saveat=0.1)
b3 = @benchmark solve(prob_custom, Tsit5(); saveat=0.1)
#plot(solve(prob_custom, Tsit5(); saveat=0.1); legend=false)
#plot!(solve(prob, Tsit5(); saveat=0.1); legend=false)

println("Original solver time $(minimum(b1))")
println("Sparse solver time $(minimum(b2))")
println("Custom solver time $(minimum(b3))")