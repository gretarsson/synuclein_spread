using DifferentialEquations 
using StatsPlots


include("helpers.jl")
data_threshold = 0.15

#=
read pathology data
=#
data, idxs = read_data("data/avg_total_path.csv", remove_nans=true, threshold=data_threshold)
timepoints = vec(readdlm("data/timepoints.csv", ','))
N = size(data)[1]

#=
Read structural data 
=#
W = readdlm("data/W.csv",',')[idxs,idxs]
LT = transpose(laplacian_out(W))  # transpose of Laplacian (so we can write LT * x, instead of x^T * L)
labels = readdlm("data/W_labeled.csv",',')[1,2:end][idxs]
seed = findall(x->x=="iCP",labels)[1]  # find index of seed region

# testing a split function
function f1(du,u,p,t;L=LT)
    ρ = p[1]

    x = u[1:N]
    du[1:N] .= -ρ*L*x   
    du[(N+1):2*N] .= 0   
end
function f2(du,u,p,t)
    α = p[2]
    β = p[3:(N+2)]
    d = p[(N+3):(2*N+2)]
    γ = p[end]

    x = u[1:N]
    y = u[(N+1):end]
    du[1:N] .= α .* x .* (β .- d .* y .- x)  
    du[(N+1):2*N] .= γ .* (1 .- y)   
end


split = SplitFunction(f1,f2)
#semilinear = SplitFunction(laplacian,f2)
# ODE settings
alg = KenCarp4()
tspan = (0.0,9.0)
# ICs
u0 = [0. for i in 1:(2*N)]  # initial conditions
u0[seed] = 0.1  # seed
# Parameters
ρ = 0.001
α = 3
#α = [rand(truncated(Normal(10.,0.5))) for _ in 1:N]
β = [1. for i in 1:N]
d = [0.5 for i in 1:N]
γ = 0.01
p = [ρ, α, β..., d..., γ]
split_prob = SplitODEProblem(split,u0,tspan)
sol = solve(split_prob,alg; abstol=1e-9, reltol=1e-9,p=p)
StatsPlots.plot(sol; legend=false)