using DifferentialEquations

# test out SIR model with structural data

include("helpers.jl")

# Set name for files to be saved in figures/ and simulations/
data_threshold = 0.0

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
data = data ./ 2

#=
Read structural data 
=#
W = readdlm("data/W.csv",',')[idxs,idxs]
W = Matrix(transpose(W))
labels = readdlm("data/W_labeled.csv",',')[1,2:end][idxs]
seed = findall(x->x=="iCP",labels)[1]  # find index of seed region
W = W ./ maximum(W[W .> 0])
for i in 1:N
    W[i,i] = 0
end
W = W ./ maximum(W[W .> 0])
W = (W,N)

#=
Define, simulate, and plot model
=#
function sir_test(du,u,p,t;W=W)
    W,N = W
    τ = p[1]
    γ = p[2:(N+1)]
    θ = p[(N+2):(2*N+1)]
    ϵ = p[end]


    x = u[1:N]
    y = u[(N+1):(2*N)]
    du[1:N] .= (ϵ*W*x) .* (1 .- y .- x) .+ τ .* x .* (1 .- γ .- y .- x) 
    du[(N+1):(2*N)] .=  θ .* x  
end
# ODE settings
alg = Tsit5()
tspan = (0.0,9.0)
# ICs
x0 = [0. for i in 1:N]  # initial conditions
x0[seed] = 1e-2  # seed
y0 = [0. for i in 1:N]
u0 = [x0...,y0...]
# Parameters
τ = 5
γ = [0.1 for _ in 1:N]
θ = [0. for _ in 1:N]
ϵ = 0.1
p0 = [τ, γ..., θ..., ϵ]
p = 1. * p0
# setting up, solve, and plot
prob = ODEProblem(sir_test, u0, tspan, p; alg=alg, maxiters=1000)
sol = solve(prob,alg; abstol=1e-6, reltol=1e-3)
StatsPlots.plot(sol; legend=false, ylim=(0,1), idxs=1:N)
