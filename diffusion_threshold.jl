using DifferentialEquations
using DelimitedFiles
using Plots
using SparseArrays
using BenchmarkTools
include("helpers.jl")
#=
Here we simulate pure-diffusion for
a range of thresholding levels of the structural connectome
and compute the mean squared error to the non-thresholded solution
-> 50% seems like a good threshold level
=#

# create random Laplacian
N = 10
L_rand = random_laplacian(N)

# read strutural connectome
thresholds = LinRange(0.,0.99,20)
mses = []
for j in eachindex(thresholds)
    threshold = thresholds[j]
    file_W = "C:/Users/cga32/Desktop/synuclein_spread/data/W.csv"
    W = readdlm(file_W, ',')
    N = size(W, 1)
    Wt = threshold_matrix(W,threshold)

    L = laplacian_out(W)
    LT = transpose(L)

    L_sparse = laplacian_out(Wt)
    LT_sparse = transpose(L_sparse)

    # Define network diffusion model.
    function diffusion(du,u,p,t;L=LT)
        ρ = p
        du .= -ρ*L*u  
    end
    function diffusion_sparse(du,u,p,t;L=LT_sparse)
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
    prob_sparse = ODEProblem(diffusion_sparse,u0, tspan,p)
    #@btime solve(prob, Tsit5())
    sol = solve(prob, Tsit5(); saveat=0.1)
    sol_sparse = solve(prob_sparse, Tsit5(); saveat=0.1)
    Plots.plot(sol, idxs=[2])
    Plots.plot!(sol_sparse, idxs=[2])

    mse = 0
    for i in 1:N
        mse = mse + sum(sol[i,:] .- sol_sparse[i,:])^2 
    end
    mse = mse / (N * length(sol[i,:]))
    #println(mse)
    append!(mses,mse)
end
println(mses)

Plots.plot(thresholds,mses; legend=false, color=:tab10)
ylims!((0,1.))
ylabel!("mean squared error")
xlabel!("Thresholding level %")
