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

# read strutural connectome
thresholds = LinRange(0.,0.99,20)
M = 25  # number of random pars and inital conditions
mses = zeros(M,length(thresholds))
for j in eachindex(thresholds)
    for m in 1:M
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
        u0 = rand(N)  # initial conditions
        p = 0.01*rand()
        tspan = (0.0,9.0)

        # run ODE
        prob = ODEProblem(diffusion,u0, tspan,p)
        prob_sparse = ODEProblem(diffusion_sparse,u0, tspan,p)
        sol = solve(prob, Tsit5(); saveat=0.1)
        sol_sparse = solve(prob_sparse, Tsit5(); saveat=0.1)

        mse = 0
        for i in 1:N
            mse = mse + sum(sol[i,:] .- sol_sparse[i,:])^2 / mean(sol[i,:]) 
        end
        mse = mse / (N * length(sol[1,:]))
        #println(mse)
        mses[m,j] = mse
    end
end
println(mses)

mean_mses = mean(eachrow(mses))
vars_mses = std(eachrow(mses))
fig = Plots.plot(thresholds,mean_mses; legend=false, color=:tab10, yerror=vars_mses)
ylims!((-Inf,0.1))
ylabel!("Normalized mean squared error")
xlabel!("Thresholding level")
savefig(fig, "figures/thresholding/mse_struct_threshold.pdf")

