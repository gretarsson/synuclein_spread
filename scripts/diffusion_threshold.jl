using DifferentialEquations
using DelimitedFiles
using Plots
using SparseArrays
using BenchmarkTools
using LinearAlgebra
include("helpers.jl")
#=
Here we simulate pure-diffusion for
a range of thresholding levels of the structural connectome
and compute the mean squared error to the non-thresholded solution
-> 50% seems like a good threshold level
=#

# find zeroth eigenvector (steady-state)
file_W = "data/W.csv"
W = transpose(readdlm(file_W, ','))
N = size(W, 1)
L = laplacian_out(W)
LT = transpose(L)

eigval = eigvals(LT)
eigvec = eigvecs(LT)
zeroth_ind = findall(abs.(eigval) .< 1e-6)[1]
zerovec = real.(eigvec[:,zeroth_ind])  # the steady-state vector
zerovec = zerovec ./ sum(zerovec)  # normalize so sum equals 1 (IC = 1)

# read strutural connectome
thresholds = LinRange(0.,0.99,100)
M = N  # number of random pars and inital conditions
mses = zeros(M,length(thresholds))
ss_diffs = zeros(length(thresholds))
hh = 0
for j in eachindex(thresholds)
    # compute thresholded Laplacian
    threshold = thresholds[j]
    Wt = threshold_matrix(W,threshold)
    L_sparse = laplacian_out(Wt)
    LT_sparse = transpose(L_sparse)
    # compare eigenvalues
    eigval_sp = eigvals(LT_sparse)
    eigvec_sp = eigvecs(LT_sparse)
    zeroth_ind_sp = findall(abs.(eigval_sp) .< 1e-6)[1]
    if hh == 0  && length(findall(abs.(eigval_sp) .< 1e-6)) > 1
        display("More than one zero-eigenvalue found for thresholds>$(thresholds[j])")
        hh = threshold
    end
    zerovec_sp = real.(eigvec_sp[:,zeroth_ind_sp])  # the steady-state vector
    zerovec_sp = zerovec_sp ./ sum(zerovec_sp)
    if zerovec_sp[1] < 0
        zerovec_sp = -1 .* zerovec_sp
    end
    #ss_diff = norm(zerovec .- zerovec_sp)
    ss_diff = abs(maximum(zerovec .- zerovec_sp))
    #ss_diff = maximum((zerovec .- zerovec_sp).^2)
    ss_diffs[j] = ss_diff
    # randomize rho and plot mse in time-series
    for m in 1:M
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
        #u0 = rand(N)  # pick random IC
        u0 = [0. for _ in 1:N]  # go through each node as a seed
        u0[m] = 1  # 79 is the empirical seed
        #p = 0.01*rand()
        p = rand(LogNormal(0,1))
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
        mses[m,j] = mse

    end
end

# plot normalized mean squared error
mean_mses = mean(eachrow(mses))
vars_mses = std(eachrow(mses))
fig = Plots.plot(thresholds,mean_mses; legend=false, color=:tab10, yerror=vars_mses, label=nothing, ylims=(-1,maximum(mean_mses .+ vars_mses)))
Plots.vline!([hh], color=palette(:tab10)[4], label="critical threshold = $(hh)", legend=true)
Plots.ylabel!("Normalized mean squared error")
Plots.xlabel!("Thresholding level")
savefig(fig, "figures/thresholding/mse_struct_threshold.pdf")

# plot 0-eigenvector distances
fig2 = Plots.plot(thresholds,ss_diffs; legend=false, color=:tab10)
#Plots.scatter!(thresholds,ss_diffs;legend=false, color=:tab10)
Plots.vline!([hh],color=palette(:tab10)[4])
Plots.ylabel!("0-eigenvector max. difference")
Plots.xlabel!("Thresholding level")
savefig(fig2, "figures/thresholding/mse_struct_threshold_eigvec.pdf")



