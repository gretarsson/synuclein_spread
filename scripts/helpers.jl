using Turing
using DelimitedFiles
using StatsPlots
using DifferentialEquations
using LSODA
using Distributions
using TuringBenchmarking  
using ReverseDiff
using Enzyme
using Zygote
using SciMLSensitivity
using LinearAlgebra
using Serialization
using CairoMakie
using ParetoSmooth
using Random
using SparseArrays
using LazyArrays
using KernelDensity
using Plots
using DataFrames, StatsBase, GLM
using Plots
using LaTeXStrings
#=
Helper functions for the project
=#


#=
create Laplacian matrix based on out degrees
=#
function laplacian_out(W; self_loops=false, retro=false)
    if retro
        W = transpose(W)
    end
    N = length(W[1,:])
    if !self_loops
        for i in 1:N  # removing self-loops
            W[i,i] = 0
        end
    end
    # create Laplacian from struct. connectome
    D = zeros(N,N)  # out-degree matrix
    for i in 1:N
        D[i,i] = sum(W[i,:])
    end
    L = D - W 
    return L
end

#=
create random Laplacian
=#
function random_laplacian(N)
    # create random Laplacian
    W = rand([0,1], N, N)
    W = W*W'
    for i in 1:N
        W[i,i] = 0
    end
    D = zeros(N,N)
    for i in 1:N
        D[i,i] = sum(W[i,:])
    end
    L = D - W
    return L
end


#=
find average of 3D matrix from dimension 3 skipping missing
=#
function mean3(M)
    N1,N2,_ = size(M)
    avg_M = Array{Union{Float64,Missing}}(missing, N1, N2)
    for j in axes(M,2), i in axes(M,1)
        avg_M[i,j] = mean(skipmissing(M[i,j,:]))
    end
    avg_M[isnan.(avg_M)] .= missing
    return avg_M
end

#=
find average of 3D matrix from dimension 3 skipping missing
=#
function var3(M)
    N1,N2,_ = size(M)
    var_M = Array{Union{Float64,Missing}}(missing, N1, N2)
    for j in axes(M,2), i in axes(M,1)
        var_M[i,j] = var(skipmissing(M[i,j,:]))
    end
    return var_M
end

#=
threshold matrix by percentage
=#
function threshold_matrix(A,d)
    A_flat = sort(vec(A))
    index = Int(round(d*length(A_flat))) + 1
    if index > length(A_flat)
        index = length(A_flat)
    end
    threshold = A_flat[index]
    A_thresh = copy(A)
    A_thresh[A_thresh.<threshold] .= 0
    return A_thresh
end


#=
find NaN rows in matrix
=#
function nonnan_rows(A)
    nonnan_idxs = [true for _ in 1:size(A)[1]]  # boolean indices of rows without any NaNs
    for j in axes(A,2), i in axes(A,1)
        if isnan(A[i,j])
            nonnan_idxs[i] = false
        end
    end
    return nonnan_idxs
end
    
#=
find rows with maximum larger than a
=#
function larger_rows(A,a)
    larger_idxs = [false for _ in 1:size(A)[1]]
    for i in axes(A,1)
        if maximum(filter(!isnan,A[i,:])) >= a
            larger_idxs[i] = true
        end
    end
    return larger_idxs
end

#=
Prune data matrix with nonnan and larger
=#
function prune_data(A,a)
    nonnan_idxs = nonnan_rows(A)
    larger_idxs = larger_rows(A,a)
    idxs = nonnan_idxs .* larger_idxs
    return A[idxs,:]
end

#=
read and prune data if told to
=#
function read_data(path; remove_nans=false, threshold=0.)
    A = readdlm(path, ',')
    idxs = [true for _ in 1:size(A)[1]]
    if remove_nans
        idxs = idxs .* nonnan_rows(A)
    end
    idxs = idxs .* larger_rows(A,threshold)
    return A[idxs,:], idxs
end

#=
Compute ParetoSmooth.psis_loo from a model and chain
This requirse finding pointwise loglikelihoods and some formatting
=#
function compute_psis_loo(model,chain)
    n_pars = length(chain.info[1])
    loglikelihoods_dict = Turing.pointwise_loglikelihoods(model,chain[:,1:n_pars,:])
    loglikelihoods = permutedims(stack(collect(values(loglikelihoods_dict))),[3,1,2])
    elpd = psis_loo(loglikelihoods, source="mcmc")
    return elpd
end

# ----------------------------------------------------
# ODEs
# ----------------------------------------------------
function diffusion(du,u,p,t;L=L,factors=nothing)
    L, _ = L
    ρ = p[1]

    du .= -ρ*L*u 
end
function diffusion2(du,u,p,t;L=(La, Lr),factors=nothing)
    La, Lr = L
    ρa = p[1]
    ρr = p[2]

    #du .= -(ρa*La+ρr*Lr)*u   # this gives very slow gradient computation
    du .= -ρa*ρr*La*u - ρa*Lr*u   # quick gradient computation
end
function diffusion3(du,u,p,t;L=L,N=1::Int, factors=nothing)
    ρ = p[1]
    γ = p[2]
    x = u[1:N]
    y = u[(N+1):(2*N)]

    du[1:N] .= -ρ*L*x
    du[(N+1):(2*N)] .= γ .* tanh.(x) .-  γ .* y
end
function diffusion_pop2(du,u,p,t;L=(La,Lr,N), factors=nothing)
    La, Lr, N = L
    ρa = p[1]
    ρr = p[2]
    γ = p[3]
    x = u[1:N]
    y = u[(N+1):(2*N)]

    du[1:N] .= -ρa*La*x-ρr*Lr*x
    #du[(N+1):(2*N)] .= 1/γ .* tanh.(x) .-  1/γ .* y
    du[(N+1):(2*N)] .= 1/γ .* x .-  1/γ .* y
end
function aggregation(du,u,p,t;L=L,factors=(1.,1.))
    L, _ = L
    kα,kβ = factors 
    ρ = p[1]
    α = kα * p[2]
    β = kβ .* p[3:end]

    du .= -ρ*L*u .+ α .* u .* (β .- u)  
end
function aggregation2(du,u,p,t;L=L,factors=(1.,1.))
    La, Lr = L
    p = factors .* p
    ρa = p[1]
    ρr = p[2]
    α = p[3]
    β = p[4:end]

    du .= -ρa*ρr*La*u .- ρa*Lr*u .+ α .* u .* (β .- u)   # quick gradient computation
end
function aggregation2_localα(du,u,p,t;L=L,factors=(1.,1.))
    La, Lr = L
    N = length(u)
    p = factors .* p
    ρa = p[1]
    ρr = p[2]
    α = p[3:(N+2)]
    β = p[(N+3):end]

    du .= -ρa*ρr*La*u .- ρa*Lr*u .+ α .* u .* (β .- u)   # quick gradient computation
end
function aggregation_pop2(du,u,p,t;L=L,factors=(1.,1.))
    La, Lr, N = L   
    p = factors .* p
    ρa = p[1]
    ρr = p[2]
    α = p[3]
    γ = p[4]
    β = p[5:end]


    x = u[1:N]
    y = u[(N+1):(2*N)]
    du[1:N] .= -ρa*ρr*La*x .- ρa*Lr*x .+ α .* x .* (β .* (1 .- y) .- x)   # quick gradient computation
    du[(N+1):(2*N)] .=  γ .* (tanh.(x) .- y)  
end
function death_local2(du,u,p,t;L=L,factors=(1.,1.))
    La, Lr, N = L  
    p = factors .* p
    ρa = p[1]
    ρr = p[2]
    α = p[3]
    β = p[4:(N+3)]
    #d = p[(N+4):(2*N+3)]
    d = p[N+4]
    γ = p[end]
    #α = p[3:N+2]
    #β = p[N+3:(2*N+2)]
    #γ = p[(2*N+3):(3*N + 2)]
    #ϵ = p[end-1]
    #d = p[end]


    x = u[1:N]
    y = u[(N+1):(2*N)]
    du[1:N] .= -ρa*ρr*La*x .- ρa*Lr*x .+ α .* x .* (β .* (1 .- d.*y) .- x)   # quick gradient computation
    du[(N+1):(2*N)] .=  γ .* (tanh.(x) .- y)  
    #du[(N+1):(2*N)] .=  ϵ .* (γ .* x .- y)  
    #du[(N+1):(2*N)] .=  (tanh.(x) .- y) ./ γ
end
function death(du,u,p,t;L=L,factors=(1.,1.))
    L,N = L
    p = factors .* p
    ρ = p[1]
    α = p[2]
    β = p[3:(N+2)]
    d = p[(N+3):(2*N+2)]
    γ = p[end]

    x = u[1:N]
    y = u[(N+1):(2*N)]
    du[1:N] .= -ρ*L*x .+ α  .* x .* (β .- d .* y .- x)   # quick gradient computation
    du[(N+1):(2*N)] .=  γ .* (1 .- y)  
end
function death_simplified(du,u,p,t;L=L,factors=(1.,1.))
    L,N = L
    p = factors .* p
    ρ = p[1]
    α = p[2]
    β = p[3:(N+2)]
    γ = p[end]

    x = u[1:N]
    y = u[(N+1):(2*N)]
    du[1:N] .= -ρ*L*x .+ α  .* x .* (β .- β .* y .- x)   # quick gradient computation
    du[(N+1):(2*N)] .=  γ .* (1 .- y)  
end
function death_simplifiedii(du,u,p,t;L=L,factors=(1.,1.))
    L,N = L
    p = factors .* p
    ρ = p[1]
    α = p[2]
    β = p[3:(N+2)]
    d = p[(N+3):(2*N+2)]
    γ = p[2*N+3]

    x = u[1:N]
    y = u[(N+1):(2*N)]
    du[1:N] .= -ρ*L*x .+ α .* x .* (β .- β .* y .- d .* y .- x)   # quick gradient computation
    du[(N+1):(2*N)] .=  γ .* (1 .- y)  
end
function death_simplifiedii_bilateral(du,u,p,t;L=L,factors=(1.,1.),M=222)
    L,N = L
    p = factors .* p
    n = N - 2*M
    ρ = p[1]
    α = p[2]
    β = repeat(p[3:(M+2)],2)  # regions with bilateral twin
    β = vcat(β, p[(M+3):(M+n+2)])  # regions without bilateral twin
    d = repeat(p[(M+n+3):(2*M+n+2)],2)  # regions with bilateral twin
    d = vcat(d, p[(2*M+n+3):(2*M+2*n+2)])  # regions without bilateral twin
    γ = p[2*M+2*n+3]

    x = u[1:N]
    y = u[(N+1):(2*N)]
    du[1:N] .= -ρ*L*x .+ α .* x .* (β .- β .* y .- d .* y .- x)   # quick gradient computation
    du[(N+1):(2*N)] .=  γ .* (1 .- y)  
end
function death_simplifiedii_bilateral2(du,u,p,t;L=L,factors=(1.,1.))
    La,Lr,N = L
    M = Int(N/2)
    p = factors .* p
    ρa = p[1]
    ρr = p[2]
    α = p[3]
    β = repeat(p[4:(M+3)],2)
    d = repeat(p[(M+4):(2*M+3)],2)
    γ = p[2*M+4]

    x = u[1:N]
    y = u[(N+1):(2*N)]
    du[1:N] .= -ρa*La*x - ρr*Lr*x .+ α .* x .* (β .- β .* y .- d .* y .- x)   # quick gradient computation
    du[(N+1):(2*N)] .=  γ .* (1 .- y)  
end
function death_simplifiediii(du,u,p,t;L=L,factors=(1.,1.))
    L,N = L
    p = factors .* p
    ρ = p[1]
    α = p[2]
    β = p[3:(N+2)]
    γ = p[(N+3):end]

    x = u[1:N]
    y = u[(N+1):(2*N)]
    du[1:N] .= -ρ*L*x .+ α .* x .* (β .- β .* y .- x)   # quick gradient computation
    du[(N+1):(2*N)] .=  γ .* (1 .- y)  
end
function death_sis(du,u,p,t;L=L,factors=(1.,1.))
    L,N = L
    p = factors .* p
    ρ = p[1]
    α = p[2]
    β = p[3:(N+2)]
    d = p[(N+3):(2*N+2)]

    x = u[1:N]
    y = u[(N+1):(2*N)]
    du[1:N] .= -ρ*L*x .+ α  .* x .* (β .- y .- x)   # quick gradient computation
    du[(N+1):(2*N)] .=  d .* x  
end
function sis(du,u,p,t;L=W,factors=(1.,1.))
    W,N = L
    p = factors .* p
    #τ = p[1:N]
    #γ = p[(N+1):2*N]
    #ϵ = p[end]
    τ = p[1]
    γ = p[2:(N+1)]
    ϵ = p[end]


    x = u[1:N]
    du[1:N] .= ϵ*W*x .* (100 .- x) .+ τ .* x .* (100 .- γ ./ τ .- x)    
    #du[1:N] .= τ .* x .* (1 .- x) .- γ .* x   
end
function sir(du,u,p,t;L=W,factors=(1.,1.))
    W,N = L
    p = factors .* p
    τ = p[1:N]
    γ = p[(N+1):(2*N)]
    θ = p[(2*N+1):(3*N)]
    M = p[(3*N+1):(4*N)]
    #ϵ = p[end]


    x = u[1:N]
    y = u[(N+1):(2*N)]
    #du[1:N] .= ϵ*W*x .* (100 .- y .- x) .+ τ .* x .* (100 .- y .- x) .- (γ .+ θ) .* x   
    du[1:N] .= τ .* x .* (M .- y .- x) .- (γ .+ θ) .* x   
    du[(N+1):(2*N)] .=  θ .* x  
end
function death2(du,u,p,t;L=L,factors=(1.,1.))
    La, Lr, N = L  
    p = factors .* p
    ρa = p[1]
    ρr = p[2]
    α = p[3]
    β = p[4:(N+3)]
    d = p[(N+4):(2*N+3)]
    γ = p[end]


    x = u[1:N]
    y = u[(N+1):(2*N)]
    du[1:N] .= -ρa*ρr*La*x .- ρa*Lr*x .+ α  .* x .* (β .- d .* y .- x)   # quick gradient computation
    du[(N+1):(2*N)] .=  γ .* (1 .- y)  
end
function death_all_local2(du,u,p,t;L=L,factors=(1.,1.))
    La, Lr, N = L  
    p = factors .* p
    ρa = p[1]
    ρr = p[2]
    α = p[3:(N+2)]
    β = p[(N+3):(2*N+2)]
    d = p[(2*N+3):(3*N+2)]
    γ = p[(3*N+3):end]


    x = u[1:N]
    y = u[(N+1):(2*N)]
    du[1:N] .= -ρa*ρr*La*x .- ρa*Lr*x .+ α  .* x .* (β .- y .- x)   # quick gradient computation
    #du[(N+1):(2*N)] .=  γ .* (1 .- y)  
    du[(N+1):(2*N)] .=  γ .* (d .* x .- y)  
end
function death_superlocal2(du,u,p,t;L=L,factors=(1.,1.))
    La, Lr, N = L  
    p = factors .* p
    ρa = p[1]
    ρr = p[2]
    α = p[3]
    β = p[4:(N+3)]
    d = p[(N+4):(2*N+3)]
    γ = p[end]


    x = u[1:N]
    y = u[(N+1):(2*N)]
    #du[1:N] .= -ρa*ρr*La*x .- ρa*Lr*x .+ α  .* x .* (β .- d.*y .- x)   # quick gradient computation
    du[1:N] .= -ρa*La*x .- ρr*Lr*x .+ α  .* x .* (β .- d.*y .- x)   # quick gradient computation
    du[(N+1):(2*N)] .=  γ .* (1 .- y)  
    #du[(N+1):(2*N)] .=  γ .* (x .- y)  
end
function death_simplifiedii_clustered(du, u, p, t; L=L, partition_sizes=[1, 1], factors=(1., 1.))
    L, N_total = L  # Total length of the system
    K = length(partition_sizes)  # Number of clusters (based on partition_sizes)
    partition_sizes = Array{Int}(partition_sizes)  # Convert to array if necessary
    
    p = factors .* p  # Adjust parameters using the factors
    
    # Extract parameters for each cluster
    ρ = p[1]
    α = p[2:2+K-1]
    β = p[2+K:2+2*K-1]
    d = p[2+2*K:2+3*K-1]
    γ = p[2+3*K:end]

    # Initialize variables
    x = u[1:N_total]  # First N elements are x values
    y = u[(N_total+1):(2*N_total)]  # Second N elements are y values

    # Keep track of the current index offset
    start_idx = 1

    # Loop through each cluster and compute the gradients
    for k in 1:K
        # Determine the number of elements in this cluster (based on partition_sizes)
        n_cluster = partition_sizes[k]
        end_idx = start_idx + n_cluster - 1

        # Apply the ODE dynamics for the current cluster
        du[start_idx:end_idx] .= -ρ * L * x[start_idx:end_idx] .+ α[k] .* x[start_idx:end_idx] .* (β[start_idx:end_idx] .- β[start_idx:end_idx] .* y[start_idx:end_idx] .- d[start_idx:end_idx] .* y[start_idx:end_idx] .- x[start_idx:end_idx])  
        du[(N_total+start_idx):(N_total+end_idx)] .= γ[k] .* (1 .- y[start_idx:end_idx])  # y dynamics
        
        # Update the start index for the next cluster
        start_idx = end_idx + 1
    end
end


#=
a dictionary containing the ODE functions
=#
odes = Dict("diffusion" => diffusion, "diffusion2" => diffusion2, "diffusion3" => diffusion3, "diffusion_pop2" => diffusion_pop2, "aggregation" => aggregation, 
            "aggregation2" => aggregation2, "aggregation_pop2" => aggregation_pop2, "death_local2" => death_local2, "aggregation2_localα" => aggregation2_localα,
            "death_superlocal2" => death_superlocal2, "death2" => death2, "death_all_local2" => death_all_local2, "death" => death, "sir" => sir, "sis" => sis, 
            "death_simplified" => death_simplified,
            "death_simplifiedii" => death_simplifiedii,
            "death_simplifiedii_clustered" => death_simplifiedii_clustered,
            "death_simplifiedii_bilateral" => death_simplifiedii_bilateral,
            "death_simplifiedii_bilateral2" => death_simplifiedii_bilateral2,
            "death_simplifiediii" => death_simplifiediii)
            



# ----------------------------------------------------------------------------------------------------------------------------------------
# Run whole simulations in one place
# the Priors dict must contain the ODE parameters in order first, and then σ. Other priors can then follow after, with seed always last.
# ----------------------------------------------------------------------------------------------------------------------------------------
function infer(ode, priors::OrderedDict, data::Array{Union{Missing,Float64},3}, timepoints::Vector{Float64}, W_file; 
               u0=[]::Vector{Float64},
               idxs=Vector{Int}()::Vector{Int},
               n_threads=1,
               alg=Tsit5(), 
               sensealg=ForwardDiffSensitivity(), 
               adtype=AutoForwardDiff(), 
               factors=[1.]::Vector{Float64},
               bayesian_seed=false,
               seed_region="iCP"::String,
               seed_value=1.::Float64,
               sol_idxs=Vector{Int}()::Vector{Int},
               abstol=1e-10, 
               reltol=1e-10,
               benchmark=false,
               benchmark_ad=[:forwarddiff, :reversediff, :reversediff_compiled],
               test_typestable=false,
               retro=true,
               M=0::Int
               )
    # verify that choice of ODE is correct wrp to retro- and anterograde
    retro_and_antero = false
    if occursin("2",string(ode))
        retro_and_antero = true 
        display("Model includes both retrograde and anterograde transport.")
    else 
        display("Model includes only retrograde transport.")
    end
    if bayesian_seed
        display("Model is inferring seeding initial conditions")
    else
        display("Model has constant initial conditions")
    end

    # read structural data 
    W_labelled = readdlm(W_file,',')
    if isempty(idxs)
        idxs = [i for i in 1:(size(W_labelled)[1] - 1)]
    end
    W = W_labelled[2:end,2:end]
    W = W[idxs,idxs]
    W = W ./ maximum( W[ W .> 0 ] )  # normalize connecivity by its maximum
    L = Matrix(transpose(laplacian_out(W; self_loops=false, retro=retro)))  # transpose of Laplacian (so we can write LT * x, instead of x^T * L)
    labels = W_labelled[1,2:end][idxs]
    seed = findall(x->x==seed_region,labels)[1]::Int  # find index of seed region
    display("Seed at region $(seed)")
    N = size(L)[1]
    if retro_and_antero  # include both Laplacians, if told to
        Lr = copy(L)
        La = Matrix(transpose(laplacian_out(W; self_loops=false, retro=!retro)))  
        if occursin("death",string(ode)) || occursin("pop",string(ode))
            L = (La,Lr,N)
        else
            L = (La,Lr)
        end
    else
        L = (L,N)
    end
    data = data[idxs,:,:]  # subindex data (idxs defaults to all regions unless told otherwise)

    # find number of ode parameters by looking at prior dictionary
    ks = collect(keys(priors))
    N_pars = findall(x->x=="σ",ks)[1] - 1

    # Define prob
    p = zeros(Float64, N_pars)
    tspan = (timepoints[1],timepoints[end])
    if string(ode) == "sis" || string(ode) == "sir"
        for i in 1:N
            W[i,i] = 0
        end
        #W = (Matrix(transpose(W)),N)  # transposing gives bad results
        L = (Matrix(W),N)  # not transposing gives excellent results
    end
    # ------
    # SDE
    # ------
    #function stochastic!(du,u,p,t)
    #        du[1:N] .= 0.0001
    #        du[(N+1):(2*N)] .= 0.0001
    #end
    #prob = SDEProblem(rhs, stochastic!, u0, tspan, p; alg=alg)
    # ------
    # ------
    #if M>0
    #    rhs(du,u,p,t;L=L, func=ode::Function) = func(du,u,p,t;L=L,factors=factors,M=M)
    #else
    #    rhs(du,u,p,t;L=L, func=ode::Function) = func(du,u,p,t;L=L,factors=factors)
    #end
    #function rhs(du, u, p, t; L=L, func=ode::Function, factors=factors, M=M)
    #    if M > 0
    #        return func(du, u, p, t; L=L, factors=factors, M=M)
    #    else
    #        return func(du, u, p, t; L=L, factors=factors)
    #    end
    #end
    rhs(du,u,p,t;L=L, func=ode::Function) = func(du,u,p,t;L=L,factors=factors,M=M)
    prob = ODEProblem(rhs, u0, tspan, p; alg=alg)
    
    # prior vector from ordered dic
    priors_vec = collect(values(priors))
    if isempty(sol_idxs)
        sol_idxs = [i for i in 1:N]
    end

    # reshape data into vector and find indices that are not of type missing
    N_samples = size(data)[3]
    vec_data = vec(data)
    nonmissing = findall(vec_data .!== missing)
    vec_data = vec_data[nonmissing]
    vec_data = identity.(vec_data)  # this changes the type from Union{Missing,Float64}Y to Float64

    ## make data array with rows that only have their (uniquely sized) nonmissing columns
    row_data = Vector{Vector{Float64}}([[] for _ in 1:size(data)[1]])
    row_nonmiss = Vector{Vector{Int}}([[] for _ in 1:size(data)[1]])
    for i in axes(data,1)
        data_subarray = vec(data[i,:,:])
        nonmissing_i = findall(data_subarray .!== missing)
        row_nonmiss[i] = identity.(nonmissing_i)
        row_data[i] = identity.(data_subarray[nonmissing_i])
    end

    # check if N_pars and length(factors) 
    if N_pars !== length(factors)
        display("Warning: The factor vector has length $(length(factors)) but the number of parameters is $(N_pars) according to the prior dictionary. Quitting...")
        return nothing
    end
    # check if global or regional variance
    global_variance::Bool = false
    if length(priors["σ"]) == 1
        global_variance = true
    end

    # use correct data type dependent on whether regional or global variance (for optimal Turing model specification)
    if global_variance
        final_data = vec_data
    else
        final_data = row_data
    end

    # EXP
    # -------
    # Organize data according to Markov chain type inference
    #data2 = create_data2(data)
    #N_samples = Int.(ones(size(data2)))
    #for i in 1:size(data2)[1]
    #    for j in 1:size(data2)[2]
    #        N_samples[i,j] = Int(size(data2[i,j])[1])
    #    end
    #end
    #data1 = [Float64[] for _ in 1:size(data2)[2]]
    #for j in 1:size(data2)[2]
    #    elm = []
    #    for i in 1:size(data2)[1]
    #        append!(elm,data2[i,j])
    #    end
    #    data1[j] = elm
    #end
    # -------
    #M = Int(length(idxs)/2)  # with bilateral
    #display("M = $(M)")
    #μ = zeros(M)
    #Σ = I(M)

    @model function bayesian_model(data, prob; ode_priors=priors_vec, priors=priors, alg=alg, timepointss=timepoints::Vector{Float64}, seedd=seed::Int, u0=u0::Vector{Float64}, bayesian_seed=bayesian_seed::Bool, seed_value=seed_value,
                                    N_samples=N_samples,
                                    nonmissing=nonmissing::Vector{Int64},
                                    row_nonmiss=row_nonmiss::Vector{Vector{Int}}
                                    )
        u00 = u0  # IC needs to be defined within model to work
        # priors
        p ~ arraydist([ode_priors[i] for i in 1:N_pars])
        # Have to set priors directly for beta-dependent d parameters
        # -------------------------
        #ρ ~ truncated(Normal(0,0.1),lower=0) 
        #α ~ filldist(truncated(Normal(0,1),lower=0),M);
        #β ~ filldist(truncated(Normal(0,1),lower=0),M);
        ##d ~ arraydist([truncated(Normal(-β[i],1), lower=-β[i], upper=0) for i in 1:M]);
        #d ~ filldist(Normal(-0.8,1),M);
        #γ ~ filldist(truncated(Normal(0,0.1),lower=0),M);
        # ------- try non truncated
        #ρ ~ truncated(Normal(0,0.1),lower=0) 
        #α ~ MvNormal(μ,Σ);
        #β ~ MvNormal(μ,Σ);
        ##d ~ arraydist([truncated(Normal(-β[i],1), lower=-β[i], upper=0) for i in 1:M]);
        #d ~ MvNormal(μ,Σ);
        #γ ~ MvNormal(μ,Σ);
        # -------------------------
        σ ~ priors["σ"] 
        if bayesian_seed
            u00[seedd] ~ priors["seed"]  
        else
            u00[seedd] = seed_value
        end

        # Simulate diffusion model 
        predicted = solve(prob, alg; u0=u00, p=p, saveat=timepointss, sensealg=sensealg, abstol=abstol, reltol=reltol, maxiters=6000)

        # pick out the variables of interest (ignore auxiliary variables)
        predicted = predicted[sol_idxs,:]

        # package predictions to match observation (when vectorizing data)
        predicted = vec(cat([predicted for _ in 1:N_samples]...,dims=3))
        predicted = predicted[nonmissing]
        data ~ MvNormal(predicted,σ^2*I) 
        return nothing
    end

    # define Turing model
    model = bayesian_model(final_data, prob)  # OG
    #model = bayesian_model(data1, prob)  # EXP Markov chain

    # test if typestable if told to, red marking in read-out means something is unstable
    #if test_typestable
    #    @code_warntype model.f(
    #        model,
    #        Turing.VarInfo(model),
    #        Turing.SamplingContext(
    #            Random.GLOBAL_RNG, Turing.SampleFromPrior(), Turing.DefaultContext(),
    #        ),
    #        model.args...,
    #    )
    #end

    # benchmark
    if benchmark
        suite = TuringBenchmarking.make_turing_suite(model;adbackends=benchmark_ad)
        println(run(suite))
        return nothing
    end

    # Sample to approximate posterior
    if n_threads == 1
        #chain = sample(model, NUTS(1000,0.65;adtype=adtype), 1000; progress=true, initial_params=[0.01 for _ in 1:(2*N+4)])  # time estimated is shown
        chain = sample(model, NUTS(10000,0.65;adtype=adtype), 1000; progress=true)  
        #chain = sample(model, HMC(0.05,10), 1000; progress=true)
    else
        chain = sample(model, NUTS(10000,0.65;adtype=adtype), MCMCDistributed(), 1000, n_threads; progress=true)
        #chain = sample(model, HMC(0.05,10), MCMCThreads(), 1000, n_threads; progress=true)
    end

    # compute elpd (expected log predictive density)
    elpd = compute_psis_loo(model,chain)
    waic = elpd.estimates[2,1] - elpd.estimates[3,1]  # total naive elpd - total p_eff

    # rescale the parameters in the chain and prior distributions
    factor_matrix = diagm(factors)
    n_chains = size(chain)[3]
    for i in 1:n_chains
        chain[:,1:N_pars,i] = Array(chain[:,1:N_pars,i]) * factor_matrix
    end
    i = 1
    for (key,value) in priors
        if i <= N_pars
            priors[key] = value * factors[i]
            i += 1
        else
            break
        end
    end
    # rename parameters
    chain = replacenames(chain, "u00[$(seed)]" => "seed")

    # save chains and metadata to a dictionary
    inference = Dict("chain" => chain, 
                     "priors" => priors, 
                     "data" => data,
                     "timepoints" => timepoints,
                     "data_indices" => idxs, 
                     "seed_idx" => seed,
                     "bayesian_seed" => bayesian_seed,
                     "seed_value" => seed_value,
                     "transform_observable" => false,
                     "ode" => string(ode),  # store var name of ode (functions cannot be saved)
                     "factors" => factors,
                     "sol_idxs" => sol_idxs,
                     "u0" => u0,
                     "L" => L,
                     "labels" => labels,
                     #"elpd" => elpd,
                     "waic" => waic
                     )

    return inference
end

#=
find regions with more than n number of measurements, from 3D data array
=#
function measured_regions(A,n)
    regions = [];
    for region in axes(A,1)
        region_i = A[region,:,:]
        nonmissing = findall(region_i .!== missing)
        N_nonmissing = length(nonmissing)
        if N_nonmissing > n
            push!(regions, region)
        end
    end
    return regions
end



#=
plot predicted vs observed plot for inference, parameters chosen from posterior mode
=#
using Makie
function predicted_observed(inference; save_path="", plotscale=log10)
    # try creating folder to save in
    try
        mkdir(save_path) 
    catch
    end

    # unpack from simulation
    fs = []  # storing figures
    chain = inference["chain"]
    data = inference["data"]
    timepoints = inference["timepoints"]
    seed = inference["seed_idx"]
    L = inference["L"]
    priors = inference["priors"]
    sol_idxs = inference["sol_idxs"]

    ks = collect(keys(priors))
    N_pars = findall(x->x=="σ",ks)[1] - 1
    factors = [1. for _ in 1:N_pars]
    ode = odes[inference["ode"]]
    N = size(data)[1]

    # simulate ODE from posterior mode
    # initialize
    tspan = (0., timepoints[end])
    u0 = inference["u0"]
    rhs(du,u,p,t;L=L, func=ode::Function) = func(du,u,p,t;L=L,factors=factors)
    prob = ODEProblem(rhs, u0, tspan; alg=Tsit5())

    # find posterior mode
    n_pars = length(chain.info[1])
    _, argmax = findmax(chain[:lp])
    mode_pars = Array(chain[argmax[1], 1:n_pars, argmax[2]])
    p = mode_pars[1:N_pars]
    if inference["bayesian_seed"]
        u0[seed] = chain["seed"][argmax]  
    else
        u0[seed] = inference["seed_value"]  
    end


    # solve ODE
    sol = solve(prob,Tsit5(); p=p, u0=u0, saveat=timepoints, abstol=1e-9, reltol=1e-6)
    sol = Array(sol[sol_idxs,:])

    # plot
    xticks = ([1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1e-0], [L"$10^{-6}$", L"$10^{-5}$", L"$10^{-4}$", L"$10^{-3}$", L"$10^{-2}$", L"$10^{-1}$", L"$10^0$"])
    yticks = xticks
    f = CairoMakie.Figure()
    ax = CairoMakie.Axis(f[1,1], title="", ylabel="Predicted", xlabel="Observed", xscale=plotscale, yscale=plotscale, xticks=xticks, yticks=yticks)

    # as we are plotting log-log, we account for zeros in the data
    regions = 1:N
    if length(size(data)) > 2
        data = mean3(data)  # find mean of data 
        # remove regions with less than 3 measuremnts 
        region_idxs = measured_regions(data,3)
        data = data[region_idxs,:,:]
        regions = copy(region_idxs)
    end

    # set x and y ticks

    x = vec(copy(data))
    y = vec(copy(sol[regions,:]))
    nonmissing = findall(x .!== missing)
    x = x[nonmissing]
    y = y[nonmissing]
    minxy = min(minimum(x),minimum(y))
    if plotscale==log10 && ((sum(x .== 0) + sum(y .== 0)) > 0)  # if zeros present, add the smallest number in plot
        #minx = minimum(x[x.>0])  # change back to this if plots are weird also see below if statemetn
        #miny = minimum(y[y.>0])
        #minxy = min(minx, miny)
        minxy = minimum(x[x.>0])  # change minimum to minimum of data to avoid super low value i.e e-44 from sims
        x = x .+ minxy
        y = y .+ minxy
    end

    CairoMakie.scatter!(ax,x,y, alpha=0.5)
    maxxy = max(maximum(x), maximum(y))
    CairoMakie.lines!([minxy,maxxy],[minxy,maxxy], color=:grey, alpha=0.5)
    if !isempty(save_path)
        CairoMakie.save(save_path * "/predicted_observed_mode.png", f)
    end
    push!(fs,f)

    # plot at different time points
    for i in eachindex(timepoints)
        f = CairoMakie.Figure()
        ax = CairoMakie.Axis(f[1,1], title="t = $(timepoints[i])", ylabel="Predicted", xlabel="Observed", xscale=plotscale, yscale=plotscale, xticks=xticks, yticks=yticks)

        # as we are plotting log-log, we account for zeros in the data
        x = vec(copy(data[:,i]))
        y = vec(copy(sol[regions,i]))
        nonmissing = findall(x .!== missing)
        x = x[nonmissing]
        y = y[nonmissing]
        #if plotscale==log10 && ((sum(x .<= 0) + sum(y .<= 0)) > 0)  # if doesn't work change back to this
        if plotscale==log10 && ((sum(x .<= 1e-8) + sum(y .<= 1e-8)) > 0)  # if zeros (or very small) present, add the smallest number in plot
            x = x .+ minxy
            y = y .+ minxy
        end

        CairoMakie.scatter!(ax,x,y, alpha=0.5)
        CairoMakie.lines!([minxy,maxxy],[minxy,maxxy], color=:grey, alpha=0.5)
        if !isempty(save_path)
            CairoMakie.save(save_path * "/predicted_observed_mode_$(i).png", f)
        end
        push!(fs,f)
    end

    return fs
end

#=
plot chains of each parameter from inference
=#
function plot_chains(inference; save_path="")
    # try creating folder to save in
    try
        mkdir(save_path) 
    catch
    end
    chain = inference["chain"]
    vars = collect(keys(inference["priors"]))
    master_fig = StatsPlots.plot(chain) 
    chain_figs = []
    for (i,var) in enumerate(vars)
        chain_i = StatsPlots.plot(master_fig[i,1], title=var)
        if !isempty(save_path)
            savefig(chain_i, save_path*"/chain_$(var).png")
        end
        push!(chain_figs,chain_i)
    end
    return chain_figs
end

#=
plot posteriors of each parameter from inference
=#
function plot_posteriors(inference; save_path="")
    # try creating folder to save in
    try
        mkdir(save_path) 
    catch
    end
    chain = inference["chain"]
    vars = collect(keys(inference["priors"]))
    master_fig = StatsPlots.plot(chain) 
    #posterior_figs = []
    for (i,var) in enumerate(vars)
        posterior_i = StatsPlots.plot(master_fig[i,2], title=var)
        if !isempty(save_path)
            savefig(posterior_i, save_path*"/posterior_$(var).png")
        end
        StatsPlots.closeall()
        #push!(posterior_figs,posterior_i)
    end
    #return posterior_figs
    return nothing
end

#=
plot retrodictino from inference result
=#
function plot_retrodiction(inference; save_path=nothing, N_samples=1, show_variance=false)
    # try creating folder to save in
    try
        mkdir(save_path) 
    catch
    end
    # unload from simulation
    data = inference["data"]
    chain = inference["chain"]
    priors = inference["priors"]
    timepoints = inference["timepoints"]
    seed = inference["seed_idx"]
    sol_idxs = inference["sol_idxs"]
    L = inference["L"]
    ks = collect(keys(inference["priors"]))
    N_pars = findall(x->x=="σ",ks)[1] - 1
    factors = [1. for _ in 1:N_pars]
    ode = odes[inference["ode"]]
    N = size(data)[1]
    M = length(timepoints)
    par_names = chain.name_map.parameters
    if inference["bayesian_seed"]
        seed_ch_idx = findall(x->x==:seed,par_names)[1]  # TODO find index of chain programmatically
    end
    # if data is 3D, find mean
    if length(size(data)) > 2
        var_data = var3(data)
        mean_data = mean3(data)
    end

    # define ODE problem 
    u0 = inference["u0"]
    tspan = (0, timepoints[end])
    rhs(du,u,p,t;L=L, func=ode::Function) = func(du,u,p,t;L=L,factors=factors)
    prob = ODEProblem(rhs, u0, tspan; alg=Tsit5())

    fs = Any[NaN for _ in 1:N]
    axs = Any[NaN for _ in 1:N]
    for i in 1:N
        f = CairoMakie.Figure(fontsize=20)
        ax = CairoMakie.Axis(f[1,1], title="Region $(i)", ylabel="Percentage area with pathology", xlabel="time (months)", xticks=0:9, limits=(0,9.1,nothing,nothing))
        fs[i] = f
        axs[i] = ax
    end
    posterior_samples = sample(chain, N_samples; replace=false)
    for sample in eachrow(Array(posterior_samples))
        # samples
        p = sample[1:N_pars]  # first index is σ and last index is seed
        if inference["bayesian_seed"]
            u0[seed] = sample[seed_ch_idx]  # TODO: find seed index automatically
            #u0[seed] = sample[end]  # TODO: find seed index automatically
        else    
            u0[seed] = inference["seed_value"]
        end
        σ = sample[end-1]
        
        # solve
        sol_p = solve(prob,Tsit5(); p=p, u0=u0, saveat=0.1, abstol=1e-9, reltol=1e-6)
        t = sol_p.t
        sol_p = Array(sol_p[sol_idxs,:])
        for i in 1:N
            lower_bound = sol_p[i,:] .- σ
            upper_bound = sol_p[i,:] .+ σ
            if show_variance
                CairoMakie.band!(axs[i], t, lower_bound, upper_bound; color=(:grey,0.1))
                CairoMakie.lines!(axs[i],t, sol_p[i,:]; alpha=0.9, color=:black)
            end
            CairoMakie.lines!(axs[i],t, sol_p[i,:]; alpha=0.5, color=:grey)
        end
    end

    # Plot simulation and noisy observations.
    # plot mean and variance
    for i in 1:N
        # =-=----
        nonmissing = findall(mean_data[i,:] .!== missing)
        data_i = mean_data[i,:][nonmissing]
        timepoints_i = timepoints[nonmissing]
        var_data_i = var_data[i,:][nonmissing]

        # skip if mean is empty
        if isempty(data_i)
            continue
        end

        indices = findall(x -> isnan(x),var_data_i)
        var_data_i[indices] .= 0
        CairoMakie.scatter!(axs[i], timepoints_i, data_i; color=RGB(0/255, 0/255, 139/255), alpha=1., markersize=15)  
        # have lower std capped at 0.01 (to be visible in the plots)
        var_data_i_lower = copy(var_data_i)
        for (n,var) in enumerate(var_data_i)
            if sqrt(var) > data_i[n]
                var_data_i_lower[n] = max(data_i[n]^2-1e-5, 0)
                #var_data_i_lower[n] = data_i[n]^2
            end
        end

        #CairoMakie.errorbars!(axs[i], timepoints_i, data_i, sqrt.(var_data_i); color=RGB(0/255, 71/255, 171/255), whiskerwidth=20, alpha=0.2)
        CairoMakie.errorbars!(axs[i], timepoints_i, data_i, sqrt.(var_data_i_lower), sqrt.(var_data_i); color=RGB(0/255, 71/255, 171/255), whiskerwidth=20, alpha=0.2, linewidth=3)
        #CairoMakie.errorbars!(axs[i], timepoints_i, data_i, [0. for _ in 1:length(timepoints_i)], sqrt.(var_data_i); color=RGB(0/255, 71/255, 171/255), whiskerwidth=20, alpha=0.2)
    end
    # plot all data points across all samples
    for i in 1:N
        jiggle = rand(Normal(0,0.01),size(data)[3])
        for k in axes(data,3)
            # =-=----
            nonmissing = findall(data[i,:,k] .!== missing)
            data_i = data[i,:,k][nonmissing]
            timepoints_i = timepoints[nonmissing] .+ jiggle[k]
            CairoMakie.scatter!(axs[i], timepoints_i, data_i; color=RGB(0/255, 71/255, 171/255), alpha=0.4, markersize=15)  
        end
        CairoMakie.save(save_path * "/retrodiction_region_$(i).png", fs[i])
    end

    # we're done
    return fs
end

#=
plot priors from inference result
=#
function plot_priors(inference; save_path="")
    # try creating folder to save in
    try
        mkdir(save_path) 
    catch
    end
    # rescale the parameters according to the factor
    priors = inference["priors"]

    i = 1
    for (var, dist) in priors
        prior_i = StatsPlots.plot(dist, title=var, ylabel="Density", xlabel="Sample value", legend=false)
        if !isempty(save_path)
            savefig(prior_i, save_path*"/prior_$(var).png")
        end
        i += 1
        StatsPlots.closeall()
    end
    return nothing 
end

#=
plot priors and posteriors together from inference result
=#
function plot_prior_and_posterior(inference; save_path="")
    # try creating folder to save in
    try
        mkdir(save_path) 
    catch
    end
    # rescale the priors
    chain = inference["chain"]
    priors = inference["priors"]
    vars = collect(keys(priors))
    master_fig = StatsPlots.plot(chain) 
    for (i,var) in enumerate(vars)
        plot_i = StatsPlots.plot(master_fig[i,2], title=var)
        StatsPlots.plot!(plot_i, priors[var])
        if !isempty(save_path)
            savefig(plot_i, save_path*"/prior_and_posterior_$(var).png")
        end
        StatsPlots.closeall()
    end
    return nothing
end

#=
master plotting function (plot everything relevant to inference)
=#
function plot_inference(inference, save_path; plotscale=log10, N_samples=300, show_variance=false)
    # load inference simulation 
    #display(inference["chain"])

    # create folder
    try
        mkdir(save_path);
    catch
    end
    if !inference["bayesian_seed"]
        try
            delete!(inference["priors"], "seed")
        catch
        end
    end
    
    # plot
    predicted_observed(inference; save_path=save_path*"/predicted_observed", plotscale=plotscale);
    plot_retrodiction(inference; save_path=save_path*"/retrodiction", N_samples=N_samples, show_variance=show_variance);
    plot_prior_and_posterior(inference; save_path=save_path*"/prior_and_posterior");
    plot_posteriors(inference, save_path=save_path*"/posteriors");
    #plot_chains(inference, save_path=save_path*"/chains");
    #plot_priors(inference; save_path=save_path*"/priors");
    return nothing
end

#=
inform beta and d priors
=#
function inform_priors(data,sample_n)
    sample_n = 4;  # sample to inform prior
    maxima = Vector{Float64}(undef,N);
    endpoints = Vector{Float64}(undef,N);
    for region in axes(data,1)
        region_timeseries = data[region,:,sample_n]
        nonmissing = findall(region_timeseries .!== missing)
        region_timeseries = identity.(region_timeseries[nonmissing])
        if isempty(region_timeseries)
            maxima[region] = 0
            endpoints[region] = 0
        else
            maxima[region] = maximum(region_timeseries)
            if ismissing(region_timeseries[end])
                endpoints[region] = 0
            else
                endpoints[region] = region_timeseries[end]
            end
        end
    end
    sample_inds = filter(x->x!==sample_n,1:size(data)[3])
    data = data[:,:,sample_inds]  
    return data, maxima, endpoints
end

#=
give a dictionary of key to index
=#
function dictionary_map(vec)
    dict_map = Dict()
    for i in eachindex(vec)
        dict_map[vec[i]] = i
    end
    return dict_map
end

function lowest_positive(arr::AbstractArray)
    # Flatten the array to handle both vectors and multi-dimensional arrays
    flat_arr = vec(arr)
    
    # Filter out `missing` values (if any) and values less than or equal to zero
    filtered_values = filter(x -> (!ismissing(x)) && x > 0, flat_arr)
    
    # Check if there are any valid positive values
    if isempty(filtered_values)
        return missing  # Return `missing` if there are no positive values
    else
        return minimum(filtered_values)
    end
end


function create_data2(data::Array{Union{Missing, Float64}, 3})
    # Get the dimensions of the input 3D array
    dim1, dim2, dim3 = size(data)
    
    # Initialize an empty 2D array where each element is a vector of Float64
    data2 = Array{Vector{Float64}, 2}(undef, dim1, dim2)
    
    # Iterate over each (i, j) index in the 2D slice
    for i in 1:dim1
        for j in 1:dim2
            # Extract the 1D slice for data[i, j, :]
            slice = data[i, j, :]
            
            # Filter out the missing values and collect the remaining values
            data2[i, j] = collect(filter(x -> !ismissing(x), slice))
        end
    end
    
    return data2
end


# --- WAIC Computation ---
function compute_waic(inference; S=10)
    # unpack the inference object
    priors = inference["priors"]
    ks = collect(keys(priors))
    chain = inference["chain"]
    data = inference["data"]
    seed = inference["seed_idx"]
    ode = odes[inference["ode"]]
    u0 = inference["u0"]
    timepoints = inference["timepoints"]
    L = inference["L"]
    #if typeof(L) != Tuple{Matrix{Float64},Int64}
    #    N = size(L)[1]
    #    L = (L,N)
    #else
    #    N = size(L[1])[1]
    #end
    N = size(L[1])[1]
    factors = inference["factors"]
    N_pars = findall(x->x=="σ",ks)[1] - 1
    par_names = chain.name_map.parameters
    if inference["bayesian_seed"]
        seed_ch_idx = findall(x->x==:seed,par_names)[1]  
    end
    sigma_idx = findall(x->x==:σ,par_names)[1]  

    # reshape data
    N_samples = size(data)[3]
    vec_data = vec(data)
    nonmissing = findall(vec_data .!== missing)
    vec_data = vec_data[nonmissing]
    vec_data = identity.(vec_data)  # this changes the type from Union{Missing,Float64}Y to Float64
    n = length(vec_data)

    # define ODE problem
    tspan = (0, timepoints[end])
    rhs(du,u,p,t;L=L, func=ode::Function) = func(du,u,p,t;L=L,factors=factors)
    prob = ODEProblem(rhs, u0, tspan; alg=Tsit5())

    # sample from posterior and store solutions of ODE for each sample
    posterior_samples = sample(chain, S; replace=false)
    solutions = Vector{Vector{Float64}}()  # Array to hold matrices of size (N, length(timepoints))
    sigmas = []
    for sample in eachrow(Array(posterior_samples))
        # samples
        p = sample[1:N_pars]  
        if inference["bayesian_seed"]
            u0[seed] = sample[seed_ch_idx]  
        else    
            u0[seed] = inference["seed_value"]
        end
        # solve
        σ = sample[sigma_idx]  # this should be done programmatically
        sol_p = solve(prob,Tsit5(); p=p, u0=u0, saveat=timepoints, abstol=1e-9, reltol=1e-9)
        sol_p = Array(sol_p[inference["sol_idxs"],:])
        solution = vec(cat([sol_p for _ in 1:N_samples]...,dims=3))
        solution = solution[nonmissing]
        push!(solutions,solution)
        push!(sigmas,σ)
    end
    means = transpose(hcat(solutions...))  # Sxn matrix (S = number of posterior samples, n = length of data)
    #return means
    
    # Compute pointwise log-likelihoods
    log_lik = zeros(S, n)  # Posterior samples × data points
    for s in 1:S
        for i in 1:n
            log_lik[s, i] = logpdf(Normal(means[s,i], sigmas[s]), vec_data[i])
        end
    end

    # Compute WAIC components
    lppd = sum(log.(mean(exp.(log_lik), dims=1)))  # Log Pointwise Predictive Density
    p_waic = sum(var(log_lik, dims=1))           # Effective number of parameters
    waic = -2 * (lppd - p_waic)                  # WAIC formula
    #display("lppd: $(lppd), p_eff: $(p_waic)")

    return waic
end


# --- WAIC and WBIC Computation ---
function compute_waic_wbic(inference; S=10)
    # unpack the inference object
    priors = inference["priors"]
    ks = collect(keys(priors))
    chain = inference["chain"]
    data = inference["data"]
    seed = inference["seed_idx"]
    ode = odes[inference["ode"]]
    u0 = inference["u0"]
    timepoints = inference["timepoints"]
    L = inference["L"]
    # assume L is given as a tuple (matrix, _)
    N = size(L[1])[1]
    factors = inference["factors"]
    N_pars = findall(x->x=="σ",ks)[1] - 1
    par_names = chain.name_map.parameters
    if inference["bayesian_seed"]
        seed_ch_idx = findall(x->x==:seed,par_names)[1]  
    end
    sigma_idx = findall(x->x==:σ,par_names)[1]  

    # reshape data
    N_samples = size(data)[3]
    vec_data = vec(data)
    nonmissing = findall(vec_data .!== missing)
    vec_data = vec_data[nonmissing]
    vec_data = identity.(vec_data)  # converts Union{Missing,Float64} to Float64
    n = length(vec_data)

    # define ODE problem
    tspan = (0, timepoints[end])
    rhs(du,u,p,t;L=L, func=ode::Function) = func(du,u,p,t;L=L,factors=factors)
    prob = ODEProblem(rhs, u0, tspan; alg=Tsit5())

    # sample from posterior and store ODE solutions for each sample
    posterior_samples = sample(chain, S; replace=false)
    solutions = Vector{Vector{Float64}}()  # each element holds the solution vector for a sample
    sigmas = []
    for sample in eachrow(Array(posterior_samples))
        # Extract parameter samples
        p = sample[1:N_pars]  
        if inference["bayesian_seed"]
            u0[seed] = sample[seed_ch_idx]  
        else    
            u0[seed] = inference["seed_value"]
        end
        σ = sample[sigma_idx]  # extract σ
        # solve the ODE
        sol_p = solve(prob, Tsit5(); p=p, u0=u0, saveat=timepoints, abstol=1e-9, reltol=1e-9)
        sol_p = Array(sol_p[inference["sol_idxs"],:])
        # Repeat solution for each of the N_samples (if data has replicated observations)
        solution = vec(cat([sol_p for _ in 1:N_samples]..., dims=3))
        solution = solution[nonmissing]
        push!(solutions, solution)
        push!(sigmas, σ)
    end
    means = transpose(hcat(solutions...))  # S x n matrix

    # Compute pointwise log-likelihoods for each sample and data point
    log_lik = zeros(S, n)
    for s in 1:S
        for i in 1:n
            log_lik[s, i] = logpdf(Normal(means[s,i], sigmas[s]), vec_data[i])
        end
    end

    # --- WAIC computation ---
    lppd = sum(log.(mean(exp.(log_lik), dims=1)))   # Log Pointwise Predictive Density
    p_waic = sum(var(log_lik, dims=1))                # Effective number of parameters
    waic = -2 * (lppd - p_waic)                       # WAIC formula

    # --- WBIC computation ---
    # Compute total log-likelihood per posterior sample
    total_log_lik = vec(sum(log_lik, dims=2))  # vector of length S
    # Temperature for WBIC: T = 1 / log(n)
    T = 1 / log(n)
    # Compute log-weights to stabilize the exponentiation
    log_weights = (T - 1) * total_log_lik
    max_log_weight = maximum(log_weights)
    stabilized_weights = exp.(log_weights .- max_log_weight)
    # Compute weighted expectation of the log-likelihood
    wbic_expectation = sum(stabilized_weights .* total_log_lik) / sum(stabilized_weights)
    wbic = -2 * wbic_expectation

    return waic, wbic
end


function compute_aic_bic(inference)
    # unpack the inference object
    chain = inference["chain"]
    data = inference["data"]
    L = inference["L"]
    #if typeof(L) != Tuple{Matrix{Float64},Int64}
    #    N = size(L)[1]
    #    L = (L,N)
    #else
    #    N = size(L[1])[1]
    #end
    par_names = chain.name_map.parameters
    N_pars = length(par_names)

    # reshape data
    vec_data = vec(data)
    nonmissing = findall(vec_data .!== missing)
    vec_data = vec_data[nonmissing]
    vec_data = identity.(vec_data)  # this changes the type from Union{Missing,Float64}Y to Float64
    n = length(vec_data)

    # maximu likelihood
    max_lp, _ = findmax(chain[:lp])

    # Compute AIC & BIC
    #display("N parameters = $(N_pars), log(n) = $(log(n)), max_lp = $(max_lp)")
    aic = -2*max_lp + 2*N_pars
    bic = -2*max_lp + N_pars * log(n)
    return aic, bic
end

# find mode from a chain object
function posterior_mode(chain)
    N_pars = length(chain.name_map.parameters)
    mode = []
    for i in 1:N_pars
        par_samples = vec(chain[:,i,1])
        posterior_i = KernelDensity.kde(par_samples)
        mode_i = posterior_i.x[argmax(posterior_i.density)]
        append!(mode,mode_i)
    end
    return mode
end

# create a dictionary with keys from vector A where each key contains a list of elements in vector B containing the key
# That is, a map from vector A to "super-elements" in vector B
function submap(a,b)
    # create a map, indexing the regions in gene expression data and relating them to regions in the structural connectome
    subdict = Dict(); 
    for a_label in a
        sublabels = []
        for b_label in b
            #if occursin(label,W_label)
            if lowercase(a_label) == lowercase(b_label[2:end]) || occursin(lowercase(a_label)*"-",lowercase(b_label))
                push!(sublabels,b_label)
            end
        end
        subdict[a_label] = sublabels
    end
    return subdict
end

# special function for gene_expression_correlation.jl, here we find a parameter vector ordered the same way as the set of genes
function create_parameter_vector_genes(parameters,gene_to_struct,gene_region_labels,W_label_map)
    para_vector = Vector{Union{Float64,Missing}}(missing,length(gene_region_labels));
    for (k,gene_region) in enumerate(gene_region_labels)
        struct_regions = gene_to_struct[gene_region]
        model_pars = []
        if !isempty(struct_regions)
            for struct_region in struct_regions
                struct_index = W_label_map[struct_region]
                model_pars_i = parameters[struct_index] 
                push!(model_pars, model_pars_i)
            end
            #display(struct_regions)
            model_pars_subregion_avg = mean(model_pars)
            para_vector[k] = model_pars_subregion_avg
        end
    end
    return para_vector
end


# do multiple linear progression, show will display significant results after Bonferroni correction
function multiple_linear_regression(vect,matr;labels=nothing,alpha=0.05,show=false,save_plots=false)
    # do linear regression with Bonferroni
    alpha = 0.05  # significance threshold before correction
    lms = []
    pvals = []
    N_genes = size(matr)[2]
    for gene_index in axes(matr,2)
        df_gene = DataFrame(X=matr[:,gene_index], Y=vect)
        ols = lm(@formula(Y ~ X),df_gene)
        coeff_p = coeftable(ols).cols[4][2]
        if save_plots
            if coeff_p < (alpha / N_genes)
                #println("R^2: $(r2(ols)), corr p-value $(coeff_p*N_genes), gene name: $(labels[gene_index])")
                Plots.scatter(matr[:,gene_index],vect;title="$(labels[gene_index])", ylabel="gene expression", xlabel="modeling parameter", label="brain region")


                # Add fitted line
                x_vals = matr[:, gene_index]
                fitted_vals = predict(ols, df_gene)
                Plots.plot!(x_vals, fitted_vals; color=:red, label="Fitted line")

                Plots.savefig("figures/gene_correlation/correlation_$(gene_index).png")
            end
        end
        push!(lms,ols)
        push!(pvals,coeff_p)
    end
    return lms,pvals
end

# do full gene analysis, with Holm-Bonferroni correction
#function gene_analysis(simulation, parameter_symbol::String; mode=true, show=false, alpha=0.05, null=false, save_plots=false)
#    # read gene data
#    gene_data_full = readdlm("data/avg_Pangea_exp.csv",',');
#    gene_labels = gene_data_full[1,2:end];
#    gene_region_labels = identity.(gene_data_full[2:end,1])
#    gene_data = gene_data_full[2:end,2:end];  # region x gene
#    N_genes = size(gene_data)[2];
#
#    # find label indexing per the computational model / structural connectome
#    W_labels = readdlm("data/W_labeled.csv",',')[2:end,1];
#    W_label_map = dictionary_map(W_labels);
#    N = length(W_labels)
#
#    # read the inference file, and find indices for beta and decay parameters 
#    inference = deserialize(simulation);
#    chain = inference["chain"];
#    priors = inference["priors"];
#    parameter_names = collect(keys(priors))
#    if parameter == "d+β"
#        para_idxs1 = findall(key -> occursin("β[",key), parameter_names)
#        para_idxs2 = findall(key -> occursin("d[",key), parameter_names)
#    else
#        para_idxs = findall(key -> occursin(parameter_symbol*"[",key), parameter_names)
#    end
#
#    # find modes of parameters
#    if mode
#        if parameter == "d+β"
#            mode = posterior_mode(chain)
#            params = mode[para_idxs1] .+ mode[para_idxs2]
#        else
#            mode = posterior_mode(chain)
#            params = mode[para_idxs]
#        end
#    elseif parameter == "d+β"
#        params_all = Array(sample(chain,1))[1,vcat(para_idxs1,para_idxs2)]
#        params = params_all[1:N] .+ params_all[(N+1):end]
#    else
#        params = Array(sample(chain,1))[1,para_idxs]
#    end
#
#    # create a dictionary from gene labels to the connectome labels, and print out number of regions not found in connectome
#    gene_to_struct = submap(gene_region_labels,W_labels)
#    #regions_not_found = [];
#    #for (keys,value) in gene_to_struct
#    #    if isempty(value)
#    #        push!(regions_not_found,keys)
#    #    end
#    #end
#    #display("Warning: $(length(regions_not_found)) gene regions not found in connectome.")
#
#    # CORRELATION ANALYISIS
#    # ---------------------------------------------------------------------------------------
#    # create vector with parameter values in same order as genes, and average over regions in gene_to_struct[region]
#    para_vector = create_parameter_vector_genes(params,gene_to_struct,gene_region_labels,W_label_map)
#
#    # regions that are not found are "missing", find regions that we do have
#    nonmissing = findall(e -> !ismissing(e), para_vector)
#    para_vector = identity.(para_vector[nonmissing])
#    gene_matrix = identity.(gene_data[nonmissing,:])
#
#    # shuffle gene matrix if null
#    if null
#        # Shuffle each column (gene) independently
#        #shuffled_gene_data = similar(gene_matrix)  # Create a matrix of the same size
#        #for g in 1:size(gene_matrix)[2]
#        #    shuffled_gene_data[:, g] = shuffle(gene_matrix[:, g])
#        #end
#        #gene_matrix = shuffled_gene_data
#        # Shuffle all the same regions for each gene
#        gene_matrix = gene_matrix[shuffle(1:size(gene_matrix, 1)), :]
#        # Shuffle entire matrix
#        #gene_matrix = Random.shuffle(gene_matrix)
#        #para_vector = shuffle(para_vector)
#    end
#
#    # do multiple linear regression over genes
#    lms,pvals = multiple_linear_regression(para_vector,gene_matrix;labels=gene_labels,alpha=alpha,show=false, save_plots=save_plots);
#
#    # Holm-Bonferroni correction (less conservative than Bonferroni)
#    lmss = []
#    r2s = r2.(lms);
#    p_inds = sortperm(pvals);
#    significant = []
#    for (k,ind) in enumerate(p_inds)
#        push!(significant,ind)
#        push!(lmss,lms[ind])
#        if show
#            println("R^2: $(r2s[ind]), corr p-value $(pvals[ind]), gene name: $(gene_labels[ind])")
#            #display(Plots.scatter(gene_matrix[:,i],para_vector;title="$(gene_labels[i])"))
#        end
#        # only store significant results (OLD)
#        #if pvals[ind] <= alpha / (N_genes - (k-1))
#        #    push!(significant,ind)
#        #    push!(lmss,lms[ind])
#        #    if show
#        #        println("R^2: $(r2s[ind]), corr p-value $(pvals[ind]), gene name: $(gene_labels[ind])")
#        #        #display(Plots.scatter(gene_matrix[:,i],para_vector;title="$(gene_labels[i])"))
#        #    end
#        #else
#        #    break
#        #end
#    end
#    return (lmss,pvals,significant,gene_labels)
#end


function gene_analysis(simulation, parameter_symbol::String; mode=true, show=false, alpha=0.05, null=false, save_plots=false)
    # read gene data
    gene_data_full = readdlm("data/avg_Pangea_exp.csv",',');
    gene_labels = gene_data_full[1,2:end];
    gene_region_labels = identity.(gene_data_full[2:end,1])
    gene_data = gene_data_full[2:end,2:end];  # region x gene
    N_genes = size(gene_data)[2];

    # find label indexing per the computational model / structural connectome
    W_labels = readdlm("data/W_labeled.csv",',')[2:end,1];
    W_label_map = dictionary_map(W_labels);
    N = length(W_labels)

    # read the inference file, and find indices for beta and decay parameters 
    inference = deserialize(simulation);
    chain = inference["chain"];
    priors = inference["priors"];
    parameter_names = collect(keys(priors))
    if parameter == "d+β"
        para_idxs1 = findall(key -> occursin("β[",key), parameter_names)
        para_idxs2 = findall(key -> occursin("d[",key), parameter_names)
    else
        para_idxs = findall(key -> occursin(parameter_symbol*"[",key), parameter_names)
    end

    # find modes of parameters
    if mode
        if parameter == "d+β"
            mode = posterior_mode(chain)
            params = mode[para_idxs1] .+ mode[para_idxs2]
        else
            mode = posterior_mode(chain)
            params = mode[para_idxs]
        end
    elseif parameter == "d+β"
        params_all = Array(sample(chain,1))[1,vcat(para_idxs1,para_idxs2)]
        params = params_all[1:N] .+ params_all[(N+1):end]
    else
        params = Array(sample(chain,1))[1,para_idxs]
    end

    # create a dictionary from gene labels to the connectome labels, and print out number of regions not found in connectome
    gene_to_struct = submap(gene_region_labels,W_labels)

    # CORRELATION ANALYISIS
    # ---------------------------------------------------------------------------------------
    # create vector with parameter values in same order as genes, and average over regions in gene_to_struct[region]
    para_vector = create_parameter_vector_genes(params,gene_to_struct,gene_region_labels,W_label_map)

    # regions that are not found are "missing", find regions that we do have
    nonmissing = findall(e -> !ismissing(e), para_vector)
    para_vector = identity.(para_vector[nonmissing])
    gene_matrix = identity.(gene_data[nonmissing,:])

    # shuffle gene matrix if null
    if null
        gene_matrix = gene_matrix[shuffle(1:size(gene_matrix, 1)), :]
    end

    # do multiple linear regression over genes
    lms,pvals = multiple_linear_regression(para_vector,gene_matrix;labels=gene_labels,alpha=alpha,show=false, save_plots=save_plots);

    return lms
end


function get_rvalue(model)
    # Calculate R-squared
    r_squared = GLM.r2(model)
    
    # Get the slope of the model
    slope = GLM.coef(model)[2]

    # Calculate r-value
    r_value = sqrt(abs(r_squared)) * sign(slope)
    
    return r_value
end

function get_pvalue(model)
    # Calculate R-squared
    coeff_p = coeftable(model).cols[4][2]
    return coeff_p
end

function holm_bonferroni(p_values::Vector{Float64}, alpha::Float64 = 0.05)
    # Sort p-values and keep track of original indices
    sorted_indices = sortperm(p_values)
    sorted_pvals = p_values[sorted_indices]
    
    # Number of p-values
    m = length(p_values)
    
    # Apply Holm-Bonferroni correction
    corrected_pvals = [sorted_pvals[i] * (m - i + 1) for i in 1:m]
    
    # Determine significance
    significant = corrected_pvals .<= alpha
    
    # Find the first non-significant index
    first_non_significant = findfirst(!, significant)
    if isnothing(first_non_significant)
        # All p-values are significant
        significant_indices = sorted_indices
    else
        # Only return significant p-values up to the first non-significant one
        significant_indices = sorted_indices[1:first_non_significant - 1]
    end
    
    return significant_indices
end

# return only idxs that have a bilateral counterpart in its labels
function only_bilateral(labels)
    parsed_labels = [(s[1], s[2:end]) for s in labels]
    grouped = Dict(base => [side for (side, b) in parsed_labels if b == base] for base in unique(s[2] for s in parsed_labels))
    missing_twins = filter(base -> length(grouped[base]) != 2, keys(grouped))
    solo_idxs = []
    for twin in missing_twins
        push!(solo_idxs,findall(s -> s[2:end] == twin, labels)[1])
    end
    idxs = setdiff(1:448,solo_idxs)
    return idxs
end



# ----------------------------------------------------------------------------------------------------------------------------------------
# Run whole simulations in one place
# the Priors dict must contain the ODE parameters in order first, and then σ. Other priors can then follow after, with seed always last.
# ----------------------------------------------------------------------------------------------------------------------------------------
function infer_clustering(ode, priors::OrderedDict, data::Array{Union{Missing,Float64},3}, timepoints::Vector{Float64}, W_file; 
               u0=[]::Vector{Float64},
               idxs=Vector{Int}()::Vector{Int},
               n_threads=1,
               alg=Tsit5(), 
               sensealg=ForwardDiffSensitivity(), 
               adtype=AutoForwardDiff(), 
               factors=[1.]::Vector{Float64},
               bayesian_seed=false,
               seed_region="iCP"::String,
               seed_value=1.::Float64,
               sol_idxs=Vector{Int}()::Vector{Int},
               abstol=1e-10, 
               reltol=1e-10,
               benchmark=false,
               benchmark_ad=[:forwarddiff, :reversediff, :reversediff_compiled],
               test_typestable=false,
               K=2
               )
    # verify that choice of ODE is correct wrp to retro- and anterograde
    retro_and_antero = false
    if occursin("2",string(ode))
        retro_and_antero = true 
        display("Model includes both retrograde and anterograde transport.")
    else 
        display("Model includes only retrograde transport.")
    end
    if bayesian_seed
        display("Model is inferring seeding initial conditions")
    else
        display("Model has constant initial conditions")
    end

    # read structural data 
    W_labelled = readdlm(W_file,',')
    if isempty(idxs)
        idxs = [i for i in 1:(size(W_labelled)[1] - 1)]
    end
    W = W_labelled[2:end,2:end]
    W = W[idxs,idxs]
    W = W ./ maximum( W[ W .> 0 ] )  # normalize connecivity by its maximum
    L = Matrix(transpose(laplacian_out(W; self_loops=false, retro=true)))  # transpose of Laplacian (so we can write LT * x, instead of x^T * L)
    labels = W_labelled[1,2:end][idxs]
    seed = findall(x->x==seed_region,labels)[1]::Int  # find index of seed region
    display("Seed at region $(seed)")
    N = size(L)[1]
    if retro_and_antero  # include both Laplacians, if told to
        Lr = copy(L)
        La = Matrix(transpose(laplacian_out(W; self_loops=false, retro=false)))  
        if occursin("death",string(ode)) || occursin("pop",string(ode))
            L = (La,Lr,N)
        else
            L = (La,Lr)
        end
    else
        L = (L,N)
    end
    data = data[idxs,:,:]  # subindex data (idxs defaults to all regions unless told otherwise)

    # find number of ode parameters by looking at prior dictionary
    ks = collect(keys(priors))
    N_pars = findall(x->x=="σ",ks)[1] - 1

    # Define prob
    p = zeros(Float64, N_pars)
    tspan = (timepoints[1],timepoints[end])
    if string(ode) == "sis" || string(ode) == "sir"
        for i in 1:N
            W[i,i] = 0
        end
        #W = (Matrix(transpose(W)),N)  # transposing gives bad results
        L = (Matrix(W),N)  # not transposing gives excellent results
    end
    # ------
    # SDE
    # ------
    #function stochastic!(du,u,p,t)
    #        du[1:N] .= 0.0001
    #        du[(N+1):(2*N)] .= 0.0001
    #end
    #prob = SDEProblem(rhs, stochastic!, u0, tspan, p; alg=alg)
    # ------
    # ------
    rhs(du,u,p,t;L=L, func=ode::Function) = func(du,u,p,t;L=L,factors=factors)
    prob = ODEProblem(rhs, u0, tspan, p; alg=alg)
    
    # prior vector from ordered dic
    priors_vec = collect(values(priors))
    if isempty(sol_idxs)
        sol_idxs = [i for i in 1:N]
    end

    # reshape data into vector and find indices that are not of type missing
    N_samples = size(data)[3]
    vec_data = vec(data)
    nonmissing = findall(vec_data .!== missing)
    vec_data = vec_data[nonmissing]
    vec_data = identity.(vec_data)  # this changes the type from Union{Missing,Float64}Y to Float64

    ## make data array with rows that only have their (uniquely sized) nonmissing columns
    row_data = Vector{Vector{Float64}}([[] for _ in 1:size(data)[1]])
    row_nonmiss = Vector{Vector{Int}}([[] for _ in 1:size(data)[1]])
    for i in axes(data,1)
        data_subarray = vec(data[i,:,:])
        nonmissing_i = findall(data_subarray .!== missing)
        row_nonmiss[i] = identity.(nonmissing_i)
        row_data[i] = identity.(data_subarray[nonmissing_i])
    end

    # check if N_pars and length(factors) 
    if N_pars !== length(factors)
        display("Warning: The factor vector has length $(length(factors)) but the number of parameters is $(N_pars) according to the prior dictionary. Quitting...")
        return nothing
    end
    # check if global or regional variance
    global_variance::Bool = false
    if length(priors["σ"]) == 1
        global_variance = true
    end

    # use correct data type dependent on whether regional or global variance (for optimal Turing model specification)
    if global_variance
        final_data = vec_data
    else
        final_data = row_data
    end

    # EXP
    # -------
    # Organize data according to Markov chain type inference
    #data2 = create_data2(data)
    #N_samples = Int.(ones(size(data2)))
    #for i in 1:size(data2)[1]
    #    for j in 1:size(data2)[2]
    #        N_samples[i,j] = Int(size(data2[i,j])[1])
    #    end
    #end
    #data1 = [Float64[] for _ in 1:size(data2)[2]]
    #for j in 1:size(data2)[2]
    #    elm = []
    #    for i in 1:size(data2)[1]
    #        append!(elm,data2[i,j])
    #    end
    #    data1[j] = elm
    #end
    # -------
        
    function death_simplifiedii_clustered(du, u, p, t; L=L)
        L, N = L  # Extract Laplacian and system size
    
        # Extract parameters
        ρ = p[1]
        α = p[2:2+N-1]
        β = p[2+N:2+2*N-1]
        d = p[2+2*N:2+3*N-1]
        γ = p[2+3*N:end]
    
        # Extract state variables
        x = u[1:N]  
        y = u[(N+1):(2*N)]  
    
        # **Vectorized ODE Computation**
        du[1:N] .= -ρ .* (L * x) .+ α .* x .* (β .- β .* y .- d .* y .- x)
        du[(N+1):2*N] .= γ .* (1 .- y)
    end

    @model function bayesian_model_cluster(data, prob; K=2::Int, ode_priors=priors_vec, priors=priors, alg=alg, 
        timepointss=timepoints::Vector{Float64}, seedd=seed::Int, 
        u0=u0::Vector{Float64}, bayesian_seed=bayesian_seed::Bool, 
        seed_value=seed_value, N_samples=N_samples, 
        nonmissing=nonmissing::Vector{Int64}, 
        row_nonmiss=row_nonmiss::Vector{Vector{Int}})

        u00 = u0  # Initial conditions inside model
        K ~ DiscreteUniform(1,N)
        # test 
        #K = 40

        # Priors for parameters
        ρ ~ truncated(Normal(0, 0.1),lower=0)
        α ~ filldist(truncated(Normal(0, 0.1),lower=0),K)  # One α for each cluster
        β ~ filldist(truncated(Normal(0, 1),lower=0),K)  # One β for each cluster
        #d ~ filldist(truncated(Normal(0, 1), lower=-Inf),K)  # One d for each cluster
        d ~ arraydist([truncated(Normal(0, 1), lower=-β[k], upper=0) for k in 1:K])  # One d for each cluster, dependent on β
        γ ~ filldist(truncated(Normal(0, 0.1),lower=0),K)  # One γ for each cluster
        
        # Sample the partition indices (categorical distribution over K clusters)
        partition_weights = ones(K) ./ K  # Probabilities for each cluster
        partition_indices ~ filldist(Categorical(partition_weights), N)  # KxN regions
        # test
        #partition_indices = rand(1:K,N)
        # make masks to reorder parameters according to their right partitions
        partition_masks = [partition_indices .== k for k in 1:K]
        partition_masks = hcat(partition_masks...)  # NxK

        # **Precompute Weighted Parameter Values Without Indexing**
        αi = partition_masks * α
        βi = partition_masks * β
        di = partition_masks * d
        γi = partition_masks * γ

        # Create the parameter vector p which includes all priors
        p = vcat(ρ, αi, βi, di, γi)  # Combine the parameters into a single vector
   
        # sigma and IC
        σ ~ priors["σ"] 
        if bayesian_seed
            u00[seedd] ~ priors["seed"]  
        else
            u00[seedd] = seed_value
        end

        # Solve ODE system using **external function**
        # Solve ODE system
        prob = ODEProblem(death_simplifiedii_clustered, u00, tspan, p; alg=alg)
        predicted = solve(prob, alg; u0=u00, p=p, saveat=timepointss, sensealg=sensealg, 
        abstol=abstol, reltol=reltol, maxiters=6000)

        # Extract relevant variables
        predicted = predicted[sol_idxs, :]

        # Expand predictions for observed data
        predicted = vec(cat([predicted for _ in 1:N_samples]..., dims=3))
        predicted = predicted[nonmissing]

        # Likelihood
        data ~ MvNormal(predicted, σ^2 * I)

        return nothing
    end
    @model function bayesian_model_cluster_soft(data, prob; K=2::Int, ode_priors=priors_vec, priors=priors, alg=alg, 
        timepointss=timepoints::Vector{Float64}, seedd=seed::Int, 
        u0=u0::Vector{Float64}, bayesian_seed=bayesian_seed::Bool, 
        seed_value=seed_value, N_samples=N_samples, 
        nonmissing=nonmissing::Vector{Int64}, 
        row_nonmiss=row_nonmiss::Vector{Vector{Int}})

        u00 = u0  # Initial conditions inside model
        #K ~ DiscreteUniform(1,N)
        #K ~ Uniform(1,N)
        #K = ceil(Int,K)
        K = 10
        # test 
        #K = 40

        # Priors for parameters
        ρ ~ truncated(Normal(0, 0.1),lower=0)
        α ~ filldist(truncated(Normal(0, 0.1),lower=0),K)  # One α for each cluster
        β ~ filldist(truncated(Normal(0, 1),lower=0),K)  # One β for each cluster
        #d ~ filldist(truncated(Normal(0, 1), lower=-Inf),K)  # One d for each cluster
        d ~ arraydist([truncated(Normal(0,1), lower=-β[k], upper=0) for k in 1:K])  # One d for each cluster, dependent on β
        γ ~ filldist(truncated(Normal(0, 0.1),lower=0),K)  # One γ for each cluster
        
        # Sample the partition indices (categorical distribution over K clusters)
        partition_weights = ones(K)  # Probabilities for each cluster
        partition_indices ~ filldist(Dirichlet(partition_weights), N)  # N regions
        partition_mask = transpose(partition_indices)
        # test
        #partition_indices = rand(1:K,N)
        # make masks to reorder parameters according to their right partitions
        #partition_masks = [partition_indices .== k for k in 1:K]
        #partition_masks = hcat(partition_masks...)

        # **Precompute Weighted Parameter Values Without Indexing**
        αi = partition_mask * α
        βi = partition_mask * β
        di = partition_mask * d
        γi = partition_mask * γ

        # Create the parameter vector p which includes all priors
        p = vcat(ρ, αi, βi, di, γi)  # Combine the parameters into a single vector
   
        # sigma and IC
        σ ~ priors["σ"] 
        if bayesian_seed
            u00[seedd] ~ priors["seed"]  
        else
            u00[seedd] = seed_value
        end

        # Solve ODE system using **external function**
        # Solve ODE system
        prob = ODEProblem(death_simplifiedii_clustered, u00, tspan, p; alg=alg)
        predicted = solve(prob, alg; u0=u00, p=p, saveat=timepointss, sensealg=sensealg, 
        abstol=abstol, reltol=reltol, maxiters=6000)

        # Extract relevant variables
        predicted = predicted[sol_idxs, :]

        # Expand predictions for observed data
        predicted = vec(cat([predicted for _ in 1:N_samples]..., dims=3))
        predicted = predicted[nonmissing]

        # Likelihood
        data ~ MvNormal(predicted, σ^2 * I)

        return nothing
    end


    # define Turing model
    model = bayesian_model_cluster_soft(final_data, prob)  # OG
    #model = bayesian_model_cluster(final_data, prob)  # OG
    #model = bayesian_model(data1, prob)  # EXP Markov chain

    # test if typestable if told to, red marking in read-out means something is unstable
    #if test_typestable
    #    @code_warntype model.f(
    #        model,
    #        Turing.VarInfo(model),
    #        Turing.SamplingContext(
    #            Random.GLOBAL_RNG, Turing.SampleFromPrior(), Turing.DefaultContext(),
    #        ),
    #        model.args...,
    #    )
    #end

    # benchmark
    if benchmark
        suite = TuringBenchmarking.make_turing_suite(model;adbackends=benchmark_ad)
        println(run(suite))
        return nothing
    end

    # Sample to approximate posterior
    if n_threads == 1
        #chain = sample(model, NUTS(1000,0.65;adtype=adtype), 1000; progress=true, initial_params=[0.01 for _ in 1:(2*N+4)])  # time estimated is shown
        chain = sample(model, NUTS(1000,0.65;adtype=adtype), 1000; progress=true)  
        #chain = sample(model, HMC(0.05,10), 1000; progress=true)
    else
        chain = sample(model, NUTS(1000,0.65;adtype=adtype), MCMCDistributed(), 1000, n_threads; progress=true)
        #chain = sample(model, HMC(0.05,10), MCMCThreads(), 1000, n_threads; progress=true)
    end

    # compute elpd (expected log predictive density)
    #elpd = compute_psis_loo(model,chain)
    #waic = elpd.estimates[2,1] - elpd.estimates[3,1]  # total naive elpd - total p_eff

    # rescale the parameters in the chain and prior distributions
    factor_matrix = diagm(factors)
    n_chains = size(chain)[3]
    for i in 1:n_chains
        chain[:,1:N_pars,i] = Array(chain[:,1:N_pars,i]) * factor_matrix
    end
    i = 1
    for (key,value) in priors
        if i <= N_pars
            priors[key] = value * factors[i]
            i += 1
        else
            break
        end
    end
    # rename parameters
    chain = replacenames(chain, "u00[$(seed)]" => "seed")

    # save chains and metadata to a dictionary
    inference = Dict("chain" => chain, 
                     "priors" => priors, 
                     "data" => data,
                     "timepoints" => timepoints,
                     "data_indices" => idxs, 
                     "seed_idx" => seed,
                     "bayesian_seed" => bayesian_seed,
                     "seed_value" => seed_value,
                     "transform_observable" => false,
                     "ode" => string(ode),  # store var name of ode (functions cannot be saved)
                     "factors" => factors,
                     "sol_idxs" => sol_idxs,
                     "u0" => u0,
                     "L" => L,
                     "labels" => labels
                     #"elpd" => elpd,
                     #"waic" => waic
                     )

    return inference
end



function plot_retrodiction_clustered(inference; save_path=nothing, N_samples=1, show_variance=false)
    # try creating folder to save in
    try
        mkdir(save_path) 
    catch
    end
    # unload from simulation
    data = inference["data"]
    chain = inference["chain"]
    priors = inference["priors"]
    timepoints = inference["timepoints"]
    seed = inference["seed_idx"]
    sol_idxs = inference["sol_idxs"]
    L = inference["L"]
    ks = collect(keys(inference["priors"]))
    N_pars = findall(x->x=="σ",ks)[1] - 1
    factors = [1. for _ in 1:N_pars]
    ode = odes[inference["ode"]]
    N = size(data)[1]
    M = length(timepoints)
    par_names = chain.name_map.parameters
    if inference["bayesian_seed"]
        seed_ch_idx = findall(x->x==:seed,par_names)[1]  # TODO find index of chain programmatically
    end
    # if data is 3D, find mean
    if length(size(data)) > 2
        var_data = var3(data)
        mean_data = mean3(data)
    end

    # define ODE problem 
    u0 = inference["u0"]
    tspan = (0, timepoints[end])
    rhs(du,u,p,t;L=L, func=ode::Function) = func(du,u,p,t;L=L)
    prob = ODEProblem(rhs, u0, tspan; alg=Tsit5())

    fs = Any[NaN for _ in 1:N]
    axs = Any[NaN for _ in 1:N]
    for i in 1:N
        f = CairoMakie.Figure(fontsize=20)
        ax = CairoMakie.Axis(f[1,1], title="Region $(i)", ylabel="Percentage area with pathology", xlabel="time (months)", xticks=0:9, limits=(0,9.1,nothing,nothing))
        fs[i] = f
        axs[i] = ax
    end
    posterior_samples = sample(chain, N_samples; replace=false)
    posterior_table = Tables.columntable(posterior_samples)
    for (s,sample) in enumerate(eachrow(Array(posterior_samples)))
        alpha_params = filter(name -> startswith(string(name), "α"), names(chain))
        K = length(alpha_params)

        # Extract the partition posteriors
        ρ = posterior_table[:ρ][s]
        αP = [posterior_table[Symbol("α[$i]")][s] for i in 1:K]
        βP = [posterior_table[Symbol("β[$i]")][s] for i in 1:K]
        dP = [posterior_table[Symbol("d[$i]")][s] for i in 1:K]
        γP = [posterior_table[Symbol("γ[$i]")][s] for i in 1:K]

        # sample each region's parameter as a combination of partition posteriors
        α = [0. for _ in 1:N]
        β = [0. for _ in 1:N]
        d = [0. for _ in 1:N]
        γ = [0. for _ in 1:N]
        for i in 1:N
            partition_weight = [posterior_table[Symbol("partition_indices[$k, $i]")][s] for k in 1:K]
            α[i] = dot(partition_weight, αP)
            β[i] = dot(partition_weight, βP)
            d[i] = dot(partition_weight, dP)
            γ[i] = dot(partition_weight, γP)
        end
        function death_simplifiedii_clustered(du, u, p, t; L=L)
            L, N = L  # Extract Laplacian and system size
        
            # Extract parameters
            ρ = p[1]
            α = p[2:2+N-1]
            β = p[2+N:2+2*N-1]
            d = p[2+2*N:2+3*N-1]
            γ = p[2+3*N:end]
        
            # Extract state variables
            x = u[1:N]  
            y = u[(N+1):(2*N)]  
        
            # **Vectorized ODE Computation**
            du[1:N] .= -ρ .* (L * x) .+ α .* x .* (β .- β .* y .- d .* y .- x)
            du[(N+1):2*N] .= γ .* (1 .- y)
        end

        # samples
        p = [ρ, α..., β..., d..., γ...]   # first index is σ and last index is seed
        if inference["bayesian_seed"]
            u0[seed] = sample[seed_ch_idx]  # TODO: find seed index automatically
            #u0[seed] = sample[end]  # TODO: find seed index automatically
        else    
            u0[seed] = inference["seed_value"]
        end
        σ = sample[end-1]
        
        # solve
        prob = ODEProblem(death_simplifiedii_clustered, u0, tspan; alg=Tsit5())
        sol_p = solve(prob,Tsit5(); p=p, u0=u0, saveat=0.1, abstol=1e-9, reltol=1e-6)
        t = sol_p.t
        sol_p = Array(sol_p[sol_idxs,:])
        for i in 1:N
            lower_bound = sol_p[i,:] .- σ
            upper_bound = sol_p[i,:] .+ σ
            if show_variance
                CairoMakie.band!(axs[i], t, lower_bound, upper_bound; color=(:grey,0.1))
                CairoMakie.lines!(axs[i],t, sol_p[i,:]; alpha=0.9, color=:black)
            end
            CairoMakie.lines!(axs[i],t, sol_p[i,:]; alpha=0.5, color=:grey)
        end
    end

    # Plot simulation and noisy observations.
    # plot mean and variance
    for i in 1:N
        # =-=----
        nonmissing = findall(mean_data[i,:] .!== missing)
        data_i = mean_data[i,:][nonmissing]
        timepoints_i = timepoints[nonmissing]
        var_data_i = var_data[i,:][nonmissing]
        indices = findall(x -> isnan(x),var_data_i)
        var_data_i[indices] .= 0
        CairoMakie.scatter!(axs[i], timepoints_i, data_i; color=RGB(0/255, 0/255, 139/255), alpha=1., markersize=15)  
        # have lower std capped at 0.01 (to be visible in the plots)
        var_data_i_lower = copy(var_data_i)
        for (n,var) in enumerate(var_data_i)
            if sqrt(var) > data_i[n]
                var_data_i_lower[n] = max(data_i[n]^2-1e-5, 0)
                #var_data_i_lower[n] = data_i[n]^2
            end
        end

        #CairoMakie.errorbars!(axs[i], timepoints_i, data_i, sqrt.(var_data_i); color=RGB(0/255, 71/255, 171/255), whiskerwidth=20, alpha=0.2)
        CairoMakie.errorbars!(axs[i], timepoints_i, data_i, sqrt.(var_data_i_lower), sqrt.(var_data_i); color=RGB(0/255, 71/255, 171/255), whiskerwidth=20, alpha=0.2, linewidth=3)
        #CairoMakie.errorbars!(axs[i], timepoints_i, data_i, [0. for _ in 1:length(timepoints_i)], sqrt.(var_data_i); color=RGB(0/255, 71/255, 171/255), whiskerwidth=20, alpha=0.2)
    end
    # plot all data points across all samples
    for i in 1:N
        jiggle = rand(Normal(0,0.01),size(data)[3])
        for k in axes(data,3)
            # =-=----
            nonmissing = findall(data[i,:,k] .!== missing)
            data_i = data[i,:,k][nonmissing]
            timepoints_i = timepoints[nonmissing] .+ jiggle[k]
            CairoMakie.scatter!(axs[i], timepoints_i, data_i; color=RGB(0/255, 71/255, 171/255), alpha=0.4, markersize=15)  
        end
        CairoMakie.save(save_path * "/retrodiction_region_$(i).png", fs[i])
    end

    # we're done
    return fs
end

#function correlation_analysis(x::Vector, y::Vector; save_path::Union{String, Nothing}=nothing, xlabel="X", ylabel="Y", title="Correlation analysis")
#    @assert length(x) == length(y) "Vectors must have the same length"
#    
#    # Compute Pearson correlation coefficient
#    r = cor(x, y)
#    println("Correlation coefficient (r): ", r)
#    
#    # If a save path is provided, generate and save the plot
#    if save_path !== nothing
#        Plots.scatter(x, y, label="r = $r", xlabel=xlabel, ylabel=ylabel, title=title)
#        Plots.savefig(save_path)
#        println("Plot saved at: ", save_path)
#    end
#    
#    return r
#end
"""
    correlation_analysis(x, y; save_path=nothing, xlabel="X", ylabel="Y", title="Correlation analysis",
                         plot_fit_line=false, alpha=0.05, plotscale=identity, xlims=nothing, ylims=nothing)

Computes the Pearson correlation coefficient between vectors `x` and `y` and generates a scatter plot.
Optionally, if `plot_fit_line` is true, a linear regression line is plotted along with its confidence interval.
The `plotscale` keyword sets the scaling of both the x- and y-axes.
Optional keyword arguments `xlims` and `ylims` set the axis limits.
Returns a tuple `(r, plt)` with the correlation coefficient and the plot object.
"""
function correlation_analysis(x::AbstractVector, y::AbstractVector;
                              save_path::Union{String, Nothing}=nothing,
                              xlabel="X", ylabel="Y", title="Correlation analysis",
                              plot_fit_line::Bool=false, alpha::Real=0.05,
                              plotscale=:identity, xlims=nothing, ylims=nothing, aspect_ratio=:auto)
    @assert length(x) == length(y) "Vectors must have the same length"

    # Compute Pearson correlation coefficient
    r = cor(x, y)
    println("Correlation coefficient (r): ", r)

    # Define muted color palette
    muted_blue = "#4a6fa5"  # For scatter markers
    muted_red  = "#a45a52"  # For best-fit line and ribbon

    # Create the base scatter plot with enhanced aesthetics.
    plt = Plots.scatter(x, y, 
                  label = "",           # Remove legend
                  xlabel = xlabel, ylabel = ylabel, 
                  title = title,
                  markersize = 5,
                  markercolor = muted_blue, 
                  markerstrokewidth = 0.5,
                  legend = false,
                  grid = true,
                  background_color = :white, 
                  framestyle = :box,
                  titlefontsize = 16,
                  guidefontsize = 14,
                  tickfontsize = 12,
                  xscale = plotscale,
                  yscale = plotscale,
                  xlims = xlims,
                  ylims = ylims,
                  aspect_ratio=aspect_ratio)

    # Optionally add best-fit line with confidence interval
    if plot_fit_line
        n = length(x)
        b = cov(x, y) / var(x)
        a = mean(y) - b * mean(x)
        x_line = range(minimum(x), stop=maximum(x), length=100)
        y_line = a .+ b .* x_line

        # Calculate residuals and standard error
        y_pred = a .+ b .* x
        resid = y .- y_pred
        s = sqrt(sum(resid .^ 2) / (n - 2))
        mean_x = mean(x)
        s_x2 = sum((x .- mean_x).^2)
        se_fit = s .* sqrt.(1/n .+ ((x_line .- mean_x).^2) ./ s_x2)
        tcrit = quantile(TDist(n - 2), 1 - alpha/2)
        ribbon = tcrit .* se_fit

        Plots.plot!(plt, x_line, y_line, ribbon = ribbon, 
              label = "", 
              lw = 3,
              linecolor = muted_red,
              fillalpha = 0.3,
              fillcolor = muted_red)
    end

    # Position annotation in lower-right corner.
    # Compute padding relative to the data range.
    x_min, x_max = extrema(x)
    y_min, y_max = extrema(y)
    x_pos = x_max - 0.05*(x_max - x_min)
    y_pos = y_min + 0.05*(y_max - y_min)
    # Annotate with bold, slightly larger text.
    Plots.annotate!(plt, 0.0, -6., Plots.text("r = $(round(r, digits=2))", :black, :right, 16))




    # Save the plot if a save_path is provided.
    if save_path !== nothing
        Plots.savefig(plt, save_path)
        println("Plot saved at: ", save_path)
    end

    return r, plt
end






"""
    extract_mode_params(inference::Dict)

Extracts the parameter set corresponding to the highest posterior probability (posterior mode)
from the `inference` dictionary. Also updates the initial conditions based on the Bayesian seed.
Returns a tuple `(p, u0, N_pars)` where `p` is the vector of mode parameters,
`u0` is the initial conditions vector, and `N_pars` is the number of parameters.
"""
function extract_mode_params(inference::Dict)
    chain = inference["chain"]
    ks = collect(keys(inference["priors"]))
    # Identify the number of parameters by finding the index of the "σ" key.
    N_pars = findfirst(x -> x == "σ", ks) - 1
    n_pars = length(chain.info[1])
    _, argmax = findmax(chain[:lp])
    mode_pars = Array(chain[argmax[1], 1:n_pars, argmax[2]])
    p = mode_pars[1:N_pars]
    u0 = copy(inference["u0"])
    if inference["bayesian_seed"]
        u0[inference["seed_idx"]] = chain["seed"][argmax]
    else
        u0[inference["seed_idx"]] = inference["seed_value"]
    end
    return p, u0, N_pars
end


"""
    simulate_ode(inference::Dict, p, u0)

Sets up and solves the ODE problem using the mode parameters `p` and initial conditions `u0`
from the `inference` dictionary. Assumes that a global dictionary `odes` exists that maps
ODE names to their functions. Returns the ODE solution array (subset according to `sol_idxs`).
"""
function simulate_ode(inference::Dict, p, u0)
    timepoints = inference["timepoints"]
    tspan = (0.0, timepoints[end])
    # Retrieve the ODE function from a global dictionary `odes`
    ode_function = odes[inference["ode"]]
    L = inference["L"]
    # Use a factor of 1 for each parameter (you can adjust this if needed)
    factors = ones(length(p))
    # Define the ODE right-hand side
    function rhs(du, u, p, t)
        ode_function(du, u, p, t; L = L, factors = factors)
    end
    prob = ODEProblem(rhs, u0, tspan; alg = Tsit5())
    sol = solve(prob, Tsit5(); p = p, u0 = u0, saveat = timepoints, abstol = 1e-9, reltol = 1e-6)
    sol = Array(sol[inference["sol_idxs"], :])
    return sol
end


"""
    prepare_plot_data(data, sol, time_index)

Extracts and cleans the observed data and corresponding simulated values for the given timepoint.
Returns a tuple `(x, y)` where `x` is the observed data vector and `y` is the simulated data vector.
"""
function prepare_plot_data(data, sol, time_index)
    # Extract the column corresponding to the given time index
    x = vec(copy(data[:, time_index]))
    y = vec(copy(sol[:, time_index]))
    # Filter out missing values if any
    nonmissing = findall(x .!== missing)
    x = x[nonmissing]
    y = y[nonmissing]
    return x, y
end


"""
    predicted_vs_observed_plots(inference; save_path="", plotscale=log10, remove_zero=false)

For each timepoint in the `inference` dictionary, extracts the highest–posterior parameters,
simulates the ODE, prepares the observed and predicted data, and generates a predicted vs. observed
scatter plot using `correlation_analysis()`. Each plot includes a best–fit line with confidence interval,
and the plot title indicates the timepoint. Optionally, figures are saved to `save_path`.
If `remove_zero` is true, data points with a 0 value (in either observed or predicted data) are removed.
The `plotscale` keyword sets the scaling for both the x- and y-axes.
Returns a list of plot objects.
"""
function predicted_vs_observed_plots(inference; save_path="", plotscale=:log10, remove_zero=false, aspect_ratio=:auto)
    # Create save folder if needed.
    if save_path != ""
        try
            mkdir(save_path)
        catch
            # Directory may already exist.
        end
    end

    # Extract mode parameters and simulate the ODE.
    p, u0, _ = extract_mode_params(inference)
    sol = simulate_ode(inference, p, u0)
    data = mean3(inference["data"])  # average over samples
    timepoints = inference["timepoints"]

    # First pass: gather all x and y data (after processing) for global limits.
    global_x = Float64[]
    global_y = Float64[]

    for i in 1:length(timepoints)
        x_i, y_i = prepare_plot_data(data, sol, i)
        # Optionally remove zeros.
        if remove_zero
            nonzero = (x_i .!= 0) .& (y_i .!= 0)
            x_i = x_i[nonzero]
            y_i = y_i[nonzero]
        end
        # If using a log scale, transform data.
        if plotscale == :log10
            x_i = log10.(x_i)
            y_i = log10.(y_i)
        end
        append!(global_x, x_i)
        append!(global_y, y_i)
    end

    # Compute global axis limits with 5% padding.
    x_min, x_max = minimum(global_x), maximum(global_x)
    y_min, y_max = minimum(global_y), maximum(global_y)
    pad_x = 0.05 * (x_max - x_min)
    pad_y = 0.05 * (y_max - y_min)
    global_xlims = (x_min - pad_x, x_max + pad_x)
    global_ylims = (y_min - pad_y, y_max + pad_y)

    figures = []

    # Second pass: generate plots with consistent axis limits.
    for (i, t) in enumerate(timepoints)
        x, y = prepare_plot_data(data, sol, i)
        # Optionally remove zeros.
        if remove_zero
            nonzero = (x .!= 0) .& (y .!= 0)
            x = x[nonzero]
            y = y[nonzero]
        end

        local_scale = plotscale
        if plotscale == :log10
            x = log10.(x)
            y = log10.(y)
            local_scale = :identity  # data are already in log-space
        end

        title_str = "$(t) MPI"
        sp = save_path == "" ? nothing : joinpath(save_path, "predicted_vs_observed_$(i).png")

        # Generate the plot using correlation_analysis with global axis limits.
        r, plt = correlation_analysis(x, y; save_path=sp,
                                      xlabel="Observed", ylabel="Predicted",
                                      title=title_str, plot_fit_line=true,
                                      plotscale=local_scale,
                                      xlims=global_xlims, ylims=global_ylims,
                                      aspect_ratio=aspect_ratio)
        push!(figures, plt)
    end

    return figures
end



using DifferentialEquations
using Distributions
using Statistics
using StatsPlots   # Provides boxplot; ensure you have this package

#--------------------------------------------------------------------------
# 1. Extract sample parameters from an inference object.
#--------------------------------------------------------------------------
"""
    extract_sample_params(inference::Dict, sample_index::Int)

Extracts the parameter vector and initial conditions from `inference`
for the posterior sample given by `sample_index`.
Returns a tuple `(p, u0)`.
"""
function extract_sample_params(inference::Dict, sample_index::Int)
    chain = inference["chain"]
    ks = collect(keys(inference["priors"]))
    # Identify the number of parameters by finding the index of "σ".
    N_pars = findfirst(x -> x == "σ", ks) - 1
    n_pars = length(chain.info[1])
    # Here we assume the chain is indexed as chain[sample, parameter, chain] (adjust if needed).
    sample_pars = Array(chain[sample_index, 1:n_pars, 1])
    p = sample_pars[1:N_pars]
    u0 = copy(inference["u0"])
    if inference["bayesian_seed"]
        u0[inference["seed_idx"]] = chain["seed"][sample_index]
    else
        u0[inference["seed_idx"]] = inference["seed_value"]
    end
    return p, u0
end

#--------------------------------------------------------------------------
# 2. Simulate the ODE for a given posterior sample.
#--------------------------------------------------------------------------
"""
    simulate_ode_sample(inference::Dict, sample_index::Int)

Simulates the ODE using the parameters extracted from the given posterior sample.
Returns the solution array.
"""
function simulate_ode_sample(inference::Dict, sample_index::Int)
    p, u0 = extract_sample_params(inference, sample_index)
    sol = simulate_ode(inference, p, u0)  # reuse your existing simulate_ode()
    return sol
end

#--------------------------------------------------------------------------
# 3. Compute r for a given sample and timepoint.
#--------------------------------------------------------------------------
"""
    compute_r_for_sample(inference::Dict, sample_index::Int, timepoint::Int)

Simulates the ODE using the posterior sample given by `sample_index`, extracts the
observed and predicted data at `timepoint`, and returns the Pearson correlation
coefficient.
"""
function compute_r_for_sample(inference::Dict, sample_index::Int, timepoint::Int)
    sol = simulate_ode_sample(inference, sample_index)
    data = mean3(inference["data"])  # average over samples
    x_obs, y_pred = prepare_plot_data(data, sol, timepoint)
    r = cor(x_obs, y_pred)
    return r
end

#--------------------------------------------------------------------------
# 4. Compute the distribution of r-values for one inference object and one timepoint.
#--------------------------------------------------------------------------
"""
    r_distribution_for_timepoint(inference::Dict, sample_indices::Vector{Int}, timepoint::Int)

For the given `inference` object and timepoint (an index), computes and returns a vector
of r-values over the specified posterior samples (given by `sample_indices`).
"""
function r_distribution_for_timepoint(inference::Dict, sample_indices::Vector{Int}, timepoint::Int)
    r_values = [compute_r_for_sample(inference, s, timepoint) for s in sample_indices]
    return r_values
end

#--------------------------------------------------------------------------
# 5. Generate and save box plots of r-value distributions for each timepoint.
#--------------------------------------------------------------------------
"""
    boxplot_r_values(inference_list::Vector{Dict}; sample_indices=1:100, save_path="", model_names=nothing)

For each timepoint, computes the r-value distributions (over the given posterior sample indices)
for each inference object in `inference_list` and produces a box plot comparing the models.
One figure per timepoint is created and saved in `save_path` (if provided; the folder is created if needed).
Optionally, you can supply a vector of `model_names` to label the different models.
Returns an array of plot objects.
"""
function boxplot_r_values(inference_list::Vector{Dict}; sample_indices=1:100, save_path="", model_names=nothing)
    # Create the save folder if save_path is provided.
    if save_path != ""
        try
            mkdir(save_path)
        catch
            # Folder may already exist.
        end
    end

    # If no model names are provided, default to "Model 1", "Model 2", etc.
    if model_names === nothing
        model_names = ["Model $i" for i in 1:length(inference_list)]
    end

    # Assume all inference objects share the same timepoints.
    timepoints = inference_list[1]["timepoints"]
    figures = []

    # First pass: gather all r-values (after processing) to compute global y-axis limits.
    #global_r_values = Float64[]
    #for t_idx in 1:length(timepoints)
    #    for inf in inference_list
    #        r_vals = r_distribution_for_timepoint(inf, collect(sample_indices), t_idx)
    #        append!(global_r_values, r_vals)
    #    end
    #end
    #global_ymin, global_ymax = minimum(global_r_values), maximum(global_r_values)
    global_ymin, global_ymax = 0,1
    pad_y = 0.05 * (global_ymax - global_ymin)
    global_ylims = (global_ymin - pad_y, global_ymax + pad_y)

    # Second pass: generate one box plot per timepoint.
    for (t_idx, t) in enumerate(timepoints)
        model_r_dists = Vector{Vector{Float64}}()
        for inf in inference_list
            r_vals = r_distribution_for_timepoint(inf, collect(sample_indices), t_idx)
            push!(model_r_dists, r_vals)
        end

        # Flatten the r-values and create a grouping vector based on model names.
        all_r = vcat(model_r_dists...)
        groups = vcat([fill(model_names[i], length(model_r_dists[i])) for i in 1:length(model_r_dists)]...)

        # Create the box plot.
        p = StatsPlots.boxplot(groups, all_r,
                    xlabel = "",
                    ylabel = "r-value",
                    title = "$(t) MPI",
                    legend = false,
                    ylims = global_ylims,
                    fillcolor = :gray,    # A muted fill color
                    linecolor = :black,   # Black borders for clarity
                    titlefontsize = 18,   # Increase title font size
                    guidefontsize = 16,   # Increase x and y label font size
                    tickfontsize = 14)    # Increase tick label font size

        # Save the plot if save_path is provided.
        if save_path != ""
            filename = joinpath(save_path, "boxplot_r_timepoint_$(t_idx).png")
            Plots.savefig(p, filename)
            println("Box plot saved at: ", filename)
        end

        push!(figures, p)
    end

    return figures
end

function shortest_paths(A::Matrix{T}) where T <: Any
    n = size(A, 1)
    M = copy(A)

    # Replace zeros (except diagonal) with Inf to indicate no direct path
    for i in 1:n, j in 1:n
        if i != j && M[i, j] == 0
            M[i, j] = Inf
        end
    end

    # Floyd-Warshall Algorithm with Inf handling
    for k in 1:n
        for i in 1:n
            for j in 1:n
                if M[i, k] < Inf && M[k, j] < Inf  # Ensure no Inf overflow
                    M[i, j] = min(M[i, j], M[i, k] + M[k, j])
                end
            end
        end
    end
    return M
end