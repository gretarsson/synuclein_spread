using Turing
using DelimitedFiles
using StatsPlots
using DifferentialEquations
using Distributions
using TuringBenchmarking  
using ReverseDiff
using SciMLSensitivity
using LinearAlgebra
using Serialization
using CairoMakie
using ParetoSmooth
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
Plot the chains and posterior dist of each estimated parameter
=#
function plot_chains(chain, path; priors=nothing)
    vars = chain.info.varname_to_symbol
    master_fig = StatsPlots.plot(chain) 
    i = 1
    for (key,value) in vars  # iterate through parameters
        #prior = priors["$(value)"]
        # plot Markov chain
        chain_i = StatsPlots.plot(master_fig[i,1])
        savefig(chain_i,path * "/chain_$(value).png")
        # plot prior alone
        #prior_i = StatsPlots.plot(prior)
        #savefig(prior_i,path * "/prior_$(value).png")
        # plot posterior alone
        posterior_i = StatsPlots.plot(master_fig[i,2])
        savefig(posterior_i,path * "/posterior_$(value).png")
        # plot posterior and prior together
        #StatsPlots.plot!(posterior_i, prior, color=:grey)  # add prior to posterior plot
        #savefig(posterior_i,path * "/prior_posterior_$(value).png")
        i += 1
    end
end

#=
Plot retrodiction of chain compared to data
=#
function plot_retrodiction(;data=nothing, chain=nothing, prob=nothing, path=nothing, timepoints=nothing, seed=0, seed_bayesian=false, u0=nothing, N_samples=300)
    N = size(data)[1]
    M = length(timepoints)
    fs = Any[NaN for _ in 1:N]
    axs = Any[NaN for _ in 1:N]
    for i in 1:N
        f = CairoMakie.Figure()
        ax = CairoMakie.Axis(f[1,1], title="Region $(i)", ylabel="Portion of cells infected", xlabel="time (months)", xticks=0:9, limits=(0,9.1,0,1))
        fs[i] = f
        axs[i] = ax
    end
    posterior_samples = sample(chain, N_samples; replace=false)
    avg_sol =   zeros(N,M)
    for sample in eachrow(Array(posterior_samples))
        # samples
        if seed_bayesian  # means IC at seed region has posterior
            p = sample[2:(end-1)]
            u0[seed] = sample[end]
        else
            p = sample[2:end]
        end
        
        # solve
        sol_p = solve(prob,Tsit5(); p=p, u0=u0, saveat=0.1, abstol=1e-9, reltol=1e-6)
        sol_p_timepoints = solve(prob,Tsit5(); p=p, u0=u0, saveat=timepoints, abstol=1e-9, reltol=1e-6)
        for i in 1:N
            CairoMakie.lines!(axs[i],sol_p.t, sol_p[i,:]; alpha=0.3, color=:grey)
        end
        # add to average
        avg_sol = avg_sol .+ (Array(sol_p_timepoints[1:N,:]) ./ N_samples)
    end

    # Plot simulation and noisy observations.
    for i in 1:N
        CairoMakie.scatter!(axs[i], timepoints, data[i,:]; colormap=:tab10)
        CairoMakie.save(path * "/retrodiction_region_$(i).png", fs[i])
    end
    # plot predicted vs data
    f = CairoMakie.Figure()
    ax = CairoMakie.Axis(f[1,1], title="", ylabel="Predicted", xlabel="Observed", aspect=1)
    CairoMakie.scatter!(ax,vec(data),vec(avg_sol), alpha=0.5)
    maxl = max(maximum(data), maximum(avg_sol))
    CairoMakie.lines!([0,maxl],[0,maxl], color=:grey, alpha=0.5)
    CairoMakie.save(path * "/predicted_observed_average.png", f)
    # plot mode prediction
    n_pars = length(chain.info[1])
    _, argmax = findmax(chain[:lp])
    mode_pars = Array(chain[argmax[1], 2:n_pars, argmax[2]])
    if seed_bayesian
        p = mode_pars[1:(end-1)]
        u0[seed] = mode_pars[end]
    else
        p = mode_pars
    end
    println(p)
    sol = solve(prob,Tsit5(); p=p, u0=u0, saveat=timepoints, abstol=1e-9, reltol=1e-6)
    f = CairoMakie.Figure()
    ax = CairoMakie.Axis(f[1,1], title="", ylabel="Predicted", xlabel="Observed", aspect=1)
    CairoMakie.scatter!(ax,vec(data),vec(Array(sol[1:N,:])), alpha=0.5)
    maxl = max(maximum(data), maximum(sol))
    CairoMakie.lines!([0,maxl],[0,maxl], color=:grey, alpha=0.5)
    CairoMakie.save(path * "/predicted_observed_mode.png", f)
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
function diffusion(du,u,p,t;L=L)
    ρ = p[1]

    du .= -ρ*L*u 
end
function aggregation(du,u,p,t;L=LT)
    ρ = p[1]
    α = p[2]
    β = p[3:end]

    du .= -ρ*L*u .+ α .* u .* (β .- u)  
end
function diffusion2(du,u,p,t;L=(La, Lr))
    La, Lr = L
    ρa = p[1]
    ρr = p[2]

    du .= -(ρa*La+ρr*Lr)*u 
end



# ----------------------------------------------------
# Run whole simulations in one place
# ----------------------------------------------------
function infer(ode, priors::OrderedDict, data_file, timepoints_file, W_file; 
               n_threads=1,
               retro_and_antero=false,
               alg=Tsit5(), 
               sensealg=ForwardDiffSensitivity(), 
               adtype=AutoForwardDiff(), 
               threshold=0., 
               abstol=1e-10, 
               reltol=1e-10,
               benchmark=false
               )
    # read empirical data
    data, idxs = read_data(data_file, remove_nans=true, threshold=threshold)
    timepoints = vec(readdlm(timepoints_file, ','))

    # read structural data 
    W_labelled = readdlm(W_file,',')
    W = W_labelled[2:end,2:end]
    W = W[idxs,idxs]
    L = Matrix(transpose(laplacian_out(W; self_loops=false, retro=false)))  # transpose of Laplacian (so we can write LT * x, instead of x^T * L)
    labels = W_labelled[1,2:end][idxs]
    seed = findall(x->x=="iCP",labels)[1]  # find index of seed region
    N = size(L)[1]
    N_pars = length(priors)-2  # minus sigma and IC prior
    if retro_and_antero  # include both Laplacians, if told to
        La = L
        Lr = Matrix(transpose(laplacian_out(W; self_loops=false, retro=true)))  
        L = (La,Lr)
    end

    # Define prob
    u0 = [0. for _ in 1:N]
    p = zeros(Float64, N_pars)
    tspan = (timepoints[1],timepoints[end])
    rhs(du,u,p,t;L=L, func=ode::Function) = func(du,u,p,t;L=L)
    prob = ODEProblem(rhs, u0, tspan, p; alg=alg)
    
    # prior vector from ordered dic
    priors_vec = collect(values(priors))

    @model function bayesian_model(data, prob; priors=priors_vec, alg=alg, timepoints=timepoints, seed=seed)
        # initializations 
        p = zeros(Float64, N_pars)
        u0 = [0. for _ in 1:N]

        # priors
        σ ~ priors[1]
        for i in 1:N_pars
            p[i] ~ priors[1+i]  
        end
        u0[seed] ~ priors[end]  

        # Simulate diffusion model 
        predicted = solve(prob, alg; u0=u0, p=p, saveat=timepoints, sensealg=sensealg, abstol=abstol, reltol=reltol)

        # Observations.
        for i in axes(predicted,1), j in axes(predicted,2)
            data[i,j] ~ Normal(predicted[i,j], σ^2)
        end

        return nothing
    end

    # define Turing model
    model = bayesian_model(data, prob)

    # benchmark
    if benchmark
        suite = TuringBenchmarking.make_turing_suite(model;adbackends=[:forwarddiff,:reversediff])
        println(run(suite))
        return nothing
    end

    # Sample to approximate posterior
    if n_threads == 1
        chain = sample(model, NUTS(;adtype=adtype), 1000; progress=true)  # time estimated is shown
    else
        chain = sample(model, NUTS(;adtype=adtype), MCMCThreads(), 1000, n_threads; progress=true)
    end
    display(chain)  

    # save chains and metadata to a dictionary
    inference = Dict("chain" => chain, 
                     "priors" => priors, 
                     "data" => data,
                     "timepoints" => timepoints,
                     "threshold" => threshold, 
                     "seed_idx" => seed,
                     "ode" => string(ode)  # store var name of ode (functions cannot be saved)
                     )

    return inference
end