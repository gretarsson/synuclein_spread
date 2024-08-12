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
using Random
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
function diffusion(du,u,p,t;L=L,factors=nothing)
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
    du[(N+1):(2*N)] .= γ .* tanh.(x) .-  1/γ .* y
end
function diffusion4(du,u,p,t;L=(La,Lr),N=1::Int, factors=nothing)
    La, Lr = L
    ρa = p[1]
    ρr = p[2]
    γ = p[3]
    x = u[1:N]
    y = u[(N+1):(2*N)]

    du[1:N] .= -ρa*La*x-ρr*Lr*x
    du[(N+1):(2*N)] .= γ .* tanh.(x) .-  1/γ .* y
end
function aggregation(du,u,p,t;L=L,factors=(1.,1.))
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
#=
a dictionary containing the ODE functions
=#
odes = Dict("diffusion" => diffusion, "diffusion2" => diffusion2, "diffusion3" => diffusion3, "diffusion4" => diffusion4, "aggregation" => aggregation, "aggregation2" => aggregation2)



# ----------------------------------------------------------------------------------------------------------------------------------
# Run whole simulations in one place
# the Priors dict must contains σ in its first key and seed in its last (independent whether seed is actually uncertain or not)
# ----------------------------------------------------------------------------------------------------------------------------------
function infer(ode, priors::OrderedDict, data_file, timepoints_file, W_file; 
               u0=[]::Vector{Float64},
               n_threads=1,
               alg=Tsit5(), 
               sensealg=ForwardDiffSensitivity(), 
               adtype=AutoForwardDiff(), 
               threshold=0., 
               factors=[1.]::Vector{Float64},
               bayesian_seed=false,
               seed_region="iCP"::String,
               seed_value=1.::Float64,
               pred_idxs=[1]::Vector{Int},
               abstol=1e-10, 
               reltol=1e-10,
               benchmark=false,
               benchmark_ad=[:forwarddiff, :reversediff, :reversediff_compiled],
               test_typestable=false,
               transform_observable=false
               )

    # verify that choice of ODE is correct wrp to retro- and anterograde
    retro_and_antero = false
    if string(ode)[end] == '2'
        retro_and_antero = true 
        display("Model includes both retrograde and anterograde transport.")
    else 
        display("Model includes only anterograde transport.")
    end
    if bayesian_seed
        display("Model is inferring seeding initial conditions")
    else
        display("Model has constant initial conditions")
    end

    # read empirical data
    data, idxs = read_data(data_file, remove_nans=true, threshold=threshold)
    timepoints = vec(readdlm(timepoints_file, ','))::Vector{Float64}
    # read only part of data
    #data = data[:,5:end]
    #timepoints = timepoints[5:end]

    # read structural data 
    W_labelled = readdlm(W_file,',')
    W = W_labelled[2:end,2:end]
    W = W[idxs,idxs]
    #W = W .* W_factor
    L = Matrix(transpose(laplacian_out(W; self_loops=false, retro=false)))  # transpose of Laplacian (so we can write LT * x, instead of x^T * L)
    labels = W_labelled[1,2:end][idxs]
    seed = findall(x->x==seed_region,labels)[1]::Int  # find index of seed region
    N = size(L)[1]
    N_pars = length(priors)-2  # minus sigma and IC prior
    if retro_and_antero  # include both Laplacians, if told to
        La = copy(L)
        Lr = Matrix(transpose(laplacian_out(W; self_loops=false, retro=true)))  
        L = (La,Lr)
    end

    # Define prob
    p = zeros(Float64, N_pars)
    tspan = (timepoints[1],timepoints[end])
    #rhs(du,u,p,t;L=L, func=ode::Function) = func(du,u,p,t;L=L,N=N)  # for diffusion3
    rhs(du,u,p,t;L=L, func=ode::Function) = func(du,u,p,t;L=L,factors=factors)
    prob = ODEProblem(rhs, u0, tspan, p; alg=alg)
    
    # prior vector from ordered dic
    priors_vec = collect(values(priors))
    if pred_idxs == [1]
        pred_idxs = [i for i in 1:N]
    end


    @model function bayesian_model(data, prob; priors=priors_vec, alg=alg, timepointss=timepoints::Vector{Float64}, seedd=seed::Int, u0=u0::Vector{Float64}, bayesian_seed=bayesian_seed::Bool, seed_value=seed_value)
        u00 = u0  # IC needs to be defined within model to work
        # priors
        σ ~ priors[1]
        p ~ arraydist([priors[1+i] for i in 1:N_pars])
        if bayesian_seed
            u00[seedd] ~ priors[end]  
        else
            u00[seedd] = seed_value
        end
        #k ~ truncated(Normal(1.,0.1), lower=0)

        # Simulate diffusion model 
        predicted = solve(prob, alg; u0=u00, p=p, saveat=timepointss, sensealg=sensealg, abstol=abstol, reltol=reltol)

        # Transform prediction
        predicted = vec(predicted[pred_idxs,:])
        if transform_observable
            #predicted = k .* predicted
            predicted = tanh.(predicted)
        end

        # Observations.
        data ~ MvNormal(predicted, σ^2 * I)  # this gives quicker evals (but recommended by TuringLang developers). For large N, this appears superior

        return nothing
    end

    # define Turing model
    model = bayesian_model(vec(data), prob)

    # test if typestable if told to, red marking in read-out means something is unstable
    if test_typestable
        @code_warntype model.f(
            model,
            Turing.VarInfo(model),
            Turing.SamplingContext(
                Random.GLOBAL_RNG, Turing.SampleFromPrior(), Turing.DefaultContext(),
            ),
            model.args...,
        )
    end

    # benchmark
    if benchmark
        suite = TuringBenchmarking.make_turing_suite(model;adbackends=benchmark_ad)
        println(run(suite))
        return nothing
    end

    # Sample to approximate posterior
    if n_threads == 1
        chain = sample(model, NUTS(1000,0.65;adtype=adtype), 1000; progress=true)  # time estimated is shown
        #chain = sample(model, HMC(0.05,10), 1000; progress=true)
    else
        chain = sample(model, NUTS(1000,0.65;adtype=adtype), MCMCThreads(), 1000, n_threads; progress=true)
        #chain = sample(model, HMC(0.05,10), MCMCThreads(), 1000, n_threads; progress=true)
    end
    display(chain)  

    # save chains and metadata to a dictionary
    inference = Dict("chain" => chain, 
                     "priors" => priors, 
                     "data" => data,
                     "timepoints" => timepoints,
                     "threshold" => threshold, 
                     "seed_idx" => seed,
                     "bayesian_seed" => bayesian_seed,
                     "transform_observable" => transform_observable,
                     "ode" => string(ode),  # store var name of ode (functions cannot be saved)
                     "factors" => factors,
                     "L" => L
                     )

    return inference
end


#=
plot predicted vs observed plot for inference, parameters chosen from posterior mode
=#
using Makie
function predicted_observed(inference; save_path="")
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
    factors = [1. for _ in 1:(length(inference["priors"])-2)]
    ode = odes[inference["ode"]]
    N = size(data)[1]

    # simulate ODE from posterior mode
    # initialize
    tspan = (0., timepoints[end])
    u0 = [0. for _ in 1:N]
    # newthing
    #u0 = [0. for _ in 1:2*N]
    rhs(du,u,p,t;L=L, func=ode::Function) = func(du,u,p,t;L=L,factors=factors)
    # newthing
    #rhs(du,u,p,t;L=L, func=ode::Function) = func(du,u,p,t;L=L, N=N)
    prob = ODEProblem(rhs, u0, tspan; alg=Tsit5())

    # find posterior mode
    n_pars = length(chain.info[1])
    _, argmax = findmax(chain[:lp])
    mode_pars = Array(chain[argmax[1], 2:n_pars, argmax[2]])
    p = mode_pars[1:(end-1)]
    u0[seed] = mode_pars[end]

    # solve ODE
    sol = solve(prob,Tsit5(); p=p, u0=u0, saveat=timepoints, abstol=1e-9, reltol=1e-6)
    if inference["transform_observable"]
        sol = tanh.(sol)
    end

    # new thing
    #sol = sol[(N+1):end,:]

    # plot
    f = CairoMakie.Figure()
    plotscale = log10
    ax = CairoMakie.Axis(f[1,1], title="", ylabel="Predicted", xlabel="Observed", xscale=plotscale, yscale=plotscale)

    # as we are plotting log-log, we account for zeros in the data
    x = vec(copy(data))
    y = vec(copy(sol[1:N,:]))
    if (sum(x .== 0) + sum(y .== 0)) > 0  # if zeros present, add the smallest number in plot
        minx = minimum(x[x.>0])
        miny = minimum(y[y.>0])
        minxy = min(minx, miny)
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
        plotscale = log10
        ax = CairoMakie.Axis(f[1,1], title="t = $(timepoints[i])", ylabel="Predicted", xlabel="Observed", xscale=plotscale, yscale=plotscale)

        # as we are plotting log-log, we account for zeros in the data
        x = vec(copy(data[:,i]))
        y = vec(copy(sol[1:N,i]))
        if (sum(x .== 0) + sum(y .== 0)) > 0  # if zeros present, add the smallest number in plot
            minx = minimum(x[x.>0])
            miny = minimum(y[y.>0])
            minxy = min(minx, miny)
            x = x .+ minxy
            y = y .+ minxy
        end

        CairoMakie.scatter!(ax,x,y, alpha=0.5)
        maxxy = max(maximum(x), maximum(y))
        minxy = min(minimum(x), minimum(y))
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
    posterior_figs = []
    for (i,var) in enumerate(vars)
        posterior_i = StatsPlots.plot(master_fig[i,2], title=var)
        if !isempty(save_path)
            savefig(posterior_i, save_path*"/posterior_$(var).png")
        end
        push!(posterior_figs,posterior_i)
    end
    return posterior_figs
end

#=
plot retrodictino from inference result
=#
function plot_retrodiction(inference; save_path=nothing, N_samples=300)
    # try creating folder to save in
    try
        mkdir(save_path) 
    catch
    end
    # unload from simulation
    data = inference["data"]
    chain = inference["chain"]
    timepoints = inference["timepoints"]
    seed = inference["seed_idx"]
    L = inference["L"]
    factors = [1. for _ in 1:(length(inference["priors"])-2)]
    ode = odes[inference["ode"]]
    N = size(data)[1]
    M = length(timepoints)

    # define ODE problem 
    u0 = [0. for _ in 1:N]
    tspan = (0, timepoints[end])
    rhs(du,u,p,t;L=L, func=ode::Function) = func(du,u,p,t;L=L,factors=factors)
    prob = ODEProblem(rhs, u0, tspan; alg=Tsit5())

    fs = Any[NaN for _ in 1:N]
    axs = Any[NaN for _ in 1:N]
    for i in 1:N
        f = CairoMakie.Figure()
        ax = CairoMakie.Axis(f[1,1], title="Region $(i)", ylabel="Portion of cells infected", xlabel="time (months)", xticks=0:9, limits=(0,9.1,0,1))
        fs[i] = f
        axs[i] = ax
    end
    posterior_samples = sample(chain, N_samples; replace=false)
    for sample in eachrow(Array(posterior_samples))
        # samples
        p = sample[2:(end-1)]  # first index is σ and last index is seed
        u0[seed] = sample[end]
        
        # solve
        sol_p = solve(prob,Tsit5(); p=p, u0=u0, saveat=0.1, abstol=1e-9, reltol=1e-6)
        t = sol_p.t
        if inference["transform_observable"]
            sol_p = tanh.(sol_p)
        end
        for i in 1:N
            CairoMakie.lines!(axs[i],t, sol_p[i,:]; alpha=0.3, color=:grey)
        end
    end

    # Plot simulation and noisy observations.
    for i in 1:N
        CairoMakie.scatter!(axs[i], timepoints, data[i,:]; colormap=:tab10)
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
    factors = [1., inference["factors"]..., 1.]  # add factor one for σ and seed
    priors = inference["priors"]

    prior_figs = []
    i = 1
    for (var, dist) in priors
        prior_i = StatsPlots.plot(dist * factors[i], title=var, ylabel="Density", xlabel="Sample value", legend=false)
        if !isempty(save_path)
            savefig(prior_i, save_path*"/prior_$(var).png")
        end
        push!(prior_figs, prior_i)
        i += 1
    end
    return 
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
    factors = [1.,inference["factors"]...,1.]  # add 1 scaling for σ and seed

    chain = inference["chain"]
    priors = inference["priors"]
    vars = collect(keys(priors))
    master_fig = StatsPlots.plot(chain) 
    prior_and_posterior_figs = []
    for (i,var) in enumerate(vars)
        plot_i = StatsPlots.plot(master_fig[i,2], title=var)
        StatsPlots.plot!(plot_i, priors[var]*factors[i])
        if !isempty(save_path)
            savefig(plot_i, save_path*"/prior_and_posterior_$(var).png")
        end
        push!(prior_and_posterior_figs,plot_i)
    end
    return prior_and_posterior_figs
end

#=
master plotting function (plot everything relevant to inference)
=#
function plot_inference(inference, save_path)
    # load inference simulation 
    display(inference["chain"])

    # create folder
    try
        mkdir(save_path);
    catch
    end
    # rescale the parameters according to the factor
    chain = inference["chain"]
    factors = inference["factors"]
    factor_matrix = diagm(factors)
    n_chains = size(new_chain)[3]
    for i in 1:n_chains
        chain[:,2:(end-13),i] = Array(chain[:,2:(end-13),i]) * factor_matrix
    end

    # plot
    predicted_observed(inference; save_path=save_path*"/predicted_observed");
    plot_chains(inference, save_path=save_path*"/chains");
    plot_priors(inference; save_path=save_path*"/priors");
    plot_posteriors(inference, save_path=save_path*"/posteriors");
    plot_retrodiction(inference; save_path=save_path*"/retrodiction");
    plot_prior_and_posterior(inference; save_path=save_path*"/prior_and_posterior");
    return nothing
end