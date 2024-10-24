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
    #du[(N+1):(2*N)] .=  γ .* (d .* x .- y)  
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
#=
a dictionary containing the ODE functions
=#
odes = Dict("diffusion" => diffusion, "diffusion2" => diffusion2, "diffusion3" => diffusion3, "diffusion_pop2" => diffusion_pop2, "aggregation" => aggregation, 
            "aggregation2" => aggregation2, "aggregation_pop2" => aggregation_pop2, "death_local2" => death_local2, "aggregation2_localα" => aggregation2_localα,
            "death_superlocal2" => death_superlocal2, "death2" => death2, "death_all_local2" => death_all_local2, "death" => death)



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
               transform_observable=false,
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
    W = W ./ maximum( W[ W .> 0 ] )  # normalize connecivity by its maximum, but this also slows MCMC down substantially...
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

    ## EXPERIMENTAL make data array with rows that only have their (uniquely sized) nonmissing columns
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

    @model function bayesian_model(data, prob; ode_priors=priors_vec, priors=priors, alg=alg, timepointss=timepoints::Vector{Float64}, seedd=seed::Int, u0=u0::Vector{Float64}, bayesian_seed=bayesian_seed::Bool, seed_value=seed_value,
                                    N_samples=N_samples,
                                    nonmissing=nonmissing::Vector{Int64},
                                    row_nonmiss=row_nonmiss::Vector{Vector{Int}}
                                    )
        u00 = u0  # IC needs to be defined within model to work
        # priors
        p ~ arraydist([ode_priors[i] for i in 1:N_pars])
        σ ~ priors["σ"] 
        if bayesian_seed
            u00[seedd] ~ priors["seed"]  
        else
            u00[seedd] = seed_value
        end

        # Simulate diffusion model 
        #predicted = solve(prob, alg; u0=u00, p=p, saveat=timepointss, sensealg=sensealg, abstol=abstol, reltol=reltol, maxiters=1000)
        predicted = solve(prob, alg; u0=u00, p=p, saveat=timepointss, sensealg=sensealg, abstol=abstol, reltol=reltol, dt=1e-3)

        # Transform prediction
        predicted = predicted[sol_idxs,:]
        if transform_observable
            predicted = tanh.(predicted)
        end

        ## Observations. REAL one
        predicted = vec(cat([predicted for _ in 1:N_samples]...,dims=3))
        predicted = predicted[nonmissing]
        for pred in predicted
            if pred < 0 || pred >= 1
                display("Negative u(t) -> Likelihood set to -Inf")
                Turing.Turing.@addlogprob! -Inf
                return nothing
            end
        end
        data ~ MvNormal(predicted,σ^2*I)  # do30es not work with psis_loo, but mucher faster
        return nothing
        #data ~ MvNormal(predicted,diagm((predicted .+ 1e-2)))  # trying out mvnormal with poisson
        #data ~ arraydist([ truncated(Normal(predicted[i],σ^2),lower=0,upper=1) for i in 1:size(predicted)[1] ])  # this works really well, took hella long though 
        #data ~ arraydist([ Normal(predicted[i], σ^2 * predicted[i]+1e-2) for i in 1:size(predicted)[1] ])  # this works really well too, took hella long though 
        data ~ arraydist([ Normal(predicted[i], σ^2 * predicted[i]*(1-predicted[i])+1e-2) for i in 1:size(predicted)[1] ])  # testing 
        return nothing

        # THIS WORKS
        #if global_variance
        #    # common variance among all regions NORMAL
        #    predicted = vec(cat([predicted for _ in 1:N_samples]...,dims=3))
        #    predicted = predicted[nonmissing]
        #    data ~ MvNormal(predicted,σ^2*I)  # does not work with psis_loo, but mucher faster
        #else
        #    # region-specific variances PER ROW
        #    predicted = cat([predicted for _ in 1:N_samples]...,dims=3)
        #    for k in axes(data,1)  
        #        nonmissing_k = row_nonmiss[k]
        #        predicted_sub = vec(predicted[k,:,:])
        #        data[k] ~ MvNormal(predicted_sub[nonmissing_k], σ[k]^2*I)
        #        #data[k] ~ arraydist([ truncated(Normal(predicted_sub[ind], σ[k]^2), lower=0,upper=1) for ind in nonmissing_k ] )
        #    end
        #end
        # exp is local definition causing memor: issues? yes, it does seem like it
        #predicted = vec(cat([predicted for _ in 1:N_samples]...,dims=3))
        #predicted = predicted[nonmissing]
        #data ~ MvNormal(predicted,σ^2*I)  # does not work with psis_loo, but mucher faster

        # EXP
        #predicted = repeat(vec(predicted),N_samples)[nonmissing]  # reformats predicted to use nonmissing (verified) but is not quicker?
        #data ~ MvNormal(predicted,σ^2*I)  # does not work with psis_loo, but mucher faster

        # put noise into the data
        # custom noise
        #predicted = predicted[sol_idxs,:]
        #predicted = vec(cat([predicted for _ in 1:N_samples]...,dims=3))
        #predicted = predicted[nonmissing]
        ##predicted = tanh.(predicted)
        #for i in eachindex(data)
        #    #data[i] ~ LogitNormal(predicted[i],σ^2)
        #    data[i] ~ truncated(Normal(predicted[i],σ^2),lower=0)
        #    #data[i] ~ abs(predicted[i]) + InverseGamma(a,b)
        #    #data[i] ~ predicted[i] + LogNormal(a,b)
        #    #data[i] ~ TanhNormal(predicted[i],σ^2)
        #    #data[i] ~ (predicted[i] + Beta(a))/2
        #end
        # ind variance
        #predicted = predicted[sol_idxs,:]
        #predicted = tanh.(predicted)
        ##predicted = cat([predicted for _ in 1:N_samples]...,dims=3)
        ## try out brownian motion noise
        #for i in axes(data,1)  
        #    for j in axes(data,2)
        #        #time = timepoints[j]
        #        pred = predicted[i,j]
        #        for k in axes(data,3)
        #            if !ismissing(data[i,j,k])
        #                data[i,j,k] ~ TanhNormal(pred, σ[i]^2*(pred+1e-3))
        #            end
        #        end
        #    end
        #end
        #for k in axes(data,1)  
        #    nonmissing_k = row_nonmiss[k]
        #    predicted_sub = vec(predicted[k,:,:])[nonmissing_k]
        #    for j in 1:size(data[k],1)
        #        data[k][j] ~ truncated(Normal(predicted_sub[j], σ[k]^2),lower=0)
        #        #data[k][j] ~ TanhNormal(tanh(predicted_sub[j]), σ[k]^2)
        #    end
        #    #data[k] ~ MvNormal(predicted_sub, σ[k]^2*I)
        #    #data[k] ~ arraydist([Normal(predicted_sub[j], σ[k]^2) for j in 1:)
        #end

        #return nothing
    end

    # define Turing model
    #model = bayesian_model(vec_data, prob)  # NORMAL
    #model = bayesian_model(row_data, prob)  # PER ROW
    model = bayesian_model(final_data, prob)  # NEW


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
        chain = sample(model, NUTS(1000,0.65;adtype=adtype), 1000; progress=true)  # time estimated is shown
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
    display(chain)  
    inference = Dict("chain" => chain, 
                     "priors" => priors, 
                     "data" => data,
                     "timepoints" => timepoints,
                     "data_indices" => idxs, 
                     "seed_idx" => seed,
                     "bayesian_seed" => bayesian_seed,
                     "seed_value" => seed_value,
                     "transform_observable" => transform_observable,
                     "ode" => string(ode),  # store var name of ode (functions cannot be saved)
                     "factors" => factors,
                     "sol_idxs" => sol_idxs,
                     "u0" => u0,
                     "L" => L
                     #"elpd" => elpd,
                     #"waic" => waic
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
    u0[seed] = chain["seed"][argmax]  

    # solve ODE
    sol = solve(prob,Tsit5(); p=p, u0=u0, saveat=timepoints, abstol=1e-9, reltol=1e-6)
    sol = Array(sol[sol_idxs,:])
    if inference["transform_observable"]
        if haskey(priors,"c")
            c = chain["c"][argmax]
            amplitude = 1 .- c*sol
        else
            amplitude = 1
        end
        if haskey(priors,"k")
            k = chain["k"][argmax]
            input = k*sol
        else
            input = sol
        end
        sol = amplitude .* tanh.(input)
    end

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
    minxy = 0
    if plotscale==log10 && ((sum(x .== 0) + sum(y .== 0)) > 0)  # if zeros present, add the smallest number in plot
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
        ax = CairoMakie.Axis(f[1,1], title="t = $(timepoints[i])", ylabel="Predicted", xlabel="Observed", xscale=plotscale, yscale=plotscale, xticks=xticks, yticks=yticks)

        # as we are plotting log-log, we account for zeros in the data
        x = vec(copy(data[:,i]))
        y = vec(copy(sol[regions,i]))
        nonmissing = findall(x .!== missing)
        x = x[nonmissing]
        y = y[nonmissing]
        if plotscale==log10 && ((sum(x .== 0) + sum(y .== 0)) > 0)  # if zeros present, add the smallest number in plot
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
        f = CairoMakie.Figure()
        ax = CairoMakie.Axis(f[1,1], title="Region $(i)", ylabel="Portion of cells infected", xlabel="time (months)", xticks=0:9, limits=(0,9.1,0,1))
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
        
        # solve
        sol_p = solve(prob,Tsit5(); p=p, u0=u0, saveat=0.1, abstol=1e-9, reltol=1e-6)
        t = sol_p.t
        sol_p = Array(sol_p[sol_idxs,:])
        if inference["transform_observable"]
            if haskey(priors,"c")
                c_ch_idx = findall(x->x==:c,par_names)[1]
                c = sample[c_ch_idx]
                amplitude = 1 .- c*sol_p
            else
                amplitude = 1
            end
            if haskey(priors,"k")
                k_ch_idx = findall(x->x==:c,par_names)[1]
                k = sample[k_ch_idx]
                input = k*sol_p
            else
                input = sol_p
            end
            sol_p = amplitude .* tanh.(input)
        end
        for i in 1:N
            CairoMakie.lines!(axs[i],t, sol_p[i,:]; alpha=0.3, color=:grey)
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
        CairoMakie.scatter!(axs[i], timepoints_i, data_i; color=RGB(0/255, 0/255, 139/255), alpha=1.)  
        # have lower std capped at 0.01 (to be visible in the plots)
        var_data_i_lower = copy(var_data_i)
        for (n,var) in enumerate(var_data_i)
            if sqrt(var) > data_i[n]
                var_data_i_lower[n] = max(data_i[n]^2-0.01, 0)
                #var_data_i_lower[n] = data_i[n]^2
            end
        end

        #CairoMakie.errorbars!(axs[i], timepoints_i, data_i, sqrt.(var_data_i); color=RGB(0/255, 71/255, 171/255), whiskerwidth=20, alpha=0.2)
        CairoMakie.errorbars!(axs[i], timepoints_i, data_i, sqrt.(var_data_i_lower), sqrt.(var_data_i); color=RGB(0/255, 71/255, 171/255), whiskerwidth=20, alpha=0.2)
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
            CairoMakie.scatter!(axs[i], timepoints_i, data_i; color=RGB(0/255, 71/255, 171/255), alpha=0.4)  
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
function plot_inference(inference, save_path; plotscale=log10)
    # load inference simulation 
    display(inference["chain"])

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
    #predicted_observed(inference; save_path=save_path*"/predicted_observed", plotscale=plotscale);
    plot_retrodiction(inference; save_path=save_path*"/retrodiction");
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
