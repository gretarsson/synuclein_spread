#=
here we create a folder of analysis plots of interence results
=#
using PathoSpread

simulation = "DIFFG_RETRO"
display("Plotting simulations: $simulation")

# read file 
inference_obj = load_inference("simulations/"*simulation*".jls")
println(names(inference_obj["chain"]))
chain = inference_obj["chain"]
println(keys(chain))


using MCMCChains, DifferentialEquations, Distributions

function rhat_matrix(x::AbstractMatrix)
    # x is S × C: samples in rows, chains in columns
    S, C = size(x)
    S < 2 && error("Need at least 2 samples per chain for Rhat")
    C < 2 && error("Need at least 2 chains for Rhat")

    # chain means: 1 × C
    chain_means = vec(mean(x; dims=1))

    # chain variances: 1 × C (unbiased)
    chain_vars = vec(var(x; dims=1, corrected=true))

    # overall mean
    mean_overall = mean(chain_means)

    # between-chain variance
    B = S * var(chain_means; corrected=true)

    # within-chain variance
    W = mean(chain_vars)

    # marginal posterior variance estimate
    var_hat = ((S - 1) / S) * W + (1 / S) * B

    # Rhat
    return sqrt(var_hat / W)
end

function loglik(inference)
    chain       = inference["chain"]
    data        = inference["data"]
    priors      = inference["priors"]
    seed        = inference["seed_idx"]
    timepoints  = inference["timepoints"]
    u0          = inference["u0"]
    N_samples   = size(data, 3)
    vec_data_raw = vec(data)
    nonmissing   = findall(vec_data_raw .!== missing)
    vec_data     = Float64.(vec_data_raw[nonmissing])
    par_names   = chain.name_map.parameters

    n_obs = length(vec_data)


    # --- Handle Bayesian or deterministic seeding ---
    if get(inference, "bayesian_seed", false)
        seed_ch_idx = findall(n -> startswith(String(n), "seed"), par_names)  # should be in chronological order
        isempty(seed_ch_idx) && error("No seed parameters found in chain.name_map.parameters")
        #seed_ch_idx = sort(seed_ch_idx)
    else
        seed_ch_idx = nothing
    end


    # total number of posterior samples
    total_samples = length(eachrow(Array(chain)))

    # prepare ODE problem
    prob = make_ode_problem(
        odes[inference["ode"]];
        labels     = inference["labels"],
        Ltuple     = inference["L"],
        factors    = inference["factors"],
        u0         = inference["u0"],
        timepoints = timepoints
    )

    # figure out σ location
    par_names = chain.name_map.parameters
    sigma_idx = findfirst(==(Symbol(:σ)), par_names)
    N_pars    = findall(x -> x == "σ", collect(keys(priors)))[1] - 1

    loglik_all = Vector{Float64}(undef, total_samples)

    # Loop across samples in the stored order
    for (sample_i, sample_vec) in enumerate(eachrow(Array(chain)))
        p = sample_vec[1:N_pars]
        σ = sample_vec[sigma_idx]
        

        # seeding
        u0_s = copy(u0)

        if seed_ch_idx === nothing
            # deterministic seeding
            if isa(seed, Int)
                u0_s[seed] = inference["seed_value"]
            else
                for sidx in seed
                    u0_s[sidx] = inference["seed_value"]
                end
            end
        else
            # Bayesian seeding
            if isa(seed, Int)
                u0_s[seed] = sample_vec[seed_ch_idx[1]]
            else
                for (i,sidx) in enumerate(seed)
                    u0_s[sidx] = sample_vec[seed_ch_idx[i]]
                end
            end
        end


        # solve ODE
        sol = solve(prob, Tsit5(); p=p, u0=u0_s, saveat=timepoints,
                    abstol=1e-6, reltol=1e-6)
        sol_p = Array(sol[inference["sol_idxs"], :])

        # expand predictions for measurement replicates
        pred = vec(cat([sol_p for _ in 1:N_samples]..., dims=3))[nonmissing]

        # compute total log-likelihood
        ll = 0.0
        for i in 1:n_obs
            ll += logpdf(Normal(pred[i], σ), vec_data[i])
        end

        loglik_all[sample_i] = ll
    end

    # reshape into a matrix with samples in rows and chains in columns
    S, P, C = size(chain.value.data)
    loglik_mat = reshape(loglik_all, S, C)
    rhat = rhat_matrix(loglik_mat)
    return loglik_mat, rhat
end

new_inference = loglik(inference_obj)




# look at chains
display(inference_obj["chain"])
#new_chain = inference_obj["chain"][:,:,[2,3,4]]
#inference_obj["chain"] = new_chain
#save_inference("simulations/" * simulation * "_CUT.jls", inference_obj)

# plot
setup_plot_theme!()  # set plotting settings
display("Plotting inference results...")
plot_inference(inference_obj,"figures/inferences/"*simulation, plot_priors_posteriors=true)  
display("Plots saved to figures/inferences/"*simulation)
display("---------------------------------------------------")

# plot with training data
#setup_plot_theme!()  # set plotting settings
#data_full, timepoints_full = PathoSpread.process_pathology("data/total_path.csv", W_csv="data/W_labeled_filtered.csv")
#plot_inference(inference_obj,"figures/"*simulation; full_data=data_full, full_timepoints=timepoints_full)  
#



using CSV, DataFrames, LinearAlgebra

df = CSV.read("data/W_labeled_filtered.csv", DataFrame)
W = Matrix{Float64}(df[:,2:end])

col_sums = sum(W, dims=1)           # 1×N row vector
row_sums = sum(W, dims=2)           # 1×N row vector
D_out    = Diagonal(vec(col_sums))  # convert to N×N diagonal matrix
D_in    = Diagonal(vec(row_sums))  # convert to N×N diagonal matrix
C_in = exp(D_in^(-1/2)*W*D_in^(-1/2))
C_out = exp(D_out^(-1/2)*W*D_out^(-1/2))
Cs = C_out[:,[1,2,3]]
Cvec = mean(Cs,dims=2)


2 .+ 0.0 .* Cvec