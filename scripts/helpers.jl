#=
Helper functions for the project
=#


#=
create Laplacian matrix based on out degrees
=#
function laplacian_out(W)
    N = length(W[1,:])
    for i in N  # removing self-loops
        W[i,i] = 0
    end
    # create Laplacian from struct. connectome
    D = zeros(N,N)  # out-degree matrix
    for i in 1:N
        W[i,i] = 0
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
        if maximum(A[i,:]) >= a
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
function plot_chains(chain, path)
    N_pars = size(chain)[2]
    vars = chain.info.varname_to_symbol
    i = 1
    for (key,value) in vars
        chain_i = Chains(chain[:,i,:], [value])
        chain_plot_i = StatsPlots.plot(chain_i)
        save(path * "/chain_$(key).png",chain_plot_i)
        i += 1
    end
end

#=
Plot retrodiction of chain compared to data
=#
function plot_retrodiction(;data=nothing, chain=nothing, prob=nothing, path=nothing, timepoints=nothing, seed=0)
    N = size(data)[1]
    fs = Any[NaN for _ in 1:N]
    axs = Any[NaN for _ in 1:N]
        for i in 1:N
        f = Figure()
            ax = Axis(f[1,1], title="Region $(i)", ylabel="Portion of cells infected", xlabel="time (months)", xticks=0:9, limits=(0,9.1,nothing,nothing))
        fs[i] = f
        axs[i] = ax
    end
    posterior_samples = sample(chain, 300; replace=false)
    for sample in eachrow(Array(posterior_samples))
        # samples
        if seed>0  # means IC at seed region has posterior
            p = sample[2:(end-1)]
            u0 = [0. for _ in 1:N]
            u0[seed] = sample[end]
        else
            p = sample[2:end]
            u0 = [0. for _ in 1:N]
        end
        
        # solve
        sol_p = solve(prob,Tsit5(); p=p, u0=u0, saveat=0.1, abstol=1e-9, reltol=1e-6)
        for i in 1:N
            lines!(axs[i],sol_p.t, sol_p[i,:]; alpha=0.3, color=:grey)
        end
    end

    # Plot simulation and noisy observations.
    for i in 1:N
        scatter!(axs[i], timepoints, data[i,:]; colormap=:tab10)
        save(path * "/retrodiction_region_$(i).png", fs[i])
    end
end


