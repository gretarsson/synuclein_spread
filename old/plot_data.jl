using DelimitedFiles
using CSV
using StatsBase
using Serialization
using Plots
using Statistics
using GLMakie
include("helpers.jl")
#=
In this file, we read the structural connectome and plot it.
We also read total_path.csv and plot each regions timeseries. remove ROIs not found in struct. connectome,
Morever, we rearrange it in the same order as the ROIs from the struct. conn.,
and average over mice for each region. 
We save the resulting regions x timepoint array as avg_total_path.csv
=#

# read data
# read strutural connectome
file_W = "data/W_labeled.csv"
W_labeled = readdlm(file_W, ',')
W_labels = W_labeled[2:end,1]
W = W_labeled[2:end,2:end]
N = size(W, 1)
L = laplacian_out(W)
writedlm("data/W.csv", W, ',')
writedlm("data/L_out.csv", L, ',')

# plot adjacency matrix
Plots.heatmap(W, title="Weighted adjacency matrix", xlabel="regions", ylabel="regions")
yflip!(true)
Plots.xticks!(0:50:N)
Plots.yticks!(reverse(0:50:N))
Plots.savefig("figures/adjacency_matrix.pdf")

# plot histogram of adjacency matrix elements
histogram(vec(W), xlims=(0,30), norm=true)

# read total pathology
file_total_path = "data/total_path.csv"
total_path_labeled = readdlm(file_total_path, ',')
total_path = total_path_labeled[2:end, 3:end]
total_path_rois = total_path_labeled[1, 3:end]
total_path_time = total_path_labeled[2:end,2]
total_path_mice = total_path_labeled[2:end,1]

# map the regions of the pathology data to the structural connectivity
region_map = []  # index i corresponds to region i (struct. conn.) in the pathology data 
for i in 1:N
    region_name = W_labels[i]
    path_index = findall(x->x==region_name, total_path_rois)
    if isempty(path_index)
        println("Warning: region $region_name not in data")
    end
    append!(region_map, path_index[1])
end

# rearrange the pathology data to reflect the order in struct. connectivity
N_mice = length(total_path_mice)
rearr_total_path = Array{Any}(undef,N_mice,N)
for i in 1:N
    path_index = region_map[i]
    for k in axes(rearr_total_path,1)
        if total_path[k,path_index] == "NA"
            rearr_total_path[k,i] = missing
        else
            rearr_total_path[k,i] = total_path[k, path_index]
        end
    end
end
# save total path in a dictionary based on timepoints, with regions in the same order as structural connectivity 
timepoints_map = total_path_labeled[2:end,2]
total_path_dict = OrderedDict{Any}{Any}()
for time in sort(unique(timepoints_map))
    total_path_t = rearr_total_path[timepoints_map .== time,:]
    total_path_dict[time] = total_path_t
end
serialize("data/total_path_dict.jls", total_path_dict)
# create 3D array (regions,timepoints,samples) where missing if not exist
max_samples = countmap(timepoints_map)[mode(timepoints_map)]
n_timepoints = length(unique(timepoints_map))
total_path_3D = Array{Union{Missing,Float64}}(missing,N,n_timepoints,max_samples)
for (ti,time) in enumerate(sort(unique(timepoints_map)))
    total_path_t = transpose(rearr_total_path[timepoints_map .== time,:])  # gives total_path (N x samples) array for timepoint time
    Nt,Mt = size(total_path_t)
    total_path_3D[1:Nt,ti,1:Mt] = total_path_t 
end
serialize("data/total_path_3D.jls", total_path_3D)
total_path_3D


# save average total path with missing
avg_total_path = mean3(total_path_3D)
avg_total_path = reshape(avg_total_path, (size(avg_total_path)...,1))
avg_total_path = identity.(avg_total_path)

serialize("data/avg_total_path.jls", avg_total_path)



# the data does not follow the same mice over time
# we therefore average over the mice at different time points
# because julia has no nanmean function, we need to discard NaNs ourselves
timepoints = sort(unique(total_path_time))
writedlm("data/timepoints.csv", timepoints, ',')
N_timepoints = length(timepoints)
average_total_path = zeros(N,N_timepoints)
for i in 1:N_timepoints
    timepoint = timepoints[i]    
    indexes_path = findall(x->x==timepoint, total_path_time)
    norm_factors = ones(N)*length(indexes_path)
    average_path_t = zeros(N)
    for k in eachindex(indexes_path) 
        index_path = indexes_path[k]
        for j in 1:N
            if ismissing(rearr_total_path[index_path,j])
                norm_factors[j] = norm_factors[j] - 1
            else
                average_path_t[j] = average_path_t[j] + rearr_total_path[index_path,j]
            end
        end
    end
    println(norm_factors[4])
    for j in 1:N
        average_path_t[j] = average_path_t[j]/norm_factors[j]
    end
    average_total_path[:,i] .= average_path_t
end

# plot all region total pathology
for i in axes(average_total_path, 1)
    using GLMakie: scatter!
    f = Figure()
    ax = Axis(f[1,1], title="$(total_path_rois[i])", ylabel="Portion of cells infected", xlabel="time (months)", xticks=0:9, limits=(0,9.2,nothing,nothing))
    CairoMakie.lines!(timepoints, average_total_path[i,:])
    scatter!(timepoints, average_total_path[i,:])
    filename = "figures/total_path/total_path_region_$(string(i)).jpeg"
    save(filename,f)
end

# save the rearranged regions x time total pathology file
writedlm("data/avg_total_path.csv", average_total_path, ',')


# plot distribution of data below certain theshold
data = average_total_path
nonnan_idxs = nonnan_rows(data)
m = 1000
ps = LinRange(minimum(data[nonnan_idxs,:]), 0.2,m)
portion_regions = Array{Float64}(undef,m)
sum(nonnan_idxs)
for (i,p) in enumerate(ps)
    larger_idxs = larger_rows(data,p)
    idxs = nonnan_idxs .* larger_idxs
    portion_regions[i] = sum(idxs) 
end
plt = StatsPlots.plot()
portion_regions
plot = StatsPlots.plot(ps, portion_regions, legend=false, ylim=(0,N))
save("figures/thresholding/threshold_total_path.png",plot)





