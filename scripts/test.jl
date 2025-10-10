using PathoSpread
using CSV, DataFrames, CairoMakie, Statistics
using LinearAlgebra

# --- Load adjacency matrix ---
w_file = "data/W_labeled_filtered.csv"
df = CSV.read(w_file, DataFrame)
labels = String.(names(df)[2:end])
W = Matrix{Float64}(df[:, 2:end])
W[diagind(W)] .= 0.0


# Compute the degree (sum of connections) for each node
outdegree = sum(W, dims=2)[:]   # sum across rows → outgoing connections
indegree  = sum(W, dims=1)[:]   # sum across columns → incoming connections

# Nodes with zero in both in/out degrees
isolated_idx = findall((outdegree .+ indegree) .== 0)

println("Number of isolated nodes: ", length(isolated_idx))
println("Isolated node indices: ", isolated_idx)
println("Isolated node labels: ", labels[isolated_idx])

outdegree = sum(W, dims=2)[:]   # sum across each row
no_out_idx = findall(outdegree .== 0)

println("Number of nodes with no outgoing edges: ", length(no_out_idx))
println("Indices: ", no_out_idx)
println("Labels: ", labels[no_out_idx])


indegree = sum(W, dims=1)[:]   # sum across columns
no_in_idx = findall(indegree .== 0)

println("Number of nodes with no incoming edges: ", length(no_in_idx))
println("Indices: ", no_in_idx)
println("Labels: ", labels[no_in_idx])




# Your suspect seed indices
suspect = [105, 201, 131, 156, 122, 167, 139, 185, 141, 159, 126, 100,
           182, 194, 54, 95, 172, 197, 81, 104, 149, 169, 180, 80, 124,
           171, 98, 191, 166, 195, 190, 130, 174, 161, 183, 88, 160, 103,
           57, 168, 125, 67, 140, 121, 128]

# Compare outdegree statistics
println("\n--- Outdegree analysis for suspect seeds ---")
println("Mean outdegree (suspects): ", mean(outdegree[suspect]))
println("Mean outdegree (others):   ", mean(outdegree[setdiff(1:length(outdegree), suspect)]))
println("Min suspect outdegree:     ", minimum(outdegree[suspect]))
println("Max suspect outdegree:     ", maximum(outdegree[suspect]))

# Sort by outdegree for inspection
sorted = sortperm(outdegree[suspect])
println("\nLowest outdegree suspects:")
for i in sorted[1:10]
    idx = suspect[i]
    println("  seed ", idx, " → outdegree=", round(outdegree[idx], digits=4))
end

# Compare indegree statistics
println("\n--- Indegree analysis for suspect seeds ---")
println("Mean indegree (suspects): ", mean(indegree[suspect]))
println("Mean indegree (others):   ", mean(indegree[setdiff(1:length(indegree), suspect)]))
println("Min suspect indegree:     ", minimum(indegree[suspect]))
println("Max suspect indegree:     ", maximum(indegree[suspect]))

# Sort by indegree for inspection
sorted_in = sortperm(indegree[suspect])
println("\nLowest indegree suspects:")
for i in sorted_in[1:10]
    idx = suspect[i]
    println("  seed ", idx, " → indegree=", round(indegree[idx], digits=4))
end


using Graphs, SimpleWeightedGraphs, Statistics

# --- Build graph ---
g = SimpleWeightedDiGraph(W)

# --- Centrality measures ---
deg_centrality = degree_centrality(g)
betweenness_centrality = betweenness_centrality(g)
eig_centrality = eigenvector_centrality(g)

# --- Compare suspects vs others ---
others = setdiff(1:length(labels), suspect)

function compare_metric(name, metric)
    println("\n--- $(name) ---")
    println("Mean (suspects): ", round(mean(metric[suspect]), digits=4))
    println("Mean (others):   ", round(mean(metric[others]), digits=4))
    println("Min suspect: ", round(minimum(metric[suspect]), digits=4))
    println("Max suspect: ", round(maximum(metric[suspect]), digits=4))
end

compare_metric("Degree centrality", deg_centrality)
#compare_metric("Betweenness centrality", betweenness_centrality)
compare_metric("Eigenvector centrality", eig_centrality)

# --- Sort and inspect lowest suspects ---
sorted_low = sortperm(deg_centrality[suspect])
println("\nLowest-degree suspects:")
for i in sorted_low[1:10]
    idx = suspect[i]
    println("  $(labels[idx]) → degree=", round(deg_centrality[idx], digits=4))
end



using Graphs, SimpleWeightedGraphs
using CairoMakie
using NetworkLayout

# --- Build graph ---
g = SimpleWeightedDiGraph(W)

# --- Compute layout ---
A = Matrix(adjacency_matrix(g))
coords = NetworkLayout.spring(A)  # Vector of Point{2, Float64}

# Unpack coordinates
xs = [p[1] for p in coords]
ys = [p[2] for p in coords]

# --- Define node colors ---
node_colors = fill(RGBf(0.7, 0.7, 0.7), nv(g))  # default grey

for i in suspect
    node_colors[i] = RGBf(0.8, 0.1, 0.1)  # red
end

true_seed = 74  # replace with your actual true seed index
node_colors[true_seed] = RGBf(0.0, 0.2, 0.8)  # blue

# --- Plot ---
fig = Figure(resolution=(800, 800))
ax = Axis(fig[1, 1], title="Network connectivity: true (blue), suspects (red)")

# Draw edges
for e in edges(g)
    src_idx = Graphs.src(e)
    dst_idx = Graphs.dst(e)
    lines!(ax, [xs[src_idx], xs[dst_idx]], [ys[src_idx], ys[dst_idx]],
           color=:gray80, linewidth=0.5)
end

# Draw nodes
scatter!(ax, xs, ys, color=node_colors, markersize=10)

hidedecorations!(ax)
hidespines!(ax)

save("figures/network_suspects_vs_true.pdf", fig)
fig
