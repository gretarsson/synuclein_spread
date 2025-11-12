#!/usr/bin/env julia
using PathoSpread
using DifferentialEquations
using CairoMakie

# -------------------------------------------------------------
# LOAD DATA AND STRUCTURAL MATRICES
# -------------------------------------------------------------
data_file = "data/total_path.csv"
w_file = "data/W_labeled_filtered.csv"

data, timepoints = process_pathology(data_file; W_csv=w_file)

# FIND TOP N REGIONS WITH HIGHEST MEAN PATHOLOGY
meandata = PathoSpread.mean3(data)
# Step 1: compute per-row maxima (ignoring missings)
rowmax = [maximum(skipmissing(meandata[i, :])) for i in 1:size(meandata, 1)]
# Step 2: mask out rows that are entirely missing
mask = .!ismissing.(rowmax)
valid_idx = findall(mask)
# Step 3: sort valid rows by descending max value
sorted_idx = sort(valid_idx; by = i -> rowmax[i], rev = true)
# Step 4: take top N rows
N = 4
top_idx = sorted_idx[1:min(N, length(sorted_idx))]
top_vals = rowmax[top_idx]
println("Top $N regions with highest peak pathology:")
for (i, val) in zip(top_idx, top_vals)
    println("Row $i → max = $val")
end





# Read Laplacians
Lr, N, labels = read_W(w_file, direction=:retro)
La, _, _ = read_W(w_file, direction=:antero)

# -------------------------------------------------------------
# DEFINE ODE (DIFF model)
# -------------------------------------------------------------
function DIFF!(du, u, p, t; L)
    ρ = p[1]
    du .= -ρ .* (L * u)
end

# -------------------------------------------------------------
# SIMULATION SETTINGS
# -------------------------------------------------------------
ρ = 0.2               # diffusion rate
p = [ρ]
tspan = (0.0, 10.0)
seed_idx = 74         # iCP, striatum
u0 = zeros(N)
u0[seed_idx] = 10.0    # unit seeding

# -------------------------------------------------------------
# SOLVE ODE
# -------------------------------------------------------------
prob = ODEProblem((du,u,p,t)->DIFF!(du,u,p,t;L=Lr), u0, tspan, p)
sol = solve(prob, Tsit5(); reltol=1e-8, abstol=1e-10)
