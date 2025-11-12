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
