# simulate_spreading.jl
#
# Reads optimal parameters and initial conditions from CSV,
# integrates chosen ODE with DifferentialEquations.jl,
# saves results to CSV, and makes a simple overlay plot.

using CSV, DataFrames
using DifferentialEquations
using Plots
using PathoSpread 

# include ODE definitions (look in this file for ODE functions)
include("../src/odes.jl")

# --- choose input and output ---
inference_csv = "simulations/optimal_parameters/posterior_mode_DIFFG.csv"  # CSV file with optimal parameters and initial conditions
sim_csv       = "simulations/PathSim/simulation_output.csv"  # where to save simulation output
plot_png      = "simulations/PathSim/simulation_plot.png"  # where to save plot

# --- read parameters and initial conditions ---
df = DataFrame(CSV.File(inference_csv))  # read data file with optimal parameters and initial conditions 

pars = sort(filter(r -> r.category == "parameter", df), :index)
p = Float64.(pars.value)  # parameter vector

ics = sort(filter(r -> r.category == "initial_condition", df), :index)
u0 = Float64.(ics.value)  # initial conditions (state vector at time zero)
labels = String.(ics.name)  # name of brain regions

# --- choose ODE and Laplacian ---
ode! = DIFFG  # pick one: DIFF (diffusion/Kate's), DIFFG, DIFFGA, DIFFGAM (in increasing complexity)
L,N,labels = PathoSpread.read_W("data/W_labeled_filtered.csv", direction=:retro);  # my own function to write and set up Laplacian correctly, can be found in src/helpers.jl
N = size(L,1)
L_tuple = (L, N)

# --- solver settings ---
tspan = (0.0, 10.0)  # how long to run the simulation for
dt = 0.01  # how often to save time steps (every dt time unit)
ode_solver = RK4()  # pick method/algorithm to integrate ODE (these vary in speed and accuracy)

prob = ODEProblem((du,u,p,t) -> ode!(du,u,p,t; L=L_tuple), u0, tspan, p)  # set up ODE problem, we need a Laplacian L, initial conditions u0, time span, and parameters p
sol = solve(prob, ode_solver; dt=dt, adaptive=false, saveat=dt)  # solve ODE problem, we need to specify a solver (ode_solver), time step to save at (dt), and whether to use adaptive time stepping (adaptive=false). if using adaptive you save time, but we want it to be reproducible so we don't use it.

# --- save simulation to CSV ---
X = Array(sol)'  # timepoints × states. This is the output of solve(), i.e. the simulation
df_out = DataFrame(time=sol.t)
for (j, nm) in enumerate(labels)
    df_out[!, Symbol(nm)] = X[:, j]
end
CSV.write(sim_csv, df_out)  # save output of solve() to CSV

# --- plot (all series in one figure) ---
plot(sol.t, X, legend=false, xlabel="time", ylabel="state")
savefig(plot_png)

