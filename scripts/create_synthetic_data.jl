using DifferentialEquations
include("helpers.jl")  # Make sure helpers.jl defines the fastslow function

nvars = 50
ode = fastslow
# READ MATRIX
W_file = "./data/W_labeled.csv"
W_labelled = readdlm(W_file,',')
W = W_labelled[2:end,2:end]
W = W ./ maximum( W[ W .> 0 ] )  # normalize connecivity by its maximum
L = Matrix(transpose(laplacian_out(W; self_loops=false, retro=true)))  # transpose of Laplacian (so we can write LT * x, instead of x^T * L)
L = L[1:nvars,1:nvars]
L = L,nvars
# ---------------------------
# Define simulation parameters
# ---------------------------
# Initial condition for the ODE (adjust as needed)
u0 = [0.0 for _ in 1:(2*nvars)]  # Example: two variables; change according to your ODE
u0[1] = 0.1

# Define the time span and the timepoints for saving the solution
tspan = (0.0, 9)
num_timepoints = 8
timepoints = range(tspan[1], tspan[2], length=num_timepoints)

# PARAMETERS
α = 2
ρ = 0.6
β = [1 for i in 1:nvars]
d = [-1 for i in 1:nvars]
γ = 0.5
p = [ρ, α, β..., d..., γ]
factors = [1. for _ in 1:length(p)]

# Create the ODE problem using the fastslow function from helpers.jl
rhs(du,u,p,t;L=L, func=ode::Function) = func(du,u,p,t;L=L,factors=factors)  # uncomment without bilateral 
prob = ODEProblem(rhs, u0, tspan, p)
sol = solve(prob, Tsit5(), abstol=1e-10, reltol=1e-10)
Plots.plot(sol;idxs=1:nvars)


# ---------------------------
# Set up parameters for synthetic data
# ---------------------------
num_samples = 5           # Number of synthetic samples

# Pre-allocate a 3D array: variables x timepoints x samples
synthetic_data = Array{Union{Missing,Float64}}(undef, nvars, num_timepoints, num_samples)

# Noise standard deviation (adjust as needed)
noise_std = 0.001

# ---------------------------
# Simulate and add noise for each sample
# ---------------------------
for i in 1:num_samples
    # Solve the ODE for the current sample using a suitable solver (e.g., Tsit5)
    sol = solve(prob, Tsit5(), saveat=timepoints)
    sol_matrix = sol[1:nvars,:]
    
    # Convert the solution (a vector of vectors) into a 2D array:
    # rows correspond to variables, columns correspond to timepoints
    #sol_matrix = hcat(sol.u[1:nvars]...)  # Each sol.u[j] is the state at a timepoint
    
    # Add normally distributed noise to each data point
    noisy_sol = sol_matrix .+ noise_std * randn(size(sol_matrix)...)
    
    # Store the noisy solution in the 3D array
    synthetic_data[:, :, i] = noisy_sol
end

# ---------------------------
# (Optional) Output the dimensions of the generated data
# -------------------------4040--
synthetic_data[synthetic_data .< 0] .= 0
println("Synthetic data generated with dimensions: ", size(synthetic_data))
serialize("./data/synthetic_data_N=$(nvars).jls", synthetic_data);
serialize("./data/synthetic_timepoints_N=$(nvars).jls", Array(timepoints));