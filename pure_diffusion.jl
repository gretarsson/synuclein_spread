using DifferentialEquations
using DelimitedFiles
using Plots

# create random Laplacian
N = 10
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

# read strutural connectome
file_W = "C:/Users/cga32/Desktop/synuclein_spread/data/W_labeled.csv"
W_labeled = readdlm(file_W, ',')
W = W_labeled[2:end,2:end]
N = size(W, 1)

# create Laplacian from struct. connectome
D = zeros(N,N)  # out-degree matrix
for i in 1:N
    W[i,i] = 0
    D[i,i] = sum(W[i,:])
end
L = D - W 

# define ODE
function diffusion(du,u,p,t;L=L)
    ρ = p
    du .= -ρ*u*L  # u is a row vector
end

# settings for ODE
u0 = transpose([0.0 for i in 1:N])  # initial conditions
u0[1] = N  # seed
p = 1.
tspan = (0.0,10.0)

# run ODE
prob = ODEProblem(diffusion,u0, tspan,p)
sol = solve(prob, Tsit5())
plot(sol)



