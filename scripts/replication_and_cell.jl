using DifferentialEquations
using Plots
function f!(du,u,p,t)
    α,β,γ, d = p 
    x,y = u
    du[1] = α*x*(β*(1-y)-x)
    du[2] = γ*(d*tanh(x) - y)
end
    
u0 = [0.01,.01]
p = [10,0.25,0.1, 1]
tspan = (0.0,9)
prob = ODEProblem(f!,u0, tspan,p)
sol = solve(prob)
Plots.plot(sol)