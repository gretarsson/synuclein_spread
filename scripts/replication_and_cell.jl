using DifferentialEquations
using Plots
function f!(du,u,p,t)
    α,γ,κ = p 
    x,y = u
    du[1] = α*x*(y-x)
    du[2] = -γ*(y-(1-κ*x))
end
    
u0 = [0.1,1.0]
p = [1.0,0.2,1.]
tspan = (0.0,20.0)
prob = ODEProblem(f!,u0, tspan,p)
sol = solve(prob)
plot(sol)