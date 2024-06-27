using Turing, StatsPlots
using ReverseDiff, Zygote, SciMLSensitivity

#=
Minimal example of Bayesian inference of ODE parameters
Cannot make reversediff work
=#


# define model
using DifferentialEquations
using OrdinaryDiffEq
function ode!(du,u,p,t)
    du[1] = -p[1]*u[1]
end
u0 = [1.]
p = [0.5]
tspan=(0.,1.)
prob = ODEProblem(ode!,u0,tspan,p)
sol = solve(prob,Tsit5();saveat=0.1)
plot(sol)
# define synthetic data
data = sol[1,:]

@model function globe_toss(data, prob)
    # prior
    c ~ Normal(0.,0.5)
    # likelihood
    predicted = solve(prob,Tsit5();p=[c],saveat=0.1)
    for i in length(predicted)
        data[i] ~ Normal(predicted[1,i], 0.1)
    end
end

# infer posterior distribution
model = globe_toss(data, prob)
sampler = NUTS(;adtype=AutoReverseDiff())
#sampler = NUTS(;adtype=AutoForwardDiff())
samples = 1_000
chain = sample(model, sampler, samples)

# visualize results
StatsPlots.plot(chain)