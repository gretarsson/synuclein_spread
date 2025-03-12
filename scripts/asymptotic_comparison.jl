using DifferentialEquations, Plots, QuadGK

# Parameters
alpha  = 0.5       # Reaction rate parameter
beta   = 2.0       # Carrying capacity
u0     = 0.1       # Initial value for seeded region u_s(0)
epsilon = 0.01     # Small parameter for diffusion strength
L_is   = -1.0       # Coupling from seeded index s to nonseeded index i (L[i,s])
t0     = 0.5       # Matching (transition) time
delta  = 0.1       # Width parameter for the blending function

# Seeded solution u_s(t): logistic growth
u_s(t) = (beta * u0 * exp(alpha*beta*t)) / (beta + u0*(exp(alpha*beta*t)-1))

# Inner solution for u_i (nonseeded component)
# u_inner(t) = -ε * L_is * ∫₀ᵗ u_s(τ)dτ (computed numerically)
#function u_inner(t)
#    I, err = quadgk(u_s, 0, t)
#    return -epsilon * L_is * I
#end
function u_inner(t, u0, α, β, l, ρ)
    return exp(t * α * β) * l * u0 * ρ * (
        log(β) - log(-α * β * (-u0 + β)) + 
        log(-exp(t * α * β) * α * β * (-u0 + β)) - 
        log((-1 + exp(t * α * β)) * u0 + β)
    ) / (α * (-u0 + β))
end


# Determine u1 = u_inner(t0) to be used as the matching value.
u1 = u_inner(t0)

# Outer solution: logistic form with matching constant C.
# Here we choose C so that u_outer(t0) = u1.
# That is, C = (β/u1 - 1) * exp(αβ*t0).
#C = (beta/u1 - 1) * exp(alpha*beta*t0)
#u_outer(t) = beta / (1 + C * exp(-alpha*beta*t))
function u_outer(t, t0, u1, α, β)
    return (exp(t * α * β) * u1 * β) / (exp(t * α * β) * u1 + exp(t0 * α * β) * (-u1 + β))
end



# Blending (transition) function, switching from 0 to 1 near t0.
phi(u) = 0.5 * (1 + tanh((u - 0.1)/delta))

# Composite (smooth blended) approximate solution.
u_approx(t) = (1 - phi(u_inner(t)))*u_inner(t) + phi(u_inner(t))*u_outer(t)

# Define the full ODE for u_i(t):
# du_i/dt = -ε * L_is * u_s(t) + α * u_i(t)*(β - u_i(t))
function f!(du, u, p, t)
    du[1] = epsilon * (L_is * u[1] - L_is * u[2]) + alpha * u[1] * (beta - u[1])
    du[2] = epsilon * (L_is * u[2] - L_is * u[1]) + alpha * u[2] * (beta - u[2])
end

# Initial condition for u_i: 0
u0_ode = [0.0, u0]
tspan = (0.0, 10)
prob = ODEProblem(f!, u0_ode, tspan)
sol = solve(prob, Tsit5())

# Prepare time points for plotting.
t_vals = range(tspan[1], tspan[2], length=500)
u_ode  = [sol(t)[1] for t in t_vals]
u_app  = [u_approx(t) for t in t_vals]

# Plot the numerical ODE solution and the smooth blended approximate solution.
Plots.plot(t_vals, u_ode, label="Numerical ODE solution", lw=2)
Plots.plot!(t_vals, u_app, label="Smooth blended approx.", lw=2, ls=:dash)
Plots.xlabel!("t")
Plots.ylabel!("u_i(t)")
Plots.title!("Comparison: ODE solution vs. Blended Approximation")
