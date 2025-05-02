# odes.jl

module ODEs

export DIFFGAM, DIFFGAM_bilateral

using LinearAlgebra
using Statistics

#### Plain (per‑region) ODEs ####

function diffusion(du,u,p,t;L=L,factors=nothing)
    L, _ = L
    ρ = p[1]

    du .= -ρ*L*u 
end
function diffusion2(du,u,p,t;L=(La, Lr),factors=nothing)
    La, Lr = L
    ρa = p[1]
    ρr = p[2]

    #du .= -(ρa*La+ρr*Lr)*u   # this gives very slow gradient computation
    du .= -ρa*ρr*La*u - ρa*Lr*u   # quick gradient computation
end
function diffusion3(du,u,p,t;L=L,N=1::Int, factors=nothing)
    ρ = p[1]
    γ = p[2]
    x = u[1:N]
    y = u[(N+1):(2*N)]

    du[1:N] .= -ρ*L*x
    du[(N+1):(2*N)] .= γ .* tanh.(x) .-  γ .* y
end
function diffusion_pop2(du,u,p,t;L=(La,Lr,N), factors=nothing)
    La, Lr, N = L
    ρa = p[1]
    ρr = p[2]
    γ = p[3]
    x = u[1:N]
    y = u[(N+1):(2*N)]

    du[1:N] .= -ρa*La*x-ρr*Lr*x
    #du[(N+1):(2*N)] .= 1/γ .* tanh.(x) .-  1/γ .* y
    du[(N+1):(2*N)] .= 1/γ .* x .-  1/γ .* y
end
function aggregation(du,u,p,t;L=L,factors=(1.,1.))
    L, _ = L
    kα,kβ = factors 
    ρ = p[1]
    α = kα * p[2]
    β = kβ .* p[3:end]

    du .= -ρ*L*u .+ α .* u .* (β .- u)  
end
function aggregation2(du,u,p,t;L=L,factors=(1.,1.))
    La, Lr = L
    p = factors .* p
    ρa = p[1]
    ρr = p[2]
    α = p[3]
    β = p[4:end]

    du .= -ρa*ρr*La*u .- ρa*Lr*u .+ α .* u .* (β .- u)   # quick gradient computation
end
function aggregation2_localα(du,u,p,t;L=L,factors=(1.,1.))
    La, Lr = L
    N = length(u)
    p = factors .* p
    ρa = p[1]
    ρr = p[2]
    α = p[3:(N+2)]
    β = p[(N+3):end]

    du .= -ρa*ρr*La*u .- ρa*Lr*u .+ α .* u .* (β .- u)   # quick gradient computation
end
function aggregation_pop2(du,u,p,t;L=L,factors=(1.,1.))
    La, Lr, N = L   
    p = factors .* p
    ρa = p[1]
    ρr = p[2]
    α = p[3]
    γ = p[4]
    β = p[5:end]


    x = u[1:N]
    y = u[(N+1):(2*N)]
    du[1:N] .= -ρa*ρr*La*x .- ρa*Lr*x .+ α .* x .* (β .* (1 .- y) .- x)   # quick gradient computation
    du[(N+1):(2*N)] .=  γ .* (tanh.(x) .- y)  
end
function death_local2(du,u,p,t;L=L,factors=(1.,1.))
    La, Lr, N = L  
    p = factors .* p
    ρa = p[1]
    ρr = p[2]
    α = p[3]
    β = p[4:(N+3)]
    #d = p[(N+4):(2*N+3)]
    d = p[N+4]
    γ = p[end]
    #α = p[3:N+2]
    #β = p[N+3:(2*N+2)]
    #γ = p[(2*N+3):(3*N + 2)]
    #ϵ = p[end-1]
    #d = p[end]


    x = u[1:N]
    y = u[(N+1):(2*N)]
    du[1:N] .= -ρa*ρr*La*x .- ρa*Lr*x .+ α .* x .* (β .* (1 .- d.*y) .- x)   # quick gradient computation
    du[(N+1):(2*N)] .=  γ .* (tanh.(x) .- y)  
    #du[(N+1):(2*N)] .=  ϵ .* (γ .* x .- y)  
    #du[(N+1):(2*N)] .=  (tanh.(x) .- y) ./ γ
end
function death(du,u,p,t;L=L,factors=(1.,1.))
    L,N = L
    p = factors .* p
    ρ = p[1]
    α = p[2]
    β = p[3:(N+2)]
    d = p[(N+3):(2*N+2)]
    γ = p[end]

    x = u[1:N]
    y = u[(N+1):(2*N)]
    du[1:N] .= -ρ*L*x .+ α  .* x .* (β .- d .* y .- x)   # quick gradient computation
    du[(N+1):(2*N)] .=  γ .* (1 .- y)  
end
function death_simplified(du,u,p,t;L=L,factors=(1.,1.))
    L,N = L
    p = factors .* p
    ρ = p[1]
    α = p[2]
    β = p[3:(N+2)]
    γ = p[end]

    x = u[1:N]
    y = u[(N+1):(2*N)]
    du[1:N] .= -ρ*L*x .+ α  .* x .* (β .- β .* y .- x)   # quick gradient computation
    du[(N+1):(2*N)] .=  γ .* (1 .- y)  
end
function death_simplifiedii(du,u,p,t;L=L,factors=(1.,1.))
    L,N = L
    p = factors .* p
    ρ = p[1]
    α = p[2]
    β = p[3:(N+2)]
    d = p[(N+3):(2*N+2)]
    γ = p[2*N+3]

    x = u[1:N]
    y = u[(N+1):(2*N)]
    du[1:N] .= -ρ*L*x .+ α .* x .* (β .- d .* y .- x)   # quick gradient computation
    du[(N+1):(2*N)] .=  γ .* (1 .- y) .* x  
end
function death_simplifiedii_regionaltime(du,u,p,t;L=L,factors=(1.,1.))
    L,N = L
    p = factors .* p
    ρ = p[1]
    α = p[2]
    β = p[3:(N+2)]
    d = p[(N+3):(2*N+2)]
    γ = p[2*N+3]

    x = u[1:N]
    y = u[(N+1):(2*N)]
    du[1:N] .= -ρ*L*x .+ β .* x .* (α .- γ .* y .- x)   # quick gradient computation
    du[(N+1):(2*N)] .=  d .* (1 .- y) .* x  
end
function heterodimer_inspired(du,u,p,t;L=L,factors=(1.,1.))
    L,N = L
    p = factors .* p
    ρ = p[1]
    α = p[2]
    β = p[3:(N+2)]
    d = p[(N+3):(2*N+2)]

    x = u[1:N]
    y = u[(N+1):(2*N)]
    du[1:N] .= -ρ*L*x .+ α .* x .* (β .- y .- x)   # quick gradient computation
    du[(N+1):(2*N)] .=  d .* x  
end
function heterodimer_inspiredii(du,u,p,t;L=L,factors=(1.,1.))
    L,N = L
    p = factors .* p
    ρ = p[1]
    α = p[2]
    β = p[3:(N+2)]
    d = p[(N+3):(2*N+2)]

    x = u[1:N]
    y = u[(N+1):(2*N)]
    du[1:N] .= -ρ*L*x .+ α .* x .* (β .- y .- x)   # quick gradient computation
    du[(N+1):(2*N)] .=  d .* x  
end
function fastslow(du,u,p,t;L=L,factors=(1.,1.))
    L,N = L
    p = factors .* p
    ρ = p[1]
    α = p[2]
    β = p[3:(N+2)]
    γ = p[(N+3):(2*N+2)]
    μ = p[2*N+3]

    x = u[1:N]
    y = u[(N+1):(2*N)]
    du[1:N] .= -ρ*L*x .+ α .* x .* (β .+ y .- x)   # quick gradient computation
    du[(N+1):(2*N)] .=  - μ .* y .* (1 .+ y.^2) .+ γ .* x
end
function fastslow_reparam(du,u,p,t;L=L,factors=(1.,1.))
    L,N = L
    p = factors .* p
    ρ = p[1]
    α = p[2]
    β = p[3:(N+2)]
    γ = p[(N+3):(2*N+2)]
    μ = p[2*N+3]
    b = p[2*N+4]

    x = u[1:N]
    y = u[(N+1):(2*N)]
    du[1:N] .= -ρ*L*x .+ α .* x .* (β .+ y .- x)   # quick gradient computation
    du[(N+1):(2*N)] .=  - μ .* y .* (1 .+ y.^2) .+ (γ .+ b .* β) .* x
end
function fastslow_reparamii(du,u,p,t;L=L,factors=(1.,1.))
    L,N = L
    p = factors .* p
    ρ = p[1]
    α = p[2]
    β = p[3:(N+2)]
    γ = p[(N+3):(2*N+2)]
    μ = p[2*N+3]
    a = p[2*N+4]

    x = u[1:N]
    y = u[(N+1):(2*N)]
    du[1:N] .= -ρ*L*x .+ α .* x .* (β .+ y .- x)   # quick gradient computation
    du[(N+1):(2*N)] .=  - μ .* y .* (1 .+ y.^2) .+ (a .+ γ .* β) .* x
end
function fastslow_regionaltime(du,u,p,t;L=L,factors=(1.,1.))
    L,N = L
    p = factors .* p
    ρ = p[1]
    β = p[2]
    α = p[3:(N+2)]
    μ = p[(N+3):(2*N+2)]
    γ = p[2*N+3]

    x = u[1:N]
    y = u[(N+1):(2*N)]
    du[1:N] .= -ρ*L*x .+ α .* x .* (β .+ y .- x)   # quick gradient computation
    du[(N+1):(2*N)] .=  - μ .* y .* (1 .+ y.^2) .+ γ .* x
end
function death_simplifiedii_uncor(du,u,p,t;L=L,factors=(1.,1.))
    L,N = L
    p = factors .* p
    ρ = p[1]
    α = p[2]
    β = p[3:(N+2)]
    d = p[(N+3):(2*N+2)]
    γ = p[2*N+3]

    x = u[1:N]
    y = u[(N+1):(2*N)]
    du[1:N] .= -ρ*L*x .+ α .* x .* (β .- β .* y - d .* y .- x)   # quick gradient computation
    du[(N+1):(2*N)] .=  γ .* (1 .- y) .* x  
end
function brennan(du,u,p,t;L=L,factors=(1.,1.))
    L,N = L
    p = factors .* p
    ρ = p[1]
    α = p[2]
    β = p[3:(N+2)]
    d = p[(N+3):(2*N+2)]
    γ = p[2*N+3]
    λ = p[2*N+4]

    x = u[1:N]
    y = u[(N+1):(2*N)]
    du[1:N] .= -ρ*L*x .+ α .* x .* (β .- λ .- y .- x)   # quick gradient computation
    du[(N+1):(2*N)] .=  γ .* (d .- λ .- y) .* x  
end
function brennanii(du,u,p,t;L=L,factors=(1.,1.))
    L,N = L
    p = factors .* p
    ρ = p[1]
    α = p[2]
    β = p[3:(N+2)]
    d = p[(N+3):(2*N+2)]
    γ = p[2*N+3]
    λ = p[2*N+4]

    x = u[1:N]
    y = u[(N+1):(2*N)]
    du[1:N] .= -ρ*L*x .+ α .* x .* (λ .- β .- y .- x)   # quick gradient computation
    du[(N+1):(2*N)] .=  γ .* (d .- β .- y) .* x  
end
function brennaniii(du,u,p,t;L=L,factors=(1.,1.))
    L,N = L
    p = factors .* p
    ρ = p[1]
    α = p[2]
    β = p[3:(N+2)]
    d = p[(N+3):(2*N+2)]
    γ = p[2*N+3]
    λ = p[2*N+4]

    x = u[1:N]
    y = u[(N+1):(2*N)]
    du[1:N] .= -ρ*L*x .+ α .* x .* (β .- d .- y .- x)   # quick gradient computation
    du[(N+1):(2*N)] .=  γ .* (λ .- d .- y) .* x  
end
function death_simplifiedii_nodecay(du,u,p,t;L=L,factors=(1.,1.))
    L,N = L
    p = factors .* p
    ρ = p[1]
    α = p[2]
    β = p[3:(N+2)]
    d = p[(N+3):(2*N+2)]

    x = u[1:N]
    y = u[(N+1):(2*N)]
    du[1:N] .= -ρ*L*x .+ α .* x .* (β .* (1 .- y) .- x)   # quick gradient computation
    du[(N+1):(2*N)] .=  d .* (1 .- y) .* x  
end
function death_simplifiedii_time(du,u,p,t;L=L,factors=(1.,1.))
    L,N = L
    p = factors .* p
    ρ = p[1]
    α = p[2]
    β = p[3:(N+2)]
    d = p[(N+3):(2*N+2)]
    γ = p[end]

    x = u[1:N]
    y = u[(N+1):(2*N)]
    du[1:N] .= -ρ*L*x .+ α .* x .* (β .+ y .- x)   # quick gradient computation
    du[(N+1):(2*N)] .=  d .* x .* (γ .- y) .* (y .- γ) 
end
function death_simplifiedii_bilateral(du,u,p,t;L=L,factors=(1.,1.),M=222)
    L,N = L
    p = factors .* p
    n = N - 2*M
    ρ = p[1]
    α = p[2]
    β = repeat(p[3:(M+2)],2)  # regions with bilateral twin
    β = vcat(β, p[(M+3):(M+n+2)])  # regions without bilateral twin
    d = repeat(p[(M+n+3):(2*M+n+2)],2)  # regions with bilateral twin
    d = vcat(d, p[(2*M+n+3):(2*M+2*n+2)])  # regions without bilateral twin
    γ = p[2*M+2*n+3]

    x = u[1:N]
    y = u[(N+1):(2*N)]
    du[1:N] .= -ρ*L*x .+ α .* x .* (β .- β .* y .- d .* y .- x)   # quick gradient computation
    du[(N+1):(2*N)] .=  γ .* (1 .- y)  
end
function death_simplifiedii_bilateral2(du,u,p,t;L=L,factors=(1.,1.))
    La,Lr,N = L
    M = Int(N/2)
    p = factors .* p
    ρa = p[1]
    ρr = p[2]
    α = p[3]
    β = repeat(p[4:(M+3)],2)
    d = repeat(p[(M+4):(2*M+3)],2)
    γ = p[2*M+4]

    x = u[1:N]
    y = u[(N+1):(2*N)]
    du[1:N] .= -ρa*La*x - ρr*Lr*x .+ α .* x .* (β .- β .* y .- d .* y .- x)   # quick gradient computation
    du[(N+1):(2*N)] .=  γ .* (1 .- y)  
end
function death_simplifiediii(du,u,p,t;L=L,factors=(1.,1.))
    L,N = L
    p = factors .* p
    ρ = p[1]
    α = p[2]
    β = p[3:(N+2)]
    γ = p[(N+3):end]

    x = u[1:N]
    y = u[(N+1):(2*N)]
    du[1:N] .= -ρ*L*x .+ α .* x .* (β .- β .* y .- x)   # quick gradient computation
    du[(N+1):(2*N)] .=  γ .* (1 .- y)  
end
function death_sis(du,u,p,t;L=L,factors=(1.,1.))
    L,N = L
    p = factors .* p
    ρ = p[1]
    α = p[2]
    β = p[3:(N+2)]
    d = p[(N+3):(2*N+2)]

    x = u[1:N]
    y = u[(N+1):(2*N)]
    du[1:N] .= -ρ*L*x .+ α  .* x .* (β .- y .- x)   # quick gradient computation
    du[(N+1):(2*N)] .=  d .* x  
end
function sis(du,u,p,t;L=W,factors=(1.,1.))
    W,N = L
    p = factors .* p
    #τ = p[1:N]
    #γ = p[(N+1):2*N]
    #ϵ = p[end]
    τ = p[1]
    γ = p[2:(N+1)]
    ϵ = p[end]


    x = u[1:N]
    du[1:N] .= ϵ*W*x .* (100 .- x) .+ τ .* x .* (100 .- γ ./ τ .- x)    
    #du[1:N] .= τ .* x .* (1 .- x) .- γ .* x   
end
function sir(du,u,p,t;L=W,factors=(1.,1.))
    W,N = L
    p = factors .* p
    ϵ = p[1]
    τ = p[2]
    γ = p[3:(N+2)]
    θ = p[N+3:(2*N+2)]


    x = u[1:N]
    y = u[(N+1):(2*N)]
    du[1:N] .= ϵ*W*x .* (100 .- y .- x) .+ τ .* x .* (100 .- y .- x) .- (γ .+ θ) .* x   
    du[(N+1):(2*N)] .=  θ .* x  
end
function death2(du,u,p,t;L=L,factors=(1.,1.))
    La, Lr, N = L  
    p = factors .* p
    ρa = p[1]
    ρr = p[2]
    α = p[3]
    β = p[4:(N+3)]
    d = p[(N+4):(2*N+3)]
    γ = p[end]


    x = u[1:N]
    y = u[(N+1):(2*N)]
    du[1:N] .= -ρa*ρr*La*x .- ρa*Lr*x .+ α  .* x .* (β .- d .* y .- x)   # quick gradient computation
    du[(N+1):(2*N)] .=  γ .* (1 .- y)  
end
function death_all_local2(du,u,p,t;L=L,factors=(1.,1.))
    La, Lr, N = L  
    p = factors .* p
    ρa = p[1]
    ρr = p[2]
    α = p[3:(N+2)]
    β = p[(N+3):(2*N+2)]
    d = p[(2*N+3):(3*N+2)]
    γ = p[(3*N+3):end]


    x = u[1:N]
    y = u[(N+1):(2*N)]
    du[1:N] .= -ρa*ρr*La*x .- ρa*Lr*x .+ α  .* x .* (β .- y .- x)   # quick gradient computation
    #du[(N+1):(2*N)] .=  γ .* (1 .- y)  
    du[(N+1):(2*N)] .=  γ .* (d .* x .- y)  
end
function death_superlocal2(du,u,p,t;L=L,factors=(1.,1.))
    La, Lr, N = L  
    p = factors .* p
    ρa = p[1]
    ρr = p[2]
    α = p[3]
    β = p[4:(N+3)]
    d = p[(N+4):(2*N+3)]
    γ = p[end]


    x = u[1:N]
    y = u[(N+1):(2*N)]
    #du[1:N] .= -ρa*ρr*La*x .- ρa*Lr*x .+ α  .* x .* (β .- d.*y .- x)   # quick gradient computation
    du[1:N] .= -ρa*La*x .- ρr*Lr*x .+ α  .* x .* (β .- d.*y .- x)   # quick gradient computation
    du[(N+1):(2*N)] .=  γ .* (1 .- y)  
    #du[(N+1):(2*N)] .=  γ .* (x .- y)  
end
function death_simplifiedii_clustered(du, u, p, t; L=L, partition_sizes=[1, 1], factors=(1., 1.))
    L, N_total = L  # Total length of the system
    K = length(partition_sizes)  # Number of clusters (based on partition_sizes)
    partition_sizes = Array{Int}(partition_sizes)  # Convert to array if necessary
    
    p = factors .* p  # Adjust parameters using the factors
    
    # Extract parameters for each cluster
    ρ = p[1]
    α = p[2:2+K-1]
    β = p[2+K:2+2*K-1]
    d = p[2+2*K:2+3*K-1]
    γ = p[2+3*K:end]

    # Initialize variables
    x = u[1:N_total]  # First N elements are x values
    y = u[(N_total+1):(2*N_total)]  # Second N elements are y values

    # Keep track of the current index offset
    start_idx = 1

    # Loop through each cluster and compute the gradients
    for k in 1:K
        # Determine the number of elements in this cluster (based on partition_sizes)
        n_cluster = partition_sizes[k]
        end_idx = start_idx + n_cluster - 1

        # Apply the ODE dynamics for the current cluster
        du[start_idx:end_idx] .= -ρ * L * x[start_idx:end_idx] .+ α[k] .* x[start_idx:end_idx] .* (β[start_idx:end_idx] .- β[start_idx:end_idx] .* y[start_idx:end_idx] .- d[start_idx:end_idx] .* y[start_idx:end_idx] .- x[start_idx:end_idx])  
        du[(N_total+start_idx):(N_total+end_idx)] .= γ[k] .* (1 .- y[start_idx:end_idx])  # y dynamics
        
        # Update the start index for the next cluster
        start_idx = end_idx + 1
    end
end
function DIFF(du,u,p,t;L=L,factors=nothing)
    L, _ = L
    ρ = p[1]

    du .= -ρ*L*u 
end
function DIFFG(du,u,p,t;L=L,factors=(1.,1.))
    L, _ = L
    kα,kβ = factors 
    ρ = p[1]
    α = kα * p[2]
    β = kβ .* p[3:end]

    du .= -ρ*L*u .+ α .* u .* (β .- u)  
end
function DIFFGA(du,u,p,t;L=L,factors=(1.,1.))
    L,N = L
    p = factors .* p
    ρ = p[1]
    α = p[2]
    β = p[3:(N+2)]
    d = p[(N+3):(2*N+2)]

    x = u[1:N]
    y = u[(N+1):(2*N)]
    du[1:N] .= -ρ*L*x .+ α .* x .* (β .- y .- x)   # quick gradient computation
    du[(N+1):(2*N)] .=  d .* x  
end
function DIFFGAM(du,u,p,t;L=L,factors=(1.,1.))
    L,N = L
    p = factors .* p
    ρ = p[1]
    α = p[2]
    β = p[3:(N+2)]
    d = p[(N+3):(2*N+2)]
    γ = p[2*N+3]
    λ = p[2*N+4]

    x = u[1:N]
    y = u[(N+1):(2*N)]
    du[1:N] .= -ρ*L*x .+ α .* x .* (λ .- β .- y .- x)   # quick gradient computation
    du[(N+1):(2*N)] .=  γ .* (d .- β .- y) .* x  
end
function DIFF_BI(du,u,p,t;L=L,factors=nothing)
    Lr,La,_ = L
    ρr = p[1]
    ρa = p[2]

    du .= -ρr*Lr*u .- ρa*La*u 
end
function DIFFG_BI(du,u,p,t;L=L,factors=(1.,1.))
    Lr, La, _ = L
    kα,kβ = factors 
    ρr = p[1]
    ρa = p[2]
    α = kα * p[3]
    β = kβ .* p[4:end]

    du .= -ρr*Lr*u .- ρa*La*u .+ α .* u .* (β .- u)  
end
function DIFFGA_BI(du,u,p,t;L=L,factors=(1.,1.))
    Lr,La,N = L
    p = factors .* p
    ρr = p[1]
    ρa = p[2]
    α = p[3]
    β = p[4:(N+3)]
    d = p[(N+4):(2*N+3)]

    x = u[1:N]
    y = u[(N+1):(2*N)]
    du[1:N] .= -ρr*Lr*x .- ρa*La*x .+ α .* x .* (β .- y .- x)   # quick gradient computation
    du[(N+1):(2*N)] .=  d .* x  
end
function DIFFGAM_BI(du,u,p,t;L=L,factors=(1.,1.))
    Lr,La,N = L
    p = factors .* p
    ρr = p[1]
    ρa = p[2]
    α = p[3]
    β = p[4:(N+3)]
    d = p[(N+4):(2*N+3)]
    γ = p[2*N+4]
    λ = p[2*N+5]

    x = u[1:N]
    y = u[(N+1):(2*N)]
    du[1:N] .= -ρr*Lr*x .- ρa*La*x .+ α .* x .* (λ .- β .- y .- x)   # quick gradient computation
    du[(N+1):(2*N)] .=  γ .* (d .- β .- y) .* x  
end
function DIFFGAM(du,u,p,t;L=L,factors=(1.,1.))
    L,N = L
    p = factors .* p
    ρ = p[1]
    α = p[2]
    β = p[3:(N+2)]
    d = p[(N+3):(2*N+2)]
    γ = p[2*N+3]
    λ = p[2*N+4]

    x = u[1:N]
    y = u[(N+1):(2*N)]
    du[1:N] .= -ρ*L*x .+ α .* x .* (λ .- β .- y .- x)   # quick gradient computation
    du[(N+1):(2*N)] .=  γ .* (d .- β .- y) .* x  
end



end # module ODEs
