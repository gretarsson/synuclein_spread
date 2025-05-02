# odes.jl

module ODEs

export DIFFGAM, DIFFGAM_bilateral

using LinearAlgebra
using Statistics

#### Plain (per‑region) ODEs ####

# Dictionary containing the ODEs

odes = Dict("diffusion" => diffusion, "diffusion2" => diffusion2, "diffusion3" => diffusion3, "diffusion_pop2" => diffusion_pop2, "aggregation" => aggregation, 
            "aggregation2" => aggregation2, "aggregation_pop2" => aggregation_pop2, "death_local2" => death_local2, "aggregation2_localα" => aggregation2_localα,
            "death_superlocal2" => death_superlocal2, "death2" => death2, "death_all_local2" => death_all_local2, "death" => death, "sir" => sir, "sis" => sis, 
            "DIFF" => DIFF,
            "DIFFG" => DIFFG,
            "DIFFGA" => DIFFGA,
            "DIFFGAM" => DIFFGAM,
            "DIFF_BI" => DIFF_BI,
            "DIFFG_BI" => DIFFG_BI,
            "DIFFGA_BI" => DIFFGA_BI,
            "DIFFGAM_BI" => DIFFGAM_BI,
            "fastslow" => fastslow,
            "fastslow_reparam" => fastslow_reparam,
            "fastslow_reparamii" => fastslow_reparamii,
            "fastslow_regionaltime" => fastslow_regionaltime,
            "heterodimer_inspired" => heterodimer_inspired,
            "brennan" => brennan,
            "brennanii" => brennanii,
            "brennaniii" => brennaniii,
            "death_simplified" => death_simplified,
            "death_simplifiedii" => death_simplifiedii,
            "death_simplifiedii_regionaltime" => death_simplifiedii_regionaltime,
            "death_simplifiedii_uncor" => death_simplifiedii_uncor,
            "death_simplifiedii_time" => death_simplifiedii_time,
            "death_simplifiedii_nodecay" => death_simplifiedii_nodecay,
            "death_simplifiedii_clustered" => death_simplifiedii_clustered,
            "death_simplifiedii_bilateral" => death_simplifiedii_bilateral,
            "death_simplifiedii_bilateral2" => death_simplifiedii_bilateral2,
            "death_simplifiediii" => death_simplifiediii)

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

    # split the state vector
    x = @view u[1    :  N]
    y = @view u[N+1  : 2*N]

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

    # split the state vector
    x = @view u[1    :  N]
    y = @view u[N+1  : 2*N]

    du[1:N] .= -ρ*L*x .+ α .* x .* (λ .- β .- y .- x)   # quick gradient computation
    du[(N+1):(2*N)] .=  γ .* (d .- β .- y) .* x  
end
function DIFF_bidirectional(du,u,p,t;L=L,factors=nothing)
    Lr,La,_ = L
    ρr = p[1]
    ρa = p[2]

    du .= -ρr*Lr*u .- ρa*La*u 
end
function DIFFG_bidirectional(du,u,p,t;L=L,factors=(1.,1.))
    Lr, La, _ = L
    kα,kβ = factors 
    ρr = p[1]
    ρa = p[2]
    α = kα * p[3]
    β = kβ .* p[4:end]

    du .= -ρr*Lr*u .- ρa*La*u .+ α .* u .* (β .- u)  
end
function DIFFGA_bidirectional(du,u,p,t;L=L,factors=(1.,1.))
    Lr,La,N = L
    p = factors .* p
    ρr = p[1]
    ρa = p[2]
    α = p[3]
    β = p[4:(N+3)]
    d = p[(N+4):(2*N+3)]

    # split the state vector
    x = @view u[1    :  N]
    y = @view u[N+1  : 2*N]
    du[1:N] .= -ρr*Lr*x .- ρa*La*x .+ α .* x .* (β .- y .- x)   # quick gradient computation
    du[(N+1):(2*N)] .=  d .* x  
end
function DIFFGAM_bidirectional(du,u,p,t;L=L,factors=(1.,1.))
    Lr,La,N = L
    p = factors .* p
    ρr = p[1]
    ρa = p[2]
    α = p[3]
    β = p[4:(N+3)]
    d = p[(N+4):(2*N+3)]
    γ = p[2*N+4]
    λ = p[2*N+5]

    # split the state vector
    x = @view u[1    :  N]
    y = @view u[N+1  : 2*N]

    du[1:N] .= -ρr*Lr*x .- ρa*La*x .+ α .* x .* (λ .- β .- y .- x)   # quick gradient computation
    du[(N+1):(2*N)] .=  γ .* (d .- β .- y) .* x  
end
function DIFFGAM_bidirectional(du,u,p,t;L=L,factors=(1.,1.))
    L,N = L
    p = factors .* p
    ρ = p[1]
    α = p[2]
    β = p[3:(N+2)]
    d = p[(N+3):(2*N+2)]
    γ = p[2*N+3]
    λ = p[2*N+4]

    # split the state vector
    x = @view u[1    :  N]
    y = @view u[N+1  : 2*N]

    du[1:N] .= -ρ*L*x .+ α .* x .* (λ .- β .- y .- x)   # quick gradient computation
    du[(N+1):(2*N)] .=  γ .* (d .- β .- y) .* x  
end
# BILATERAL
function DIFFGAM_bilateral(du, u, p, t; L, factors = (1.,1.), region_group::Vector{Int})
    # unpack Laplacian & sizes
    Lmat, N = L
    M       = maximum(region_group)    # # of hemisphere‑groups

    # apply global scaling
    p = factors .* p

    # globals: ρ, α, then later γ, λ
    ρ, α = p[1], p[2]
    γ, λ = p[end-1], p[end]

    # group‑level β’s and d’s
    βg = @view p[3        : 2 + M]    # p[3] through p[2+M]
    dg = @view p[3 + M    : 2*M + 2]  # p[3+M] through p[2M+2]

    # broadcast each group value into its N regions
    β = βg[region_group]
    d =  dg[region_group]

    # split the state vector
    x = @view u[1    :  N]
    y = @view u[N+1  : 2*N]

    # ODEs
    du[1:N]     .= -ρ*Lmat*x .+ α .* x .* (λ .- β .- y .- x)
    du[N+1:2*N] .=  γ .* (d .- β .- y) .* x

    return nothing
end


end # module ODEs
