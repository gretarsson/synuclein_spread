# create custom distribution
using Distributions
using Random
using StatsPlots


struct TanhNormal{T<:Real} <: Distribution{Univariate,Continuous}
    μ::T
    σ::T

    # inner contrsuctor function
    function TanhNormal{T}(μ::T, σ::T; check_args=true) where {T<:Real}
        check_args && Distributions.@check_args(TanhNormal, σ > 0)
        return new{T}(μ,σ)
    end
end

# outer constructors
TanhNormal(μ::T, σ::T) where {T<:Real} = TanhNormal{T}(μ,σ)
function TanhNormal(μ::Float64,σ::Float64; check_args=true)
    return TanhNormal{Float64}(μ,σ; check_args=check_args)
end

TanhNormal(μ::Real, σ::Real) = TanhNormal(promote(μ,σ)...)
TanhNormal(μ::Int, σ::Int) = TanhNormal(float(μ), float(σ))

import Base.rand, StatsBase.params
import Random, Distributions, Statistics, StatsBase
using SpecialFunctions
params(d::TanhNormal) = (d.μ, d.σ)

# rand
function Base.rand(rng::AbstractRNG, d::TanhNormal)
    (μ, σ) = params(d)
    x = Base.rand(truncated(Normal(μ,σ),lower=0))
    return tanh(x)
end

# sampler
Distributions.sampler(rng::AbstractRNG,d::TanhNormal) = Base.rand(rng::AbstractRNG, d::TanhNormal)

# log pdf
normal_cdf(x,μ,σ) = 0.5 * (1 + erf((x-μ)/(sqrt(2)*σ)))  # need to normalize truncated normal
function Distributions.pdf(d::TanhNormal{T}, x::Real) where {T<:Real}
    (μ, σ) = params(d)
    if x < 0
        return zero(T)
    elseif x>= 1
        return zero(T)
    else
        #return 1/sqrt(2*pi*σ^2) * exp(-(atanh(x)-μ)^2/(2*σ^2)) * 1/(1-x^2)
        #return pdf(truncated(Normal(μ,σ),lower=0),atanh(x)) * 1/(1-x^2)
        return 1/sqrt(2*pi*σ^2) * exp(-(atanh(x)-μ)^2/(2*σ^2)) * 1/(1-x^2) / (1-normal_cdf(atanh(0),μ,σ))
        
    end
end
Distributions.logpdf(d::TanhNormal, x::Real) = log(pdf(d,x))



# cdf (doing it numerically for now)
#function Distributions.cdf(d::TanhNormal{T}, x::Real) where T<:Real
#    (μ,σ) = params(d)
#    if x <= 0
#        return zero(T)
#    elseif x >= 1
#        return one(T)
#    else
#        f(xx) = pdf(TanhNormal(μ,σ),xx)
#        return quadgk(f,0,x)[1]
#    end
#end
#
## quantile
#function Statistics.quantile(d::TanhNormal{T}, x::Real) where T<:Real
#    (μ, σ) = params(TanhNormal)
#    if x <= 0
#        return zero(T)
#    elseif x >= 1
#        return one(T)
#    else
#        eq = x + pdf(TanhNormal(μ,σ),0) - pdf
#        nlsolve = pdf(TanhNormal(μ,σ),xx)
#        return quadgk(f,0,x)[1]
#    end
#end
#
## minimum
#function Base.minimum(d::TanhNormal)
#    return(0)
#end
#
## maximum
#function Base.maximum(d::TanhNormal)
#    return(1)
#end
#
## insupport
#function Distributions.insupport(d::TanhNormal)
#    insupport(d::TanhNormal, x::Real) = zero(x) <= x <= one(x)
#end

##using QuadGK
#N = 1e5
#d = TanhNormal(0,0.5)
#heyhey = [rand(d) for _ in 1:N]
#x = [1:N...] ./ N
#y = [pdf(d,xx) for xx in x]
#ly = [logpdf(d,xx) for xx in x]
#cdf(d,1) - cdf(d,0)
#
##histogram(density(heyhey))
#plot(x,y;)
