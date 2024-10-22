using Turing
using ReverseDiff



@model function gdemo(x, y)
    # priors
    σ ~ InverseGamma(2, 3)
    m ~ truncated(Normal(0,σ^2),lower=0, upper=Inf)
    # generative model
    #x ~ truncated(Normal(m, σ^2),0,1)
    #y ~ truncated(Normal(m, σ^2), 0,1)
    x ~ Normal(m, σ^2)
    y ~ Normal(m, σ^2)
end


chain = sample(gdemo(0.2, 0.6), NUTS(;adtype=AutoReverseDiff()), 1000, progress=false)



# --------------