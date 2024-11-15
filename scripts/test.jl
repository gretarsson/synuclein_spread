using Turing

# Simple model
@model function simple_model(p)
    s ~ Normal(0,1)
    p ~ Normal(s, 5)
end

# Callback function
function my_callback(rng, model, sampler, sample, iteration)
    println("Iteration: $iteration, p: $(sample[:p])")
end

# Sample with callback
chain = sample(simple_model(1.), NUTS(), 1000; callback=my_callback)
