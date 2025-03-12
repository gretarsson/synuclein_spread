using Serialization
include("helpers.jl");


# read inference results
simulations = ["total_death_simplifiedii_bilateral_N=444_threads=1_var1",
               "total_death_simplifiedii_bilateral_antero_N=444_threads=1_var1",
               "total_death_simplifiedii_bilateral2_retroantero_N=444_threads=1_var1"]
model_names = ["retrograde", "anterograde", "bidirectional"]
inferences = [];
for simulation in simulations
    push!(inferences,deserialize("simulations/"*simulation*".jls"))
end


# Compute WAIC, AIC, and BIC for models
println("WAIC:")
for (i,inference) in enumerate(inferences)
    waic = compute_waic(inference; S=1000);
    println("$(model_names[i]): ", waic)
end
println("AIC:")
for (i,inference) in enumerate(inferences)
    aic,bic = compute_aic_bic(inference);
    println("$(model_names[i]): ", aic)
end
println("BIC:")
for (i,inference) in enumerate(inferences)
    aic,bic = compute_aic_bic(inference);
    println("$(model_names[i]): ", bic)
end
