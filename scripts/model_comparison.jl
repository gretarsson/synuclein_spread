using Serialization
include("helpers.jl");


# read inference results
simulation_diffusion = "total_diffusion_N=448_threads=4_var1_normalpriors";
simulation_aggregation = "total_aggregation_N=448_threads=4_var1_normalpriors";
simulation_decay = "total_death_N=448_threads=4_var1_normalpriors";
simulations = [simulation_diffusion, simulation_aggregation, simulation_decay];
#simulations = [simulation_aggregation];
inferences = [];
for simulation in simulations
    push!(inferences,deserialize("simulations/"*simulation*".jls"))
end


# Compute WAIC, AIC, and BIC for models
println("WAIC:")
for (i,inference) in enumerate(inferences)
    waic = compute_waic(inference; S=1000);
    println("$(inferences[i]["ode"]): ", waic)
end
println("AIC:")
for (i,inference) in enumerate(inferences)
    aic,bic = compute_aic_bic(inference);
    println("$(inferences[i]["ode"]): ", aic)
end
println("BIC:")
for (i,inference) in enumerate(inferences)
    aic,bic = compute_aic_bic(inference);
    println("$(inferences[i]["ode"]): ", bic)
end
