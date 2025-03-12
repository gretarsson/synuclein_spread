using Serialization
include("helpers.jl")
using PrettyTables

# Read inference results
simulations = [
    "total_diffusion_N=448_threads=4_var1_normalpriors",
    "total_aggregation_N=448_threads=4_var1_normalpriors",
    "total_death_simplifiedii_N=448_threads=4_var1_normalpriors"
]
model_names = ["diffusion-only", "diffusion+aggregation", "diffusion+aggregation+decay"]
inferences = []
for simulation in simulations
    push!(inferences, deserialize("simulations/" * simulation * ".jls"))
end

# Compute WAIC, AIC, and BIC for models
waic_vals = Float64[]
aic_vals  = Float64[]
bic_vals  = Float64[]
for inference in inferences
    waic = compute_waic(inference; S=1000)
    push!(waic_vals, waic)
    aic, bic = compute_aic_bic(inference)
    push!(aic_vals, aic)
    push!(bic_vals, bic)
end

# Compute delta metrics relative to best (lowest) value
min_waic = minimum(waic_vals)
min_aic  = minimum(aic_vals)
min_bic  = minimum(bic_vals)

delta_waic = [w - min_waic for w in waic_vals]
delta_aic  = [a - min_aic for a in aic_vals]
delta_bic  = [b - min_bic for b in bic_vals]

# Build a DataFrame to display the results nicely
df = DataFrame(
    Model      = model_names,
    WAIC       = round.(waic_vals, digits=0),
    ΔWAIC    = round.(delta_waic, digits=0),
    AIC        = round.(aic_vals, digits=0),
    ΔAIC     = round.(delta_aic, digits=0),
    BIC        = round.(bic_vals, digits=0),
    ΔBIC     = round.(delta_bic, digits=0)
)

# Use PrettyTables to produce a LaTeX table
pretty_table(df; formatters = ft_printf("%5d"), backend = Val(:latex))