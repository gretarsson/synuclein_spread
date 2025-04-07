using Serialization
include("helpers.jl")
using PrettyTables, DataFrames

# Read inference results
simulations = [
    "total_diffusion_N=448_threads=4_var1_normalpriors",
    "total_aggregation_N=448_threads=4_var1_normalpriors",
    "total_death_simplifiedii_N=448_threads=1_var1_olddecay_withx",
    "total_death_simplifiedii_bilateral_N=448_threads=1_var1_NEW"
]
model_names = ["diffusion-only", "diffusion+aggregation", "diffusion+aggregation+decay", "bilateral"]
inferences = []
for simulation in simulations
    push!(inferences, deserialize("simulations/" * simulation * ".jls"))
end

# Compute WAIC, AIC, BIC, and MSE for models
waic_vals = Float64[]
aic_vals  = Float64[]
bic_vals  = Float64[]
mse_vals  = Float64[]
for inference in inferences
    waic, _ = compute_waic_wbic(inference; S=2)  # WBIC is computed but not used in the table.
    push!(waic_vals, waic)
    aic, bic = compute_aic_bic(inference)
    push!(aic_vals, aic)
    push!(bic_vals, bic)
    mse = compute_mse_mc(inference)
    push!(mse_vals, mse)
end

# Compute delta metrics relative to the best (lowest) value
min_waic = minimum(waic_vals)
min_aic  = minimum(aic_vals)
min_bic  = minimum(bic_vals)
min_mse  = minimum(mse_vals)
mse_vals

delta_waic = [w - min_waic for w in waic_vals]
delta_aic  = [a - min_aic for a in aic_vals]
delta_bic  = [b - min_bic for b in bic_vals]
delta_mse  = [m - min_mse for m in mse_vals]

# Build a DataFrame to display the results
df = DataFrame(
    Model   = model_names,
    WAIC    = round.(waic_vals, digits=0),
    ΔWAIC   = round.(delta_waic, digits=0),
    AIC     = round.(aic_vals, digits=0),
    ΔAIC    = round.(delta_aic, digits=0),
    BIC     = round.(bic_vals, digits=0),
    ΔBIC    = round.(delta_bic, digits=0),
    MSE     = round.(mse_vals, digits=4),
    ΔMSE    = round.(delta_mse, digits=4)
)

ormatters = (
    ft_printf("%s"),
    ft_printf("%5d"),
    ft_printf("%5d"),
    ft_printf("%5d"),
    ft_printf("%5d"),
    ft_printf("%5d"),
    ft_printf("%5d"),
    ft_printf("%7.2f"),
    ft_printf("%7.2f")
)



# Produce a LaTeX table using PrettyTables
#pretty_table(df; formatters=formatters, backend=Val(:latex))
# Use PrettyTables to produce a LaTeX table
pretty_table(df; formatters = ft_printf("%5d"), backend = Val(:latex))