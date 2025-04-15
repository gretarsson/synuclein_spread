using Serialization
include("helpers.jl")
using PrettyTables, DataFrames

# Read inference results
simulations = [
    "total_diffusion_N=448_threads=4_var1_normalpriors",
    "total_aggregation_N=448_threads=4_var1_normalpriors",
    "total_heterodimer_inspired_N=448_threads=1_var1",
    "total_death_simplifiedii_N=448_threads=1_var1_olddecay_withx_notrunc",
    "total_death_simplifiedii_uncor_N=448_threads=1_var1_olddecay_withx_notrunc",
    "total_brennanii_N=448_threads=1_var1"
]
model_names = ["diffusion-only", "diffusion+aggregation", "DAD (heterodimer)", "DAD (no trunc.)", "DAD (no trunc. uncor.)", "DAD (brennanii)"]
inferences = []
for simulation in simulations
    push!(inferences, deserialize("simulations/" * simulation * ".jls"))
end

# Compute WAIC, AIC, BIC, MSE, and Frobenius covariance norm for models
waic_vals = Float64[]
aic_vals  = Float64[]
bic_vals  = Float64[]
mse_vals  = Float64[]
covnorm_vals = Float64[]

for inference in inferences
    waic, _ = compute_waic_wbic(inference; S=1000)
    push!(waic_vals, waic)
    aic, bic = compute_aic_bic(inference)
    push!(aic_vals, aic)
    push!(bic_vals, bic)
    mse = compute_mse_mc(inference)
    push!(mse_vals, mse)
    covnorm = mean(compute_regional_covariances(inference))
    push!(covnorm_vals, covnorm)
end

# Compute delta metrics relative to the best (lowest) value
min_waic = minimum(waic_vals)
min_aic  = minimum(aic_vals)
min_bic  = minimum(bic_vals)
min_mse  = minimum(mse_vals)
min_cov  = minimum(filter(!isnan, abs.(covnorm_vals)))

delta_waic = [w - min_waic for w in waic_vals]
delta_aic  = [a - min_aic for a in aic_vals]
delta_bic  = [b - min_bic for b in bic_vals]
delta_mse  = [m - min_mse for m in mse_vals]
delta_cov  = [c - min_cov for c in covnorm_vals]

# Build a DataFrame to display the results
df = DataFrame(
    Model   = model_names,
    WAIC    = round.(waic_vals, digits=0),
    ∆WAIC   = round.(delta_waic, digits=0),
    AIC     = round.(aic_vals, digits=0),
    ∆AIC    = round.(delta_aic, digits=0),
    BIC     = round.(bic_vals, digits=0),
    ∆BIC    = round.(delta_bic, digits=0),
    MSE     = round.(mse_vals, digits=6),
    ∆MSE    = round.(delta_mse, digits=6),
    Cov = round.(covnorm_vals, digits=4),
    ∆Cov   = round.(delta_cov, digits=4)
)

# Print LaTeX table
pretty_table(df; formatters = ft_printf("%5d"), backend = Val(:latex))
