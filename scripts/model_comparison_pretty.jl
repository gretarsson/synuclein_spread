using Serialization
include("helpers.jl")
using PrettyTables, DataFrames

# Read inference results
simulations = [
    "simulations/DIFF_EUCL",
    "simulations/DIFF_ANTERO",
    "simulations/DIFF_RETRO",
    "simulations/DIFF_BIDIR",
    #
    #"simulations/DIFFG_ANTERO",
    #"simulations/DIFFG_RETRO",
    #"simulations/DIFFG_BIDIR",
    ##
    #"simulations/DIFFGA_ANTERO",
    #"simulations/DIFFGA_RETRO",
    #"simulations/DIFFGA_BIDIR",
    ##
    #"simulations/DIFFGAM_ANTERO",
    #"simulations/DIFFGAM_RETRO",
    #"simulations/DIFFGAM_BIDIR",
]
model_names = [
    "DIFF euclidean", 
    "DIFF anterograde", 
    "DIFF retrograde", 
    "DIFF bidirectional", 
    #
    #"DIFFG anterograde", 
    #"DIFFG retrograde", 
    #"DIFFG bidirectional", 
    ##
    #"DIFFGA anterograde", 
    #"DIFFGA retrograde", 
    #"DIFFGA bidirectional", 
    ##
    #"DIFFGAM anterograde", 
    #"DIFFGAM retrograde", 
    #"DIFFGAM bidirectional", 
    #
]
inferences = []
for simulation in simulations
    push!(inferences, deserialize(simulation * ".jl"))
end

# Compute WAIC, AIC, BIC, MSE, and Frobenius covariance norm for models
waic_vals = Float64[]
aic_vals  = Float64[]
bic_vals  = Float64[]
mse_vals  = Float64[]
covnorm_vals = Float64[]
regcov = []

for inference in inferences
    waic, _ = compute_waic_wbic(inference; S=10)
    push!(waic_vals, waic)
    aic, bic = compute_aic_bic(inference)
    push!(aic_vals, aic)
    push!(bic_vals, bic)
    mse = compute_mse_mc(inference)
    push!(mse_vals, mse)
    regional_cov = compute_regional_correlations(inference)
    #covnorm = mean(abs.(regional_cov))  # avg |r|
    covnorm = mean((regional_cov).^2)  # avg R^2
    push!(covnorm_vals, covnorm)
    push!(regcov, regional_cov)
end

# Compute delta metrics relative to the best (lowest) value
min_waic = minimum(waic_vals)
min_aic  = minimum(aic_vals)
min_bic  = minimum(bic_vals)
min_mse  = minimum(mse_vals)
#min_cov  = minimum(filter(!isnan, abs.(covnorm_vals)))

# Handle the case where all covnorm_vals are NaN
valid_cov = filter(!isnan, covnorm_vals)  # values are already ≥0; no need for abs here
if isempty(valid_cov)
    min_cov   = NaN
    delta_cov = fill(NaN, length(covnorm_vals))
else
    min_cov   = minimum(valid_cov)
    delta_cov = [c - min_cov for c in covnorm_vals]  # NaN stays NaN here automatically
end

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
    ParCor = round.(covnorm_vals, digits=4),
    ∆ParCor   = round.(delta_cov, digits=4)
)

# Print LaTeX table
pretty_table(df; formatters = ft_printf("%5d"), backend = Val(:latex))

# ─── reorder DataFrame so deltas come last ─────────────────────────────────────
ordered = [
  :Model,
  :WAIC, :AIC, :BIC,      # main big metrics
  :MSE,  :ParCor,         # main small metrics
  :∆WAIC, :∆AIC, :∆BIC,   # deltas for the big metrics
  :∆MSE,  :∆ParCor        # deltas for the small metrics
]
df2 = df[:, ordered]

# ─── print LaTeX table with mixed formatting ───────────────────────────────────
pretty_table(
  df2;
  formatters = (
    ft_printf("%s",        1),     # Model (string)
    ft_printf("%7.0f",   2:4),     # WAIC, AIC, BIC  (zero‑decimal floats)
    ft_printf("%.2e",    5:6),     # MSE, ParCor     (sci‑notation 6 d.p.)
    ft_printf("%7.0f",   7:9),     # ∆WAIC, ∆AIC, ∆BIC (zero‑decimal floats)
    ft_printf("%.2e",     10),     # ∆MSE           (sci‑notation 6 d.p.)
    ft_printf("%.2e",     11)      # ∆ParCor        (sci‑notation 4 d.p.)
  ),
  backend = Val(:latex),
)


