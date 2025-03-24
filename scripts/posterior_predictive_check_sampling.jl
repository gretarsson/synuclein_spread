using Serialization
using Distributions
using Statistics
using DataFrames
using PrettyTables
using Plots

# -------------------------------
# 1. Load Inference and Data
# -------------------------------
simulation = "total_death_simplifiedii_bilateral_N=448_threads=1_var1_NEW"
inference_obj = deserialize("simulations/" * simulation * ".jls")
data = inference_obj["data"]  # dimensions: n_vars × n_time × n_samples_data

n_vars, n_time, n_samples_data = size(data)

# -------------------------------
# 2. Aggregate Posterior Samples
# -------------------------------
# Here we use a subset of the full chain (e.g., 100 samples) for speed.
n_chain_samples = size(inference_obj["chain"], 1)
n_samples_chain = min(1000, n_chain_samples)
chain_indices = collect(1:n_samples_chain)  # or use randperm(n_chain_samples)[1:n_samples_chain]

# We'll accumulate the predictive means and second moments.
aggregated_mean   = zeros(n_vars, n_time)
aggregated_sq_moment = zeros(n_vars, n_time)

# Loop over selected posterior samples
for s in chain_indices
    # Use the helper function to get the predictive distributions for this sample.
    # This returns a vector of Normal distributions.
    posterior_pred = posterior_pred_sample(inference_obj, s)
    
    # If the returned vector is flat (length n_vars*n_time), reshape it to (n_vars, n_time)
    post_pred_matrix = reshape(posterior_pred, n_vars, n_time)
    
    # For each variable and time, extract the predictive mean and variance.
    for i in 1:n_vars, t in 1:n_time
        # Each element is a Normal distribution.
        μ_pred = mean(post_pred_matrix[i, t])
        σ_pred = std(post_pred_matrix[i, t])
        aggregated_mean[i, t] += μ_pred
        # To accumulate the second moment we add (mean² + variance)
        aggregated_sq_moment[i, t] += μ_pred^2 + σ_pred^2
    end
end

# Average over the number of posterior samples
aggregated_mean   ./= n_samples_chain
aggregated_sq_moment ./= n_samples_chain

# Compute the aggregated variance (E[X^2] - (E[X])^2)
aggregated_variance = aggregated_sq_moment .- aggregated_mean.^2
aggregated_std = sqrt.(aggregated_variance)

# -------------------------------
# 3. Posterior Predictive Checks Using Aggregated Predictions
# -------------------------------

# 3.1 Residual Analysis: Compute residuals (observed - aggregated predictive mean)
residuals = similar(data, Float64)
for i in 1:n_vars, t in 1:n_time, k in 1:n_samples_data
    if ismissing(data[i, t, k])
        residuals[i, t, k] = NaN
    else
        residuals[i, t, k] = data[i, t, k] - aggregated_mean[i, t]
    end
end
all_res = [r for r in residuals if !isnan(r)]
rmse = sqrt(mean(all_res .^ 2))
mae  = mean(abs.(all_res))
println("Posterior Predictive Check using Posterior Samples:")
println("RMSE: ", rmse)
println("MAE: ", mae)

# 3.2 Posterior Predictive p-values:
# For each observed value, we simulate a replicate from the aggregated predictive distribution.
ppp_vals = Float64[]
for i in 1:n_vars, t in 1:n_time, k in 1:n_samples_data
    if !ismissing(data[i, t, k])
        y_obs = data[i, t, k]
        # Use the aggregated mean and std for the predictive distribution at (i, t)
        y_rep = rand(Normal(aggregated_mean[i, t], aggregated_std[i, t]))
        push!(ppp_vals, (y_rep > y_obs) ? 1.0 : 0.0)
    end
end
ppp = mean(ppp_vals)
println("Overall Posterior Predictive p-value: ", ppp)

# 3.3 Error Heatmap: Average absolute residual per variable/time point.
avg_abs_res = zeros(n_vars, n_time)
for i in 1:n_vars, t in 1:n_time
    cell_res = [abs(data[i, t, k] - aggregated_mean[i, t]) for k in 1:n_samples_data if !ismissing(data[i, t, k])]
    avg_abs_res[i, t] = isempty(cell_res) ? NaN : mean(cell_res)
end
Plots.heatmap(avg_abs_res, xlabel="Time Point", ylabel="Variable Index", 
    title="Error Heatmap (Aggregated Predictions)")

# 3.4 Calibration Plot: Compare the cumulative probabilities of the observed values.
calibration_probs = Float64[]
for i in 1:n_vars, t in 1:n_time, k in 1:n_samples_data
    if !ismissing(data[i, t, k])
        p_val = cdf(Normal(aggregated_mean[i, t], aggregated_std[i, t]), data[i, t, k])
        push!(calibration_probs, p_val)
    end
end
sorted_cal = sort(calibration_probs)
n_cal = length(sorted_cal)
theoretical = [(i - 0.5) / n_cal for i in 1:n_cal]
Plots.plot(theoretical, sorted_cal, seriestype=:scatter,
    xlabel="Theoretical Quantiles", ylabel="Empirical Quantiles",
    label="Calibration", title="Calibration Plot")
Plots.plot!(theoretical, theoretical, label="Ideal", lw=2)

# 3.5 QQ Plot of Standardized Residuals: Check if standardized residuals follow a N(0,1)
standardized_res = Float64[]
for i in 1:n_vars, t in 1:n_time, k in 1:n_samples_data
    if !ismissing(data[i, t, k]) && aggregated_std[i, t] > 0
        s_res = (data[i, t, k] - aggregated_mean[i, t]) / aggregated_std[i, t]
        push!(standardized_res, s_res)
    end
end
sorted_std_res = sort(standardized_res)
n_std = length(sorted_std_res)
theoretical_std = [quantile(Normal(0, 1), (i - 0.5) / n_std) for i in 1:n_std]
Plots.scatter(theoretical_std, sorted_std_res,
    xlabel="Theoretical Quantiles", ylabel="Standardized Residuals",
    label="Residuals", title="QQ Plot of Standardized Residuals")
Plots.plot!(theoretical_std, theoretical_std, label="45° Line", lw=2)

# 3.6 Summary Tables:
summary_stats = DataFrame(
    Metric = ["RMSE", "MAE", "Posterior Predictive p-value"],
    Value  = [rmse, mae, ppp]
)
PrettyTables.pretty_table(summary_stats)
