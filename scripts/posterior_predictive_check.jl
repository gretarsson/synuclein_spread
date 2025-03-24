using Plots
using Statistics
using PrettyTables
using Serialization

# Include your helper functions (make sure helpers.jl defines posterior_pred_mode, etc.)
include("helpers.jl")

# -------------------------------
# 1. Load Inference and Data
# -------------------------------
simulation = "total_death_simplifiedii_N=448_threads=4_var1_normalpriors"
inference_obj = deserialize("simulations/" * simulation * ".jls")
data = inference_obj["data"]  # dimensions: 448 × 8 × 12

# Compute the posterior predictive distributions using the mode.
# Here, post_pred is assumed to be a 2D array of Normal distributions (n_vars × n_time)
post_pred = posterior_pred_mode(inference_obj)

# ------------------------------------------------------------------
# Preliminary: Extract dimensions and compute predictive summaries
# ------------------------------------------------------------------
n_vars, n_time, n_samples = size(data)

# Pre-compute predicted means and standard deviations from post_pred.
pred_means = Array{Float64}(undef, n_vars, n_time)
pred_stds  = Array{Float64}(undef, n_vars, n_time)
for i in 1:n_vars
    for t in 1:n_time
         d = post_pred[i, t]  # assumed to be a Normal distribution
         pred_means[i, t] = mean(d)  # Alternatively, d.μ if available
         pred_stds[i, t]  = std(d)   # Alternatively, d.σ if available
    end
end

# ------------------------------------------------------------------
# 1. Residual Analysis
# ------------------------------------------------------------------
# Compute residuals for each observed value.
# Note: For missing data, we assign NaN so that they are ignored in numerical summaries.
residuals = similar(data, Float64)
for i in 1:n_vars
    for t in 1:n_time
         for k in 1:n_samples
              if ismissing(data[i, t, k])
                  residuals[i, t, k] = NaN
              else
                  residuals[i, t, k] = data[i, t, k] - pred_means[i, t]
              end
         end
    end
end

# Flatten residuals (ignoring NaN) to compute overall RMSE and MAE.
all_res = [r for r in residuals if !isnan(r)]
rmse = sqrt(mean(all_res .^ 2))
mae = mean(abs.(all_res))

println("Residual Analysis:")
println("RMSE: ", rmse)
println("MAE: ", mae)

# ------------------------------------------------------------------
# 2. Coverage Analysis
# ------------------------------------------------------------------
# For each observation, check if it lies within the 95% credible interval.
coverage = zeros(n_vars, n_time)
for i in 1:n_vars
    for t in 1:n_time
         lower = quantile(post_pred[i, t], 0.025)
         upper = quantile(post_pred[i, t], 0.975)
         # Extract non-missing observations for this variable and time point.
         vals = [data[i, t, k] for k in 1:n_samples if !ismissing(data[i, t, k])]
         if !isempty(vals)
              cov = sum(x -> (x ≥ lower && x ≤ upper) ? 1 : 0, vals) / length(vals)
         else
              cov = NaN
         end
         coverage[i, t] = cov
    end
end

overall_coverage = mean([c for c in coverage if !isnan(c)])
println("Overall 95% coverage: ", overall_coverage)

# ------------------------------------------------------------------
# 3. Posterior Predictive p-values
# ------------------------------------------------------------------
# For each non-missing observation, simulate one replicate from the predictive
# distribution and check if the simulated value is greater than the observed value.
ppp_vals = Float64[]
for i in 1:n_vars
    for t in 1:n_time
         for k in 1:n_samples
              if !ismissing(data[i, t, k])
                  y_obs = data[i, t, k]
                  y_rep = rand(post_pred[i, t])
                  push!(ppp_vals, (y_rep > y_obs) ? 1.0 : 0.0)
              end
         end
    end
end
ppp = mean(ppp_vals)
println("Overall Posterior Predictive p-value: ", ppp)

# ------------------------------------------------------------------
# 4. Error Heatmap
# ------------------------------------------------------------------
# For each variable and time, compute the average absolute residual.
avg_abs_res = Array{Float64}(undef, n_vars, n_time)
for i in 1:n_vars
    for t in 1:n_time
         cell_res = [abs(data[i, t, k] - pred_means[i, t]) for k in 1:n_samples if !ismissing(data[i, t, k])]
         avg_abs_res[i, t] = isempty(cell_res) ? NaN : mean(cell_res)
    end
end

# Plot the heatmap (variables on y-axis, time points on x-axis).
Plots.heatmap(avg_abs_res,
    xlabel = "Time Point",
    ylabel = "Variable Index",
    title = "Heatmap of Average Absolute Residuals")

# ------------------------------------------------------------------
# 5. Calibration Plot
# ------------------------------------------------------------------
# Compute the cumulative probability (cdf) of each observed value under its predictive distribution.
calibration_probs = Float64[]
for i in 1:n_vars
    for t in 1:n_time
         for k in 1:n_samples
              if !ismissing(data[i, t, k])
                  p_val = cdf(post_pred[i, t], data[i, t, k])
                  push!(calibration_probs, p_val)
              end
         end
    end
end

# Sort the empirical probabilities and compute the corresponding theoretical quantiles.
sorted_cal = sort(calibration_probs)
n_cal = length(sorted_cal)
theoretical = [(i - 0.5) / n_cal for i in 1:n_cal]

# Create a scatter plot of the empirical vs. theoretical quantiles.
Plots.plot(theoretical, sorted_cal,
    seriestype = :scatter,
    xlabel = "Theoretical Quantiles",
    ylabel = "Empirical Quantiles",
    label = "Calibration",
    title = "Calibration Plot")
# Add the 45° reference line.
Plots.plot!(theoretical, theoretical, label = "Ideal", lw = 2)

# ------------------------------------------------------------------
# 6. QQ Plot of Standardized Residuals
# ------------------------------------------------------------------
# Compute standardized residuals: (observed - predicted mean) / predicted std.
standardized_res = Float64[]
for i in 1:n_vars
    for t in 1:n_time
         for k in 1:n_samples
              if !ismissing(data[i, t, k]) && pred_stds[i, t] > 0
                  s_res = (data[i, t, k] - pred_means[i, t]) / pred_stds[i, t]
                  push!(standardized_res, s_res)
              end
         end
    end
end

sorted_std_res = sort(standardized_res)
n_std = length(sorted_std_res)
theoretical_std = [quantile(Normal(0, 1), (i - 0.5) / n_std) for i in 1:n_std]

# Scatter plot of standardized residuals vs. theoretical quantiles.
Plots.scatter(theoretical_std, sorted_std_res,
    xlabel = "Theoretical Quantiles",
    ylabel = "Standardized Residuals",
    label = "Residuals",
    title = "QQ Plot of Standardized Residuals")
# Add the 45° line.
Plots.plot!(theoretical_std, theoretical_std, label = "45° Line", lw = 2)

# ------------------------------------------------------------------
# 7. Summary Tables
# ------------------------------------------------------------------
# Create a table summarizing the overall metrics.
summary_stats = DataFrame(
    Metric = ["RMSE", "MAE", "Overall 95% Coverage", "Posterior Predictive p-value"],
    Value  = [rmse, mae, overall_coverage, ppp]
)
PrettyTables.pretty_table(summary_stats)

# Additionally, you can create a table per time point.
time_stats = DataFrame(Time = 1:n_time, RMSE = zeros(n_time), MAE = zeros(n_time), Coverage = zeros(n_time))
for t in 1:n_time
    cell_res = Float64[]
    for i in 1:n_vars
         for k in 1:n_samples
              if !ismissing(data[i, t, k])
                  r = data[i, t, k] - pred_means[i, t]
                  push!(cell_res, r)
              end
         end
    end
    if !isempty(cell_res)
         time_stats.RMSE[t] = sqrt(mean(cell_res .^ 2))
         time_stats.MAE[t]  = mean(abs.(cell_res))
         # Coverage per time point
         total = 0
         in_interval = 0
         for i in 1:n_vars
             for k in 1:n_samples
                if !ismissing(data[i, t, k])
                    total += 1
                    lower = quantile(post_pred[i, t], 0.025)
                    upper = quantile(post_pred[i, t], 0.975)
                    if data[i, t, k] ≥ lower && data[i, t, k] ≤ upper
                        in_interval += 1
                    end
                end
             end
         end
         time_stats.Coverage[t] = in_interval / total
    else
         time_stats.RMSE[t] = NaN
         time_stats.MAE[t] = NaN
         time_stats.Coverage[t] = NaN
    end
end
PrettyTables.pretty_table(time_stats)
