using PathoSpread
using Statistics
using CairoMakie
using GLM
using DataFrames

# ------------------------------------------------------------
# Setup output directory
# ------------------------------------------------------------

outdir = "figures/data_variance"
isdir(outdir) || mkpath(outdir)

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

data_file = "data/total_path.csv"

data, timepoints = process_pathology(
    data_file;
    W_csv = "data/W_labeled.csv"
)

n_regions, n_times, n_samples = size(data)

# ------------------------------------------------------------
# 1) Variance over time (pooled regions × samples)
# ------------------------------------------------------------

mean_over_time = Float64[]
var_over_time  = Float64[]

for t in 1:n_times
    vals = vec(data[:, t, :])
    vals = skipmissing(vals) |> collect

    push!(mean_over_time, mean(vals))
    push!(var_over_time,  var(vals))
end

f1 = Figure(resolution = (600, 450))
ax1 = Axis(f1[1,1],
    xlabel = "Time index",
    ylabel = "Variance (pooled)"
)

lines!(ax1, 1:n_times, var_over_time, linewidth=3)
scatter!(ax1, 1:n_times, var_over_time)

save(joinpath(outdir, "variance_over_time.png"), f1)

# ------------------------------------------------------------
# 2) Mean–variance per region per timepoint
# ------------------------------------------------------------

mean_rt = Float64[]
var_rt  = Float64[]

for r in 1:n_regions
    for t in 1:n_times
        vals = data[r, t, :]
        vals = skipmissing(vals) |> collect

        if length(vals) > 1
            push!(mean_rt, mean(vals))
            push!(var_rt,  var(vals))
        end
    end
end

df = DataFrame(mean = mean_rt, var = var_rt)

# ------------------------------------------------------------
# Fit linear and quadratic models
# ------------------------------------------------------------

m_linear = lm(@formula(var ~ mean), df)
m_quad   = lm(@formula(var ~ mean + mean^2), df)

println("\nLinear model:")
println(m_linear)

println("\nQuadratic model:")
println(m_quad)

println("\nR² comparison:")
println("Linear R²     = ", r2(m_linear))
println("Quadratic R²  = ", r2(m_quad))

# ------------------------------------------------------------
# Create smooth fits
# ------------------------------------------------------------

μ_min = minimum(mean_rt)
μ_max = maximum(mean_rt)
μ_grid = range(μ_min, μ_max; length=400)

df_pred = DataFrame(mean = μ_grid)

var_lin  = Float64.(predict(m_linear, df_pred))
var_quad = Float64.(predict(m_quad, df_pred))

# ------------------------------------------------------------
# Plot mean–variance with fits
# ------------------------------------------------------------

f2 = Figure(resolution = (700, 500))
ax2 = Axis(f2[1,1],
    xlabel = "Mean (region,time)",
    ylabel = "Variance (across samples)"
)

scatter!(ax2, mean_rt, var_rt, markersize=14)
lines!(ax2, μ_grid, var_lin, linewidth=5)
lines!(ax2, μ_grid, var_quad, linewidth=5, linestyle=:dash)

save(joinpath(outdir, "variance_vs_mean_with_fits.png"), f2)



# ------------------------------------------------------------
# Build (mean, var, time) dataset
# ------------------------------------------------------------

mean_rt = Float64[]
var_rt  = Float64[]
time_rt = Int[]

for r in 1:n_regions
    for t in 1:n_times
        vals = data[r, t, :]
        vals = skipmissing(vals) |> collect

        if length(vals) > 1
            push!(mean_rt, mean(vals))
            push!(var_rt,  var(vals))
            push!(time_rt, t)
        end
    end
end

df = DataFrame(mean = mean_rt,
               var  = var_rt,
               time = time_rt)

# ------------------------------------------------------------
# Fit variance models
# ------------------------------------------------------------

m1 = lm(@formula(var ~ mean), df)
m2 = lm(@formula(var ~ mean + mean^2), df)
m3 = lm(@formula(var ~ mean + mean^2 + time), df)

# ------------------------------------------------------------
# Print results to file
# ------------------------------------------------------------

outfile = joinpath(outdir, "variance_model_fits.txt")

open(outfile, "w") do io

    println(io, "==============================")
    println(io, "Variance modeling results")
    println(io, "==============================\n")

    println(io, "Model 1: var ~ mean")
    println(io, m1)
    println(io, "R² = ", r2(m1))
    println(io, "\n---------------------------------\n")

    println(io, "Model 2: var ~ mean + mean^2")
    println(io, m2)
    println(io, "R² = ", r2(m2))
    println(io, "\n---------------------------------\n")

    println(io, "Model 3: var ~ mean + mean^2 + time")
    println(io, m3)
    println(io, "R² = ", r2(m3))
    println(io, "\n---------------------------------\n")

    println(io, "R² comparison:")
    println(io, "M1 (linear):        ", r2(m1))
    println(io, "M2 (quadratic):     ", r2(m2))
    println(io, "M3 (+ time term):   ", r2(m3))
end

println("Saved variance model results to: ", outfile)