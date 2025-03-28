using Serialization
include("helpers.jl")
using Plots

# Load one inference object (or loop over multiple if desired)
# For demonstration, we assume one inference object.
inference = deserialize("simulations/total_death_simplifiedii_N=448_threads=4_var1_normalpriors.jls")
errors = compute_region_errors(inference; M=1000)  # errors is an (n_regions x n_timepoints) matrix
n_regions, n_timepoints = size(errors)
timepoints = inference["timepoints"]

# Create one scatter plot per timepoint:
timepoint_plots = []
for t in 1:n_timepoints
    p = Plots.scatter(1:n_regions, errors[:, t],
        xlabel = "Region",
        ylabel = "MSE",
        title = "MSE across regions at time = $(timepoints[t])",
        label = "Time $(timepoints[t])",
        legend = :topleft)
    push!(timepoint_plots, p)
end

# Save or display individual timepoint plots
# For example, to save:
for (t, p) in enumerate(timepoint_plots)
    Plots.savefig(p, "mse_timepoint_$(t).png")
end

# Create a summary plot overlaying scatter points for all timepoints:
p_summary = Plots.plot(xlabel = "Region", ylabel = "MSE",
    title = "MSE across regions for all timepoints", legend = :topleft)
for t in 1:n_timepoints
    Plots.scatter!(p_summary, 1:n_regions, errors[:, t], label = "Time $(timepoints[t])")
end
Plots.savefig(p_summary, "mse_summary.png")

# Optionally, display the plots
display(p_summary)
for p in timepoint_plots
    display(p)
end
