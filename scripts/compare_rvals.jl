
include("helpers.jl");
using CSV

# simulation to analyze
simulations = ["total_death_simplifiedii_bilateral_N=444_threads=1_var1",
               "total_death_simplifiedii_bilateral_antero_N=444_threads=1_var1",
               "total_death_simplifiedii_bilateral2_retroantero_N=444_threads=1_var1"]
model_names = ["retrograde", "anterograde", "bidirectional"]
inferences = Dict[];
for simulation in simulations
    push!(inferences,deserialize("simulations/"*simulation*".jls"))
end


# Choose the posterior sample indices to use.
# For example, use the first 100 samples.
sample_indices = 1:1000

# Specify a folder in which to save the box plot figures.
save_folder = "boxplots_r_values"

# Call the function to generate box plots.
# This will create one box plot per timepoint comparing the r-value distributions across models.
figures = boxplot_r_values(inferences, sample_indices=sample_indices, save_path=save_folder, model_names=model_names)
