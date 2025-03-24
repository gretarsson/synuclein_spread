
include("helpers.jl");
using CSV

# simulation to analyze
simulation = "total_death_simplifiedii_bilateral_N=444_threads=1_var1";
inference_obj = deserialize("simulations/"*simulation*".jls")

# predicted vs observed plots with correlation
figs = predicted_vs_observed_plots(inference_obj; save_path="figures/predicted_observed_retro", plotscale=:log10, remove_zero=true, aspect_ratio=:equal, plot_identity_line=true, plot_fit_line=false)
