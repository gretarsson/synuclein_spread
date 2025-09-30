module PathoSpread

# ---- dependencies your helpers/odes use ----
using LinearAlgebra, Statistics, Random, SparseArrays
using DifferentialEquations, Distributions
using Turing, SciMLSensitivity 
using KernelDensity
using DataFrames, StatsBase, GLM
using ParetoSmooth
using StatsPlots, Plots, CairoMakie, LaTeXStrings
using DelimitedFiles, Serialization
using LSODA
using ReverseDiff, Enzyme, Zygote
using LazyArrays
using Tables
using DataStructures: OrderedDict   # you use OrderedDict in helpers
using CSV

# ---- include order matters: ODEs first so helpers see `odes` ----
include("odes.jl")
include("ode_dimensions.jl")
include("model_priors.jl")
include("helpers.jl")
include("helpers_plots.jl")
include("data_processing.jl")

# ---- exports (start minimal; add more when you need them) ----
# helpers.jl
export infer, make_ode_problem, build_region_groups, read_W, read_data, load_inference, save_inference
# helpers_plots.jl
export plot_inference, setup_plot_theme!
# data_processing.jl
export process_pathology
# odes.jl
export odes
# ode_dimensions.jl
export ode_dimensions
# model_priors.jl  
export get_priors



#export odes
#export ode_dimensions
#export infer, compute_psis_loo, compute_waic, compute_aic_bic, posterior_mode
#export read_W, make_ode_problem, nonzero_regions
#export predicted_vs_observed_plots


end # module PathoSpread
