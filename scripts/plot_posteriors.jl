#!/usr/bin/env julia
using StatsPlots
using Serialization
using Distributions
using OrderedCollections
using MCMCChains
using Plots
using PathoSpread
using LaTeXStrings


# ------------------------------------------------------------
# Helper: get posterior samples for p[i]
#
# IMPORTANT:
# We use Symbol("p[i]") exactly because this is how your chains are structured,
# and the index i matches the priors OrderedDict ordering.
# ------------------------------------------------------------
function posterior_samples_from_p(chain::Chains, i::Int)
    var = Symbol("p[$i]")

    # check existence
    if !(var in names(chain))
        return nothing
    end

    # MCMCChains 5.x: correct extraction
    vals = chain[var]               # returns parameter series
    return vec(Array(vals))         # convert to Vector{Float64}
end

using LaTeXStrings

latex_names = Dict(
    "rho"   => L"\rho",
    "alpha" => L"\alpha",
    "beta"  => L"\beta",
    "gamma" => L"\gamma",
    "sigma" => L"\sigma",
    "tau"   => L"\tau",
    "d"     => L"d"
)



# ============================================================
# USER SETTINGS
# ============================================================

global_parameter_names = ["rho", "alpha"]  # extend manually as needed

inference_files = Dict(
    "DIFF"   => "simulations/DIFF_RETRO.jls",
    "DIFFG"  => "simulations/DIFFG_RETRO.jls",
    "DIFFGA" => "simulations/DIFFGA_RETRO.jls"
)

save_dir = "figures/posteriors"
mkpath(save_dir)


# ============================================================
# LOAD INFERENCES
# ============================================================
println("Loading inference files...")
inferences = Dict{String,Dict}()

for (model, path) in inference_files
    inferences[model] = load_inference(path)
end

println("Loaded models: ", join(keys(inferences), ", "))


# ============================================================
# COLORS FOR CONSISTENCY
# ============================================================
model_colors = Dict(
    "DIFF"   => :blue,
    "DIFFG"  => :red,
    "DIFFGA" => :green
)


# ============================================================
# MAIN LOOP
# ============================================================
setup_plot_theme!()  # set plotting settings
for pname in global_parameter_names
    println("Plotting parameter: $pname")

    # ---------------- PRIOR ----------------
    prior_dist = nothing
    for model in keys(inferences)
        priors = inferences[model]["priors"]
        if haskey(priors, pname)
            prior_dist = priors[pname]
            break
        end
    end

    if prior_dist === nothing
        @warn "No prior found for parameter $pname in any model; skipping."
        continue
    end

    # only add legend for rho
    plt = if pname == "rho"
        # legend ON, but smaller
        Plots.plot(
            xlabel = latex_names[pname],
            ylabel = "Density",
            legend = :topleft,
            legendfontsize = 14,   # smaller legend
        )
    else
        # legend OFF
        Plots.plot(
            xlabel = latex_names[pname],
            ylabel = "Density",
            legend = false,
        )
    end

    Plots.plot!(plt, prior_dist; lw=4, ls=:dash, color=:black, label="prior")

    # ---------------- POSTERIORS ----------------
    for model in keys(inferences)
        inf    = inferences[model]
        priors = inf["priors"]
        chain  = inf["chain"]

        # find index i such that priors order matches p[i]
        prior_keys = collect(keys(priors))
        idx = findfirst(==(pname), prior_keys)

        if idx === nothing
            @info "Parameter $pname not in $model; skipping."
            continue
        end

        samples = posterior_samples_from_p(chain, idx)

        if samples === nothing
            @warn "Chain for $model does not have p[$idx] for $pname."
            continue
        end

        density!(
            plt,
            samples;
            lw    = 4,
            color = model_colors[model],
            label = model
        )
    end

    outfile = joinpath(save_dir, "$(pname)_comparison.pdf")
    savefig(plt, outfile)
    println("Saved → $outfile")
end

println("Done.")



# ============================================================
# SCATTER PLOT OF MEAN β[i] : DIFFG vs DIFFGA
# ============================================================

println("Computing β[i] scatter plot…")


# Identify β parameters from the priors of DIFFG or DIFFGA
beta_names = String[]
for model in ["DIFFG", "DIFFGA"]
    priors = inferences[model]["priors"]
    append!(beta_names, filter(n -> startswith(n, "beta["), collect(keys(priors))))
end

beta_names = unique(beta_names)  # ensure no duplicates
println("Found $(length(beta_names)) beta parameters.")

# Extract mean β[i] for each model
means_G  = Float64[]
means_GA = Float64[]

for bname in beta_names
    # --- DIFFG ---
    priors_G  = inferences["DIFFG"]["priors"]
    chain_G   = inferences["DIFFG"]["chain"]
    idx_G     = findfirst(==(bname), collect(keys(priors_G)))
    push!(means_G, mean(posterior_samples_from_p(chain_G, idx_G)))

    # --- DIFFGA ---
    priors_GA = inferences["DIFFGA"]["priors"]
    chain_GA  = inferences["DIFFGA"]["chain"]
    idx_GA    = findfirst(==(bname), collect(keys(priors_GA)))
    push!(means_GA, mean(posterior_samples_from_p(chain_GA, idx_GA)))
end

# Scatter plot: DIFFG vs DIFFGA
beta_scatter = Plots.scatter(
    means_G, means_GA;
    xlabel = "DIFFG",
    ylabel = "DIFFGA",
    title = L"β_i",
    markersize = 8,
    color = :blue,
    legend = false,
    alpha=0.8
)
findall(means_G .> 10)

# Add diagonal y = x
mx = max(maximum(means_G), maximum(means_GA))
Plots.plot!(beta_scatter, [0, mx], [0, mx]; color = :black, lw=2, ls=:dash)


outfile_scatter = joinpath(save_dir, "beta_scatter.pdf")
savefig(beta_scatter, outfile_scatter)
println("Saved → $outfile_scatter")



# ============================================================
# SCATTER PLOT β[i] vs γ[i] for DIFFGA
# ============================================================

println("Computing β[i] vs γ[i] scatter for DIFFGA…")

priors_GA = inferences["DIFFGA"]["priors"]
chain_GA  = inferences["DIFFGA"]["chain"]

# Find beta[i] and gamma[i] parameter names
beta_names_GA  = filter(n -> startswith(n, "beta["),  collect(keys(priors_GA)))
gamma_names_GA = filter(n -> startswith(n, "gamma["), collect(keys(priors_GA)))

# Sort the names to guarantee consistent ordering:
# beta[1], beta[2], ..., beta[412]
sort!(beta_names_GA,  by = x -> parse(Int, match(r"beta\[(\d+)\]", x).captures[1]))
sort!(gamma_names_GA, by = x -> parse(Int, match(r"gamma\[(\d+)\]", x).captures[1]))

# Sanity check
if length(beta_names_GA) != length(gamma_names_GA)
    error("Mismatch between β and γ counts in DIFFGA")
end

# Extract posterior means
mean_beta = Float64[]
mean_gamma = Float64[]

for (bname, gname) in zip(beta_names_GA, gamma_names_GA)

    # β index
    idx_b = findfirst(==(bname), collect(keys(priors_GA)))
    push!(mean_beta, mean(posterior_samples_from_p(chain_GA, idx_b)))

    # γ index
    idx_g = findfirst(==(gname), collect(keys(priors_GA)))
    push!(mean_gamma, mean(posterior_samples_from_p(chain_GA, idx_g)))
end

# Scatter plot β[i] vs γ[i]
beta_gamma_scatter = Plots.scatter(
    mean_beta, mean_gamma;
    xlabel = L"\beta_i",
    ylabel = L"\gamma_i",
    title = "DIFFGA",
    markersize = 8,
    color = :green,
    alpha = 0.7,
    legend = false,
)

outfile_bg = joinpath(save_dir, "beta_gamma_scatter_DIFFGA.pdf")
savefig(beta_gamma_scatter, outfile_bg)
println("Saved → $outfile_bg")
