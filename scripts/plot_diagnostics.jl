#!/usr/bin/env julia

using PathoSpread
using Serialization
using StatsPlots
using Plots
using Glob
using Printf
using MCMCChains
using LaTeXStrings
using OrderedCollections

# ============================================================
# SETTINGS
# ============================================================

# Add as many as you want — automatically handled
inference_files = Dict(
    "DIFF_RETRO"      => "simulations/DIFF_RETRO.jls",
    "DIFF_ANTERO"     => "simulations/DIFF_ANTERO.jls",
    "DIFF_BIDIR"      => "simulations/DIFF_BIDIR.jls",
    "DIFF_EUCL"       => "simulations/DIFF_EUCL.jls",

    "DIFFG_RETRO"     => "simulations/DIFFG_RETRO.jls",
    "DIFFG_ANTERO"    => "simulations/DIFFG_ANTERO.jls",
    "DIFFG_BIDIR"     => "simulations/DIFFG_BIDIR.jls",
    "DIFFG_EUCL"      => "simulations/DIFFG_EUCL.jls",

    "DIFFGA_RETRO"    => "simulations/DIFFGA_RETRO.jls",
    "DIFFGA_ANTERO"   => "simulations/DIFFGA_ANTERO_CUT.jls",  # removed one chain
    "DIFFGA_BIDIR"    => "simulations/DIFFGA_BIDIR.jls",
    "DIFFGA_EUCL"     => "simulations/DIFFGA_EUCL_CUT.jls",  # removed one chain
)

save_dir = "figures/rhat_plots"
mkpath(save_dir)

is_local_param(name::String) =
    startswith(name, "beta[") || startswith(name, "gamma[")

rhat_lines = [
    (1.00, :gray,   1.0),
    (1.01, :green,  1.5),
    (1.05, :orange, 2.0),
    (1.10, :red,    2.0),
]

setup_plot_theme!()

# ============================================================
# Identify model type
# ============================================================

function get_model_type(model_key::String)
    if startswith(model_key, "DIFFGA")
        return "DIFFGA"
    elseif startswith(model_key, "DIFFG")
        return "DIFFG"
    elseif startswith(model_key, "DIFF")
        return "DIFF"
    else
        error("Unknown model key: $model_key")
    end
end

# ============================================================
# R̂ extraction → semantic
# ============================================================

function compute_rhat_semantic(chain::Chains,
                               priors::OrderedDict{String,Any})

    rhat_obj = MCMCChains.MCMCDiagnosticTools.rhat(chain)

    raw_names = String.(rhat_obj.nt.parameters)
    raw_vals  = rhat_obj.nt.rhat
    priorkeys = collect(keys(priors))

    semantic_rhat = Dict{String,Float64}()

    for (pname_raw, rhat_val) in zip(raw_names, raw_vals)

        if pname_raw == "lp" || pname_raw == "lp__"
            continue
        end

        if startswith(pname_raw, "p[")
            idx = parse(Int, match(r"p\[(\d+)\]", pname_raw).captures[1])
            if idx <= length(priorkeys)
                semantic_name = priorkeys[idx]
                semantic_rhat[semantic_name] = rhat_val
            end

        else
            semantic_rhat[pname_raw] = rhat_val
        end
    end

    return semantic_rhat
end

# ============================================================
# Split local beta / gamma
# ============================================================

function split_local_params(rhats::Dict{String,Float64})
    beta, gamma = Dict{String,Float64}(), Dict{String,Float64}()
    for (name, val) in rhats
        if startswith(name, "beta[")
            beta[name] = val
        elseif startswith(name, "gamma[")
            gamma[name] = val
        end
    end
    return beta, gamma
end

# ============================================================
# PLOTTER — loglik always first
# ============================================================

function plot_rhat_scatter(model_key::String,
                           model_dir::String,
                           rhats::Dict{String,Float64};
                           suffix::String)

    selected = collect(keys(rhats))

    if isempty(selected)
        @info "No parameters for $model_key ($suffix)"
        return
    end

    # ---- FIXED SORTING: make loglik first ----
    sort!(selected; lt = (a, b) -> begin
        if a == "loglik" && b != "loglik"
            return true
        elseif b == "loglik" && a != "loglik"
            return false
        else
            return a < b
        end
    end)

    ys = [rhats[n] for n in selected]
    xs = 1:length(selected)

    title_str = ""
    outfile = joinpath(model_dir, "$(suffix)_rhat.pdf")

    plt = scatter(xs, ys;
        xlabel = "Parameter index",
        ylabel = L"\hat{R}",
        title = title_str,
        #markersize = 7,
        alpha = 0.8,
        color = :blue,
        legend = false,
    )

    for (val, col, lw) in rhat_lines
        hline!(plt, [val]; color = col, lw = lw, ls = :dash)
    end

    if length(selected) <= 25
        xticks!(plt, xs, selected)
        plot!(plt; xrotation=45)
    end

    savefig(plt, outfile)
    println("Saved → $outfile")
end

# ============================================================
# MAIN LOOP
# ============================================================

for (model_key, path) in inference_files
    println("Processing model: $model_key")

    # Folder
    model_dir = joinpath(save_dir, model_key)
    mkpath(model_dir)

    # Load inference
    inf    = load_inference(path)
    chain  = inf["chain"]
    priors = inf["priors"]

    # Standard parameter R̂s
    rhats = compute_rhat_semantic(chain, priors)
    modeltype = get_model_type(model_key)

    # ---------- LOG-LIKELIHOOD ----------
    loglik_mat =  inf["loglik_mat"]
    rhat_loglik = inf["loglik_rhat"]

    # ---------- GLOBAL ----------
    global_rhats = Dict{String,Float64}()

    # 1) insert loglik first
    global_rhats["loglik"] = rhat_loglik

    # 2) insert normal global parameters
    for (name, val) in rhats
        if !is_local_param(name)
            global_rhats[name] = val
        end
    end

    plot_rhat_scatter(model_key, model_dir, global_rhats; suffix="global")

    # ---------- LOCAL ----------
    if modeltype == "DIFF"
        continue
    end

    beta_rhats, gamma_rhats = split_local_params(rhats)

    if modeltype == "DIFFG"
        plot_rhat_scatter(model_key, model_dir, beta_rhats; suffix="beta")

    elseif modeltype == "DIFFGA"
        plot_rhat_scatter(model_key, model_dir, beta_rhats;  suffix="beta")
        plot_rhat_scatter(model_key, model_dir, gamma_rhats; suffix="gamma")
    end
end

println("Done.")
