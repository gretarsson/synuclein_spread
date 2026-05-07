#!/usr/bin/env julia

using PathoSpread
using StatsPlots
using Plots
using Printf
using MCMCChains
using OrderedCollections
using LaTeXStrings

const MODEL_KEY = get(ENV, "RHAT_MODEL_KEY", "HIPPO_DIFFGA_RETRO")
const INFERENCE_PATH = get(ENV, "RHAT_INFERENCE_PATH", "simulations/hippo_DIFFGA_RETRO.jls")
const SAVE_DIR = "figures/rhat_plots"

const RHAT_LINES = [
    (1.00, :gray,   1.0),
    (1.01, :green,  1.5),
    (1.05, :orange, 2.0),
    (1.10, :red,    2.0),
]


is_local_param(name::String) = startswith(name, "beta[") || startswith(name, "gamma[")


function compute_rhat_semantic(chain::Chains, priors::OrderedDict{String,Any})
    rhat_obj = MCMCChains.MCMCDiagnosticTools.rhat(chain)
    raw_names = String.(rhat_obj.nt.parameters)
    raw_vals = rhat_obj.nt.rhat
    priorkeys = collect(keys(priors))

    semantic_rhat = Dict{String,Float64}()
    for (pname_raw, rhat_val) in zip(raw_names, raw_vals)
        if pname_raw == "lp" || pname_raw == "lp__"
            continue
        end

        if startswith(pname_raw, "p[")
            idx = parse(Int, match(r"p\[(\d+)\]", pname_raw).captures[1])
            if idx <= length(priorkeys)
                semantic_rhat[priorkeys[idx]] = rhat_val
            end
        else
            semantic_rhat[pname_raw] = rhat_val
        end
    end

    return semantic_rhat
end


function split_local_params(rhats::Dict{String,Float64})
    beta = Dict{String,Float64}()
    gamma = Dict{String,Float64}()
    for (name, val) in rhats
        if startswith(name, "beta[")
            beta[name] = val
        elseif startswith(name, "gamma[")
            gamma[name] = val
        end
    end
    return beta, gamma
end


function plot_rhat_scatter(model_key::String, model_dir::String, rhats::Dict{String,Float64}; suffix::String)
    selected = collect(keys(rhats))
    isempty(selected) && return nothing

    sort!(selected; lt = (a, b) -> begin
        if a == "loglik" && b != "loglik"
            true
        elseif b == "loglik" && a != "loglik"
            false
        else
            a < b
        end
    end)

    ys = [rhats[n] for n in selected]
    xs = 1:length(selected)
    outfile = joinpath(model_dir, "$(suffix)_rhat.pdf")

    plt = scatter(
        xs,
        ys;
        xlabel = "Parameter index",
        ylabel = L"\hat{R}",
        title = "",
        markersize = 14,
        alpha = 0.8,
        color = :blue,
        legend = false,
    )

    for (val, col, lw) in RHAT_LINES
        hline!(plt, [val]; color=col, lw=lw, ls=:dash)
    end

    if length(selected) <= 25
        xticks!(plt, xs, selected)
        plot!(plt; xrotation=45)
    end

    savefig(plt, outfile)
    println("Saved → $outfile")
    return nothing
end


function main()
    mkpath(SAVE_DIR)
    setup_plot_theme!()

    model_dir = joinpath(SAVE_DIR, MODEL_KEY)
    mkpath(model_dir)

    inf = load_inference(INFERENCE_PATH)
    chain = inf["chain"]
    priors = inf["priors"]

    rhats = compute_rhat_semantic(chain, priors)
    global_rhats = Dict{String,Float64}()
    if haskey(inf, "loglik_rhat")
        global_rhats["loglik"] = inf["loglik_rhat"]
    end
    for (name, val) in rhats
        if !is_local_param(name)
            global_rhats[name] = val
        end
    end

    plot_rhat_scatter(MODEL_KEY, model_dir, global_rhats; suffix="global")

    beta_rhats, gamma_rhats = split_local_params(rhats)
    plot_rhat_scatter(MODEL_KEY, model_dir, beta_rhats; suffix="beta")
    plot_rhat_scatter(MODEL_KEY, model_dir, gamma_rhats; suffix="gamma")

    println("Done.")
end


main()
