#!/usr/bin/env julia

using PathoSpread
using CairoMakie
using DataFrames
using CSV
using Statistics
using Printf
using MCMCChains

const MAPT_PATH = get(ENV, "COPATHOLOGY_MAPT_PATH", "simulations/syn_mapt_DIFFGA_RETRO.jls")
const APP_PATH = get(ENV, "COPATHOLOGY_APP_PATH", "simulations/syn_app_DIFFGA_RETRO.jls")
const OUTROOT = get(ENV, "COPATHOLOGY_OUTROOT", "figures/copathology")
const INFERENCES = [("MAPT", MAPT_PATH), ("APP", APP_PATH)]
const OUTDIR = joinpath(OUTROOT, "statistics")
const RHAT_THRESHOLDS = [1.0, 1.01, 1.05, 1.1]


function parameter_family(name::AbstractString)
    startswith(name, "beta[") && return "beta"
    startswith(name, "gamma[") && return "gamma"
    return "global"
end


function semantic_diagnostics(chain::Chains, priors)
    diag = MCMCChains.ess_rhat(chain)
    raw_names = String.(diag.nt.parameters)
    priorkeys = collect(keys(priors))

    ess_bulk = hasproperty(diag.nt, :ess_bulk) ? diag.nt.ess_bulk : diag.nt.ess
    ess_tail = hasproperty(diag.nt, :ess_tail) ? diag.nt.ess_tail : diag.nt.ess

    rows = DataFrame(
        raw_name = String[],
        semantic_name = String[],
        family = String[],
        rhat = Float64[],
        ess_bulk = Float64[],
        ess_tail = Float64[],
    )

    for (raw_name, rhat_val, ess_b, ess_t) in zip(raw_names, diag.nt.rhat, ess_bulk, ess_tail)
        raw_name in ("lp", "lp__") && continue
        semantic_name = raw_name
        if startswith(raw_name, "p[")
            idx = parse(Int, match(r"p\[(\d+)\]", raw_name).captures[1])
            if idx <= length(priorkeys)
                semantic_name = priorkeys[idx]
            end
        end
        push!(rows, (
            raw_name,
            semantic_name,
            parameter_family(semantic_name),
            Float64(rhat_val),
            Float64(ess_b),
            Float64(ess_t),
        ))
    end

    return rows
end


function add_threshold_lines!(ax::Axis)
    colors = [:gray60, :forestgreen, :darkorange, :firebrick]
    widths = [1.5, 2.0, 2.0, 2.5]
    for (thr, col, lw) in zip(RHAT_THRESHOLDS, colors, widths)
        hlines!(ax, [thr]; color=col, linestyle=:dash, linewidth=lw)
    end
    return nothing
end


function diagnostic_scatter(df::DataFrame, model_name::String, family::String)
    family_df = df[df.family .== family, :]
    sort!(family_df, :rhat, rev=true)

    fig = Figure(size=(1400, 800))
    ax = Axis(
        fig[1, 1],
        title = family == "global" ? "$model_name global diagnostics" : "$model_name $family diagnostics",
        xlabel = "Parameter index",
        ylabel = "Gelman-Rubin Rhat",
        titlesize = 34,
        xlabelsize = 28,
        ylabelsize = 28,
        xticklabelsize = 18,
        yticklabelsize = 18,
    )

    xs = 1:nrow(family_df)
    scatter!(ax, xs, family_df.rhat; color=:dodgerblue4, markersize=16)
    add_threshold_lines!(ax)
    ylims!(ax, 0.995, max(1.11, maximum(family_df.rhat) + 0.01))

    worst = first(family_df, min(8, nrow(family_df)))
    label_text = join([@sprintf("%s (%.3f)", r.semantic_name, r.rhat) for r in eachrow(worst)], "\n")
    text!(
        ax,
        0.02,
        0.98;
        space = :relative,
        text = "Worst Rhat values\n" * label_text,
        align = (:left, :top),
        fontsize = 18,
    )

    return fig
end


function main()
    mkpath(OUTDIR)
    PathoSpread.setup_plot_theme!(font="Arial", base=20, lw=3, markersize=12, dpi=300)

    summary_rows = DataFrame(
        model = String[],
        family = String[],
        n_parameters = Int[],
        max_rhat = Float64[],
        median_rhat = Float64[],
        n_rhat_gt_1p01 = Int[],
        n_rhat_gt_1p05 = Int[],
        n_rhat_gt_1p10 = Int[],
        min_ess_bulk = Float64[],
        min_ess_tail = Float64[],
    )

    for (model_name, path) in INFERENCES
        inf = load_inference(path)
        df = semantic_diagnostics(inf["chain"], inf["priors"])
        CSV.write(joinpath(OUTDIR, lowercase(model_name) * "_diagnostics.csv"), df)

        for family in ("global", "beta", "gamma")
            family_df = df[df.family .== family, :]
            isempty(family_df) && continue
            push!(summary_rows, (
                model_name,
                family,
                nrow(family_df),
                maximum(family_df.rhat),
                median(family_df.rhat),
                count(>(1.01), family_df.rhat),
                count(>(1.05), family_df.rhat),
                count(>(1.10), family_df.rhat),
                minimum(family_df.ess_bulk),
                minimum(family_df.ess_tail),
            ))
            fig = diagnostic_scatter(family_df, model_name, family)
            stem = lowercase(model_name) * "_" * family * "_rhat"
            save(joinpath(OUTDIR, stem * ".pdf"), fig)
            save(joinpath(OUTDIR, stem * ".png"), fig)
        end
    end

    CSV.write(joinpath(OUTDIR, "diagnostic_summary.csv"), summary_rows)
    open(joinpath(OUTDIR, "diagnostic_summary.txt"), "w") do io
        for row in eachrow(summary_rows)
            println(
                io,
                @sprintf(
                    "%s %s: n=%d, max Rhat=%.3f, median Rhat=%.3f, >1.01=%d, >1.05=%d, >1.10=%d, min ESS bulk=%.1f, min ESS tail=%.1f",
                    row.model,
                    row.family,
                    row.n_parameters,
                    row.max_rhat,
                    row.median_rhat,
                    row.n_rhat_gt_1p01,
                    row.n_rhat_gt_1p05,
                    row.n_rhat_gt_1p10,
                    row.min_ess_bulk,
                    row.min_ess_tail,
                ),
            )
        end
    end

    println("Saved diagnostics to $OUTDIR")
end


main()
