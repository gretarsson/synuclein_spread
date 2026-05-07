#!/usr/bin/env julia

using PathoSpread
using CairoMakie
using DataFrames
using CSV
using Statistics
using Printf

const MAPT_PATH = get(ENV, "COPATHOLOGY_MAPT_PATH", "simulations/syn_mapt_DIFFGA_RETRO.jls")
const APP_PATH = get(ENV, "COPATHOLOGY_APP_PATH", "simulations/syn_app_DIFFGA_RETRO.jls")
const OUTROOT = get(ENV, "COPATHOLOGY_OUTROOT", "figures/copathology")
const OUTDIR = joinpath(OUTROOT, "parameter_scatter")


function parameter_df(family::String)
    inf_mapt = load_inference(MAPT_PATH)
    inf_app = load_inference(APP_PATH)
    labels = String.(inf_mapt["labels"])
    labels == String.(inf_app["labels"]) || error("Region labels differ between MAPT and APP.")

    mapt = PathoSpread.posterior_mean_vector_from_priors(inf_mapt["chain"], inf_mapt["priors"], family, length(labels))
    app = PathoSpread.posterior_mean_vector_from_priors(inf_app["chain"], inf_app["priors"], family, length(labels))

    df = DataFrame(region=labels, mean_mapt=mapt, mean_app=app)
    df.delta = df.mean_app .- df.mean_mapt
    df.abs_delta = abs.(df.delta)
    sort!(df, :abs_delta, rev=true)
    return df
end


function add_identity_line!(ax::Axis, x::AbstractVector, y::AbstractVector)
    lo = min(minimum(x), minimum(y))
    hi = max(maximum(x), maximum(y))
    pad = 0.06 * (hi - lo + eps())
    lines!(ax, [lo - pad, hi + pad], [lo - pad, hi + pad]; color=:gray55, linestyle=:dash, linewidth=3)
    xlims!(ax, lo - pad, hi + pad)
    ylims!(ax, lo - pad, hi + pad)
    return nothing
end


function scatter_figure(df::DataFrame, family::String)
    fig = Figure(size=(1200, 1000))
    ax = Axis(
        fig[1, 1],
        title = uppercase(family) * " posterior means: APP vs MAPT",
        xlabel = "MAPT posterior mean",
        ylabel = "APP posterior mean",
        titlesize = 36,
        xlabelsize = 30,
        ylabelsize = 30,
        xticklabelsize = 22,
        yticklabelsize = 22,
    )

    scatter!(ax, df.mean_mapt, df.mean_app; color=(:black, 0.65), markersize=18)
    add_identity_line!(ax, df.mean_mapt, df.mean_app)

    top_df = first(df, min(10, nrow(df)))
    xspan = maximum(df.mean_mapt) - minimum(df.mean_mapt) + eps()
    yspan = maximum(df.mean_app) - minimum(df.mean_app) + eps()
    for row in eachrow(top_df)
        text!(
            ax,
            row.mean_mapt + 0.01 * xspan,
            row.mean_app + 0.01 * yspan;
            text = row.region,
            fontsize = 16,
            align = (:left, :bottom),
        )
    end

    text!(
        ax,
        0.02,
        0.98;
        space = :relative,
        text = @sprintf("Pearson r = %.3f\nN = %d regions", cor(df.mean_mapt, df.mean_app), nrow(df)),
        align = (:left, :top),
        fontsize = 22,
    )
    return fig
end


function main()
    mkpath(OUTDIR)
    PathoSpread.setup_plot_theme!(font="Arial", base=22, lw=3, markersize=14, dpi=300)

    for family in ("beta", "gamma")
        df = parameter_df(family)
        CSV.write(joinpath(OUTDIR, family * "_app_vs_mapt_summary.csv"), df)
        fig = scatter_figure(df, family)
        save(joinpath(OUTDIR, family * "_app_vs_mapt_scatter.pdf"), fig)
        save(joinpath(OUTDIR, family * "_app_vs_mapt_scatter.png"), fig)
    end

    println("Saved parameter scatter figures to $OUTDIR")
end


main()
