#!/usr/bin/env julia

using PathoSpread
using CSV
using DataFrames
using Statistics
using CairoMakie
using HypothesisTests

const APP_PATH = get(ENV, "COPATHOLOGY_APP_PATH", "simulations/syn_app_DIFFGA_RETRO.jls")
const ABETA40_PATH = "data/syn_tau_abeta/ab40_pathology_mpff.csv"
const ABETA42_PATH = "data/syn_tau_abeta/ab42_pathology_mpff.csv"
const OUTROOT = get(ENV, "COPATHOLOGY_OUTROOT", "figures/copathology")
const OUTDIR = joinpath(OUTROOT, "amyloid_beta")
const CSV_DIR = joinpath(OUTDIR, "csv")


function region_means_from_wide(path::AbstractString)
    df = CSV.read(path, DataFrame)
    means = Dict{String, Float64}()
    for c in names(df)[3:end]
        vals = collect(skipmissing(df[!, c]))
        isempty(vals) && continue
        eltype(vals) <: Number || continue
        means[String(c)] = mean(Float64.(vals))
    end
    return means
end


function app_parameter_df()
    inf = load_inference(APP_PATH)
    labels = String.(inf["labels"])
    beta = PathoSpread.posterior_mean_vector_from_priors(inf["chain"], inf["priors"], "beta", length(labels))
    gamma = PathoSpread.posterior_mean_vector_from_priors(inf["chain"], inf["priors"], "gamma", length(labels))
    return DataFrame(region=labels, beta_app=beta, gamma_app=gamma)
end


function join_abeta(df_params::DataFrame)
    ab40_mean = region_means_from_wide(ABETA40_PATH)
    ab42_mean = region_means_from_wide(ABETA42_PATH)
    df = copy(df_params)
    df.ab40_log10p1 = [log10(1 + get(ab40_mean, r, NaN)) for r in df.region]
    df.ab42_log10p1 = [log10(1 + get(ab42_mean, r, NaN)) for r in df.region]
    return df
end


function scatter_panel(df::DataFrame, xcol::Symbol, ycol::Symbol, title::String, xlabel::String, ylabel::String)
    mask = .!isnan.(df[!, xcol]) .& .!isnan.(df[!, ycol])
    x = df[mask, xcol]
    y = df[mask, ycol]
    fig = Figure(size=(1000, 800))
    ax = Axis(fig[1, 1], title=title, xlabel=xlabel, ylabel=ylabel,
        titlesize=30, xlabelsize=24, ylabelsize=24, xticklabelsize=18, yticklabelsize=18)
    scatter!(ax, x, y; color=(:black, 0.65), markersize=16)
    if length(x) >= 3
        r = cor(x, y)
        p = pvalue(CorrelationTest(x, y))
        X = hcat(ones(length(x)), x)
        β = X \ y
        xs = collect(range(minimum(x), maximum(x), length=200))
        lines!(ax, xs, β[1] .+ β[2] .* xs; color=:firebrick3, linewidth=4)
        text!(ax, 0.02, 0.98; space=:relative,
            text="Pearson r = " * string(round(r, sigdigits=3)) * "\np = " * string(round(p, sigdigits=3)) * "\nN = " * string(length(x)),
            align=(:left, :top), fontsize=18)
    end
    return fig, mask
end


function main()
    mkpath(OUTDIR)
    mkpath(CSV_DIR)
    PathoSpread.setup_plot_theme!(font="Arial", base=20, lw=3, markersize=12, dpi=300)

    df = join_abeta(app_parameter_df())
    CSV.write(joinpath(CSV_DIR, "regional_abeta_parameter_table.csv"), df)

    fig, _ = scatter_panel(df, :ab40_log10p1, :beta_app, "Ab40 vs APP beta", "log10(1 + Ab40 burden)", "APP rise parameter beta")
    save(joinpath(OUTDIR, "ab40_vs_beta.pdf"), fig)
    fig, _ = scatter_panel(df, :ab40_log10p1, :gamma_app, "Ab40 vs APP gamma", "log10(1 + Ab40 burden)", "APP fall parameter gamma")
    save(joinpath(OUTDIR, "ab40_vs_gamma.pdf"), fig)
    fig, _ = scatter_panel(df, :ab42_log10p1, :beta_app, "Ab42 vs APP beta", "log10(1 + Ab42 burden)", "APP rise parameter beta")
    save(joinpath(OUTDIR, "ab42_vs_beta.pdf"), fig)
    fig, _ = scatter_panel(df, :ab42_log10p1, :gamma_app, "Ab42 vs APP gamma", "log10(1 + Ab42 burden)", "APP fall parameter gamma")
    save(joinpath(OUTDIR, "ab42_vs_gamma.pdf"), fig)

    println("Saved amyloid figures to $OUTDIR")
end


main()
