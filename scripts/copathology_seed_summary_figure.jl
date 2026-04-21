#!/usr/bin/env julia

using CairoMakie
using CSV
using DataFrames
using Statistics
using Printf

const OUTPUT_DIR = "figures/copathology_seed_summary"
const MODES = ["all", "updated", "updated_growth"]
const MODE_LABELS = ["All", "Updated", "Growth"]

const PARAM_DIRS = Dict(
    ("seeded", "all") => "figures/copathology",
    ("seeded", "updated") => "figures/copathology_updated",
    ("seeded", "updated_growth") => "figures/copathology_updated_growth",
    ("noseed", "all") => "figures/copathology_noseed",
    ("noseed", "updated") => "figures/copathology_updated_noseed",
    ("noseed", "updated_growth") => "figures/copathology_updated_growth_noseed",
)

const GENE_DIRS = Dict(
    ("seeded", "all") => "figures/copathology/gene_axis",
    ("seeded", "updated") => "figures/copathology_updated/gene_axis",
    ("seeded", "updated_growth") => "figures/copathology_updated_growth/gene_axis",
    ("noseed", "all") => "figures/copathology_noseed/gene_axis",
    ("noseed", "updated") => "figures/copathology_updated_noseed/gene_axis",
    ("noseed", "updated_growth") => "figures/copathology_updated_growth_noseed/gene_axis",
)

const AMYLOID_DIRS = Dict(
    ("seeded", "all") => "figures/copathology/amyloid_beta/csv/abeta_syn_app_association_stats.csv",
    ("seeded", "updated") => "figures/copathology_updated/amyloid_beta/csv/abeta_syn_app_association_stats_updated.csv",
    ("seeded", "updated_growth") => "figures/copathology_updated_growth/amyloid_beta/csv/abeta_syn_app_association_stats_updated_growth.csv",
    ("noseed", "all") => "figures/copathology_noseed/amyloid_beta/csv/abeta_syn_app_association_stats_all.csv",
    ("noseed", "updated") => "figures/copathology_updated_noseed/amyloid_beta/csv/abeta_syn_app_association_stats_updated.csv",
    ("noseed", "updated_growth") => "figures/copathology_updated_growth_noseed/amyloid_beta/csv/abeta_syn_app_association_stats_updated_growth.csv",
)

const COLORS = Dict("seeded" => colorant"#8c2d04", "noseed" => colorant"#1d4f91")
const PARAM_MARKERS = Dict("beta" => :circle, "gamma" => :rect)
const STYLE_LABELS = Dict("seeded" => "With seeding sites", "noseed" => "Without seeding sites")
const PARAM_LABELS = Dict("beta" => "beta", "gamma" => "gamma")


function family_corr(path::AbstractString)
    df = CSV.read(path, DataFrame)
    return cor(df.mean_mapt, df.mean_app), nrow(df)
end


function angle_difference(path::AbstractString)
    for line in eachline(path)
        if occursin("PC1 angle difference", line)
            m = match(r"=\s*([0-9eE+\-\.]+)", line)
            m === nothing && error("Could not parse angle from $path")
            return parse(Float64, m.captures[1])
        end
    end
    error("PC1 angle difference not found in $path")
end


function amyloid_metric(path::AbstractString, parameter::AbstractString)
    df = CSV.read(path, DataFrame)
    normalized_parameter = replace(parameter, "beta" => "β", "gamma" => "γ")
    row = filter(r -> r.abeta_measure == "log10(1+Ab42)" && r.parameter == normalized_parameter, eachrow(df))
    isempty(row) && error("Could not find Ab42 / $parameter in $path")
    r = row[1]
    return Float64(r.pearson_r), Int(r.n), Float64(r.pearson_q_fdr)
end


function significance_tag(q::Float64)
    if q < 0.001
        return "***"
    elseif q < 0.01
        return "**"
    elseif q < 0.05
        return "*"
    end
    return ""
end


function line_panel!(ax::Axis, xs, ys_seeded, ys_noseed; ylabel::AbstractString, title::AbstractString)
    for (kind, ys) in [("seeded", ys_seeded), ("noseed", ys_noseed)]
        lines!(ax, xs, ys; color=COLORS[kind], linewidth=4)
        scatter!(ax, xs, ys; color=COLORS[kind], markersize=16)
    end
    ax.xticks = (xs, MODE_LABELS)
    ax.ylabel = ylabel
    ax.title = title
    ax.xticklabelrotation = pi / 8
    return nothing
end


function annotate_significance!(ax::Axis, xs, ys, qs, kind::AbstractString)
    for i in eachindex(xs)
        tag = significance_tag(qs[i])
        isempty(tag) && continue
        yoff = kind == "seeded" ? 0.03 : -0.05
        text!(ax, xs[i], ys[i] + yoff; text=tag, align=(:center, :center), fontsize=18, color=COLORS[kind])
    end
    return nothing
end


function main()
    mkpath(OUTPUT_DIR)
    set_theme!(Theme(font="Arial", fontsize=18, linewidth=3))

    xs = 1:3

    beta_corr_seeded = Float64[]
    beta_corr_noseed = Float64[]
    gamma_corr_seeded = Float64[]
    gamma_corr_noseed = Float64[]
    beta_n_seeded = Int[]
    beta_n_noseed = Int[]
    gamma_n_seeded = Int[]
    gamma_n_noseed = Int[]
    angle_seeded = Float64[]
    angle_noseed = Float64[]
    abeta_beta_seeded = Float64[]
    abeta_beta_noseed = Float64[]
    abeta_gamma_seeded = Float64[]
    abeta_gamma_noseed = Float64[]
    abeta_beta_q_seeded = Float64[]
    abeta_beta_q_noseed = Float64[]
    abeta_gamma_q_seeded = Float64[]
    abeta_gamma_q_noseed = Float64[]

    for mode in MODES
        beta_r, beta_n = family_corr(joinpath(PARAM_DIRS[("seeded", mode)], "beta_app_minus_mapt_summary.csv"))
        gamma_r, gamma_n = family_corr(joinpath(PARAM_DIRS[("seeded", mode)], "gamma_app_minus_mapt_summary.csv"))
        push!(beta_corr_seeded, beta_r)
        push!(gamma_corr_seeded, gamma_r)
        push!(beta_n_seeded, beta_n)
        push!(gamma_n_seeded, gamma_n)

        beta_r, beta_n = family_corr(joinpath(PARAM_DIRS[("noseed", mode)], "beta_app_minus_mapt_summary.csv"))
        gamma_r, gamma_n = family_corr(joinpath(PARAM_DIRS[("noseed", mode)], "gamma_app_minus_mapt_summary.csv"))
        push!(beta_corr_noseed, beta_r)
        push!(gamma_corr_noseed, gamma_r)
        push!(beta_n_noseed, beta_n)
        push!(gamma_n_noseed, gamma_n)

        push!(angle_seeded, angle_difference(joinpath(GENE_DIRS[("seeded", mode)], "csv", "pc1_axis_comparison_summary.txt")))
        push!(angle_noseed, angle_difference(joinpath(GENE_DIRS[("noseed", mode)], "csv", "pc1_axis_comparison_summary.txt")))

        beta_r, _, beta_q = amyloid_metric(AMYLOID_DIRS[("seeded", mode)], "APP syn beta")
        gamma_r, _, gamma_q = amyloid_metric(AMYLOID_DIRS[("seeded", mode)], "APP syn gamma")
        push!(abeta_beta_seeded, beta_r)
        push!(abeta_gamma_seeded, gamma_r)
        push!(abeta_beta_q_seeded, beta_q)
        push!(abeta_gamma_q_seeded, gamma_q)

        beta_r, _, beta_q = amyloid_metric(AMYLOID_DIRS[("noseed", mode)], "APP syn beta")
        gamma_r, _, gamma_q = amyloid_metric(AMYLOID_DIRS[("noseed", mode)], "APP syn gamma")
        push!(abeta_beta_noseed, beta_r)
        push!(abeta_gamma_noseed, gamma_r)
        push!(abeta_beta_q_noseed, beta_q)
        push!(abeta_gamma_q_noseed, gamma_q)
    end

    fig = Figure(size=(1800, 1200))

    ax1 = Axis(fig[1, 1], title="APP vs MAPT beta similarity", ylabel="Pearson r")
    line_panel!(ax1, xs, beta_corr_seeded, beta_corr_noseed; ylabel="Pearson r", title="APP vs MAPT beta similarity")
    ylims!(ax1, 0, 1)

    ax2 = Axis(fig[1, 2], title="APP vs MAPT gamma similarity", ylabel="Pearson r")
    line_panel!(ax2, xs, gamma_corr_seeded, gamma_corr_noseed; ylabel="Pearson r", title="APP vs MAPT gamma similarity")
    ylims!(ax2, 0, 1)

    ax3 = Axis(fig[1, 3], title="Gene-axis rotation", ylabel="PC1 angle difference (deg)")
    line_panel!(ax3, xs, angle_seeded, angle_noseed; ylabel="Degrees", title="Gene-axis rotation")
    ylims!(ax3, 0, max(maximum(angle_seeded), maximum(angle_noseed)) * 1.15)

    ax4 = Axis(fig[2, 1], title="Ab42 vs APP beta", ylabel="Pearson r")
    line_panel!(ax4, xs, abeta_beta_seeded, abeta_beta_noseed; ylabel="Pearson r", title="Ab42 vs APP beta")
    annotate_significance!(ax4, xs, abeta_beta_seeded, abeta_beta_q_seeded, "seeded")
    annotate_significance!(ax4, xs, abeta_beta_noseed, abeta_beta_q_noseed, "noseed")
    ylims!(ax4, 0, 0.5)

    ax5 = Axis(fig[2, 2], title="Ab42 vs APP gamma", ylabel="Pearson r")
    line_panel!(ax5, xs, abeta_gamma_seeded, abeta_gamma_noseed; ylabel="Pearson r", title="Ab42 vs APP gamma")
    annotate_significance!(ax5, xs, abeta_gamma_seeded, abeta_gamma_q_seeded, "seeded")
    annotate_significance!(ax5, xs, abeta_gamma_noseed, abeta_gamma_q_noseed, "noseed")
    ylims!(ax5, 0, 0.5)

    ax6 = Axis(fig[2, 3], title="Retained regions", ylabel="N regions")
    lines!(ax6, xs, beta_n_seeded; color=COLORS["seeded"], linewidth=4, linestyle=:solid)
    scatter!(ax6, xs, beta_n_seeded; color=COLORS["seeded"], marker=:circle, markersize=16)
    lines!(ax6, xs, gamma_n_seeded; color=COLORS["seeded"], linewidth=4, linestyle=:dash)
    scatter!(ax6, xs, gamma_n_seeded; color=COLORS["seeded"], marker=:rect, markersize=16)
    lines!(ax6, xs, beta_n_noseed; color=COLORS["noseed"], linewidth=4, linestyle=:solid)
    scatter!(ax6, xs, beta_n_noseed; color=COLORS["noseed"], marker=:circle, markersize=16)
    lines!(ax6, xs, gamma_n_noseed; color=COLORS["noseed"], linewidth=4, linestyle=:dash)
    scatter!(ax6, xs, gamma_n_noseed; color=COLORS["noseed"], marker=:rect, markersize=16)
    ax6.xticks = (xs, MODE_LABELS)
    ax6.xticklabelrotation = pi / 8

    for (label, ax) in zip(["A", "B", "C", "D", "E", "F"], [ax1, ax2, ax3, ax4, ax5, ax6])
        text!(ax, 0.02, 0.98; space=:relative, text=label, align=(:left, :top), fontsize=28, font=:bold)
    end

    Legend(
        fig[0, 1:3],
        [
            LineElement(color=COLORS["seeded"], linewidth=4),
            LineElement(color=COLORS["noseed"], linewidth=4),
            MarkerElement(color=:black, marker=:circle, markersize=15),
            MarkerElement(color=:black, marker=:rect, markersize=15),
        ],
        ["With seeding sites", "Without seeding sites", "beta", "gamma"],
        orientation=:horizontal,
        framevisible=false,
        tellwidth=false,
    )

    Label(
        fig[3, 1:3],
        "Asterisks in panels D-E mark FDR-significant Ab42 associations (* q<0.05, ** q<0.01, *** q<0.001).",
        fontsize=18,
    )

    pdf_path = joinpath(OUTPUT_DIR, "copathology_seed_vs_noseed_summary.pdf")
    png_path = joinpath(OUTPUT_DIR, "copathology_seed_vs_noseed_summary.png")
    save(pdf_path, fig)
    save(png_path, fig)

    println("Saved summary figure:")
    println(pdf_path)
    println(png_path)
end


main()
