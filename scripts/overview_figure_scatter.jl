using CairoMakie
using Random
using Statistics

Random.seed!(1)

outdir = "figures/overview/"

# muted colors
point_color = RGBf(0.15, 0.25, 0.55)   # deep muted blue
line_color  = RGBf(0.65, 0.20, 0.20)   # muted dark red

function make_scatter(name; slope=0.8, n=60, noise=0.18, markersize=14)

    x = rand(n)
    y = 0.5 .+ slope .* (x .- 0.5) .+ noise .* randn(n)
    y = clamp.(y, 0.0, 1.0)

    # linear fit
    X = hcat(ones(n), x)
    β = X \ y
    xx = range(0, 1; length=200)
    yy = β[1] .+ β[2] .* xx

    fig = Figure(size=(260,220))
    ax = Axis(fig[1,1];
        xticksvisible=false, yticksvisible=false,
        xticklabelsvisible=false, yticklabelsvisible=false,
        rightspinevisible=false, topspinevisible=false
    )

    scatter!(ax, x, y;
        markersize=markersize,
        color=point_color
    )

    lines!(ax, xx, yy;
        linewidth=3,
        color=line_color
    )

    xlims!(ax, 0, 1)
    ylims!(ax, 0, 1)

    save(outdir * name * ".pdf", fig)
    save(outdir * name * ".svg", fig)
end

make_scatter("D_scatter_pos"; slope=0.9)
make_scatter("D_scatter_neg"; slope=-0.9)