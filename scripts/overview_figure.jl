using CairoMakie

t = range(0.0, 1.0; length=400)
tt = collect(t)

# --- schematic curves ---
emp   = tt.^3 .* exp.(-8 .* tt);  emp   ./= maximum(emp)
diff  = 0.35 .* (1 .- exp.(-3 .* tt))
diffg = 0.95 ./ (1 .+ exp.(-10 .* (tt .- 0.35)))
diffga = tt.^3 .* exp.(-6.5 .* tt); diffga ./= maximum(diffga)

for (name, y) in [
    ("C_empirical", emp),
    ("C_diff",      diff),
    ("C_diffg",     diffg),
    ("C_diffga",    diffga),
]

    fig = Figure(size=(260,220))
    ax = Axis(fig[1,1];
        xticksvisible=false, yticksvisible=false,
        xticklabelsvisible=false, yticklabelsvisible=false,
        rightspinevisible=false, topspinevisible=false
    )

    if name == "C_empirical"
        n = length(t)
        # 8 interior timepoints (exclude endpoints)
        idx = round.(Int, range(0.05n, 0.88n, length=8))
        # Scatter = actual measured timepoints
        scatter!(ax, t[idx], y[idx]; markersize=16)
        # Dashed line should visually start at 0
        t_line = vcat(0.0, t[idx])
        y_line = vcat(0.0, y[idx])
        lines!(ax, t_line, y_line; linestyle=:dash, linewidth=2)
    else
        lines!(ax, t, y; linewidth=3)
    end

    xlims!(ax, 0, 1)
    ylims!(ax, 0, 1.05)

    save("figures/overview/" * name * ".pdf", fig)
    save("figures/overview/" * name * ".svg", fig)
end