using PathoSpread
using DifferentialEquations
using Statistics
using Base.Threads
using ProgressMeter
using CairoMakie

# create figure folder
outdir = joinpath("figures", "DIFF_sweep_log")
mkpath(outdir)
setup_plot_theme!()  # set plotting settings

# -------------------------------------------------------------
# SETTINGS
# -------------------------------------------------------------
use_logspace = false                 # toggle between linear vs log space comparison
selected_timepoints = nothing       # e.g. [1, 2, 3, 4] or nothing to use all

# -------------------------------------------------------------
# LOAD DATA AND STRUCTURAL MATRICES
# -------------------------------------------------------------
data_file = "data/total_path.csv"
w_file = "data/W_labeled_filtered.csv"

data, timepoints = process_pathology(data_file; W_csv=w_file)
Lr, N, labels = read_W(w_file, direction=:retro)

# Restrict to selected timepoints (if provided)
if selected_timepoints !== nothing
    data = data[:, selected_timepoints, :]
    timepoints = timepoints[selected_timepoints]
end

# -------------------------------------------------------------
# DEFINE ODE (DIFF model)
# -------------------------------------------------------------
function DIFF!(du, u, p, t; L)
    ρ = p[1]
    du .= -ρ .* (L * u)
end

# -------------------------------------------------------------
# R² FUNCTION
# -------------------------------------------------------------
function r2_score(y::AbstractVector, ŷ::AbstractVector)
    mask = .!ismissing.(y) .& .!ismissing.(ŷ)
    yv, ŷv = y[mask], ŷ[mask]
    if isempty(yv) || isempty(ŷv)
        return NaN
    end
    ss_res = sum((yv .- ŷv).^2)
    ss_tot = sum((yv .- mean(yv)).^2)
    return 1 - ss_res / ss_tot
end


# -------------------------------------------------------------
# OBSERVED DATA
# -------------------------------------------------------------
seed_idx = 74
data[seed_idx,:,:] .= missing
observed = Array(PathoSpread.mean3(data))  # N × T (after time selection)

# -------------------------------------------------------------
# PARAMETER GRID
# -------------------------------------------------------------
ρ_values   = range(1e-6, 1.0; length=50)
u0_values  = range(1e-6, 20.0; length=50)
tspan      = (0.0, maximum(timepoints))
results    = Matrix{Union{Float64,Missing}}(missing, length(ρ_values), length(u0_values))

# -------------------------------------------------------------
# PARALLEL SWEEP WITH PROGRESS BAR
# -------------------------------------------------------------
n_total = length(ρ_values) * length(u0_values)
progress = Progress(n_total; desc="Running parameter sweep...", showspeed=true)

@threads for iρ in eachindex(ρ_values)
    ρ = ρ_values[iρ]
    for iu0 in eachindex(u0_values)
        u0_seed = u0_values[iu0]
        u0 = zeros(N)
        u0[seed_idx] = u0_seed
        p = [ρ]

        prob = ODEProblem((du,u,p,t)->DIFF!(du,u,p,t;L=Lr), u0, tspan, p)
        sol = solve(prob, Tsit5(); reltol=1e-8, abstol=1e-10, saveat=timepoints)
        predicted = Array(sol[1:N, :])

        mask = .!ismissing.(observed)
        obs_vec = vec(observed[mask])
        pred_vec = vec(predicted[mask])

        # --- logspace toggle ---
        if use_logspace
            valid = (obs_vec .> 0) .& (pred_vec .> 0)
            obs_vec = log.(obs_vec[valid])
            pred_vec = log.(pred_vec[valid])
        end

        results[iρ, iu0] = r2_score(obs_vec, pred_vec)
        next!(progress)
    end
end

# -------------------------------------------------------------
# FIND BEST COMBINATION
# -------------------------------------------------------------
best_idx = argmax(results)
best_r2 = results[best_idx]
best_ρ = ρ_values[CartesianIndices(results)[best_idx][1]]
best_u0 = u0_values[CartesianIndices(results)[best_idx][2]]

println("\n✅ Best R² = $(round(best_r2, digits=4)) at ρ=$(best_ρ), u0=$(best_u0)")

# -------------------------------------------------------------
# RE-RUN BEST MODEL FOR PLOTTING
# -------------------------------------------------------------
u0 = zeros(N)
u0[seed_idx] = best_u0
p = [best_ρ]

prob = ODEProblem((du,u,p,t)->DIFF!(du,u,p,t;L=Lr), u0, tspan, p)
sol = solve(prob, Tsit5(); reltol=1e-8, abstol=1e-10, saveat=timepoints)
predicted = Array(sol[1:N, :])

# -------------------------------------------------------------
# PLOT: PREDICTED VS OBSERVED
# -------------------------------------------------------------
mask = .!ismissing.(observed)
x = vec(observed[mask])
y = vec(predicted[mask])

if use_logspace
    valid = (x .> 0) .& (y .> 0)
    x, y = log.(x[valid]), log.(y[valid])
end

if isempty(x) || isempty(y)
    @warn "No valid data points after filtering (check selected timepoints or log transform)."
else
    fig = Figure(size=(500, 400))
    ax = Axis(fig[1, 1],
              xlabel = use_logspace ? "log(Observed)" : "Observed",
              ylabel = use_logspace ? "log(Predicted)" : "Predicted",
              title = "")

    scatter!(ax, x, y, markersize=12, alpha=0.4)

    xmin, xmax = extrema(x)
    lines!(ax, [xmin, xmax], [xmin, xmax], color=:red, linewidth=2)

    r2_val = r2_score(x, y)
    text!(ax, "R² = $(round(r2_val, digits=3))", position=(xmin, maximum(y)), align=(:left, :top))

    fig  # shows interactively
    fname = "DIFF_parsweep_all.png"
    save(joinpath(outdir, fname), fig)
    
end

# -------------------------------------------------------------
# SAVE: PREDICTED VS OBSERVED (ONE PNG PER TIMEPOINT)
# -------------------------------------------------------------
for (k, t) in enumerate(timepoints)
    xk = observed[:, k]
    yk = predicted[:, k]

    maskk = .!ismissing.(xk)
    x = Vector{Float64}(xk[maskk])
    y = Vector{Float64}(yk[maskk])

    if use_logspace
        valid = (x .> 0) .& (y .> 0)
        x = log.(x[valid])
        y = log.(y[valid])
    end

    fig = Figure(size=(500, 400))
    ax = Axis(fig[1, 1],
              xlabel = use_logspace ? "log(Observed)" : "Observed",
              ylabel = use_logspace ? "log(Predicted)" : "Predicted",
              title  = "")

    if isempty(x) || isempty(y)
        text!(ax, "No valid points", position=(0, 0))
    else
        scatter!(ax, x, y, alpha=0.6)
        xmin, xmax = extrema(x)
        lines!(ax, [xmin, xmax], [xmin, xmax], color=:red, linewidth=2)

        r2k = r2_score(x, y)
        text!(ax, "R² = $(round(r2k, digits=3))",
              position=(xmin, maximum(y)),
              align=(:left, :top))
    end

    # make filenames stable + sortable
    t_str = replace(string(t), "." => "p")
    fname = "DIFF_parsweep_t$(k).png"
    save(joinpath(outdir, fname), fig)
end

# -------------------------------------------------------------
# SAVE: REGION-WISE TRAJECTORIES (OBSERVED vs PREDICTED)
# -------------------------------------------------------------
traj_dir = joinpath(outdir, "region_trajectories")
mkpath(traj_dir)

T = length(timepoints)

for i in 1:N
    # observed is Float64 with missing possible
    obs_i = observed[i, :]              # Vector{Union{Missing,Float64}} or similar
    pred_i = vec(predicted[i, :])       # Vector{Float64}

    # mask missings in observed
    mask = .!ismissing.(obs_i)

    t = timepoints[mask]
    x = Vector{Float64}(obs_i[mask])
    y = pred_i[mask]

    # optional log transform (consistent with your scatter logic)
    if use_logspace
        valid = (x .> 0) .& (y .> 0)
        t = t[valid]
        x = log.(x[valid])
        y = log.(y[valid])
    end

    fig = Figure(size=(700, 300))
    ax = Axis(fig[1, 1],
              xlabel = "Time",
              ylabel = use_logspace ? "log(pathology)" : "Pathology",
              title  = "")

    if isempty(t)
        text!(ax, "No valid points", position=(0, 0))
    else
        lines!(ax, t, x; linewidth=2, label="Observed")
        lines!(ax, t, y; linewidth=2, linestyle=:dash, label="Predicted")
        axislegend(ax; position=:rb)
    end

    # filename: keep it sortable and filesystem-safe
    lab = replace(string(labels[i]), r"[^A-Za-z0-9_\-]+" => "_")
    fname = "region_$(lpad(string(i), 3, '0'))_$(lab).png"
    save(joinpath(traj_dir, fname), fig)
end
