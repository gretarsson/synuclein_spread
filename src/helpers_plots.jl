# helpers_plotting.jl
# Clean, readable plotting helpers for Bayesian ODE inference results
# Keeps your original public API: predicted_observed, plot_retrodiction, plot_posteriors,
# plot_priors, plot_prior_and_posterior, plot_chains, plot_inference.

using CairoMakie
using Makie
using StatsPlots
using Statistics
using StatsBase
using Distributions
using MathTeXEngine  # for LaTeXStrings L"..." font control (optional)
using Plots
using Printf
using Graphs
using SimpleWeightedGraphs



############################
# Global plotting theme
############################
"""
    setup_plot_theme!(; font="TeX Gyre Heros", base=18, lw=2, markersize=10, dpi=300)

Set a global plotting theme for both Makie (CairoMakie) and StatsPlots/Plots.
Call this once (e.g., at the top of your script) to standardize fonts & sizes.
"""

function setup_plot_theme!(; font="Arial", base=18, lw=2, markersize=10, dpi=300)
    set_theme!(Theme(
        # ensure every text uses the same family
        fonts = (regular=font, bold=font, italic=font),
        font = font,
        fontsize = base,
        Figure = (resolution = (1200, 900),),
        Axis = (
            titlesize      = round(Int, 1.6*base),
            titlealign     = :left,
            xlabelsize     = round(Int, 1.5*base),
            ylabelsize     = round(Int, 1.5*base),
            xticklabelsize = round(Int, 1.4*base),
            yticklabelsize = round(Int, 1.4*base),
            xgridvisible   = false,
            ygridvisible   = false,
        ),
        Legend   = (labelsize = round(Int, 1.0*base), titlesize = round(Int, 1.2*base)),
        Colorbar = (labelsize = round(Int, 1.0*base), ticklabelsize = round(Int, 1.0*base)),
        Scatter  = (markersize = round(Int, 0.8*base),),
        Lines    = (linewidth  = round(Int, 0.5*base),)
    ))

    # StatsPlots / Plots defaults
    Plots.default(
        fontfamily = font,
        guidefont  = Plots.font(round(Int, 1.2*base)),
        tickfont   = Plots.font(round(Int, 1.0*base)),
        legendfont = Plots.font(round(Int, 1.0*base)),
        titlefont  = Plots.font(round(Int, 1.6*base)),
        linewidth  = lw,
        markersize = markersize,
        dpi        = dpi,
    )
    return nothing
end


setup_plot_theme!()
# Alternative fonts you may like: "CMU Serif", "TeX Gyre Termes", "TeX Gyre Pagella", "Helvetica", "Arial"

function predicted_observed(inference; save_path="", plotscale=log10)
    # try creating folder to save in
    try
        mkdir(save_path) 
    catch
    end

    # unpack from simulation
    fs = []  # storing figures
    chain = inference["chain"]
    data = inference["data"]
    timepoints = inference["timepoints"]
    seed = inference["seed_idx"]
    Ltuple = inference["L"]
    priors = inference["priors"]
    sol_idxs = inference["sol_idxs"]
    labels = inference["labels"]

    ks = collect(keys(priors))
    N_pars = findall(x->x=="σ",ks)[1] - 1
    factors = [1. for _ in 1:N_pars]
    ode = odes[inference["ode"]]
    N = size(data)[1]

    # simulate ODE from posterior mode
    # initialize
    tspan = (0., timepoints[end])
    u0 = inference["u0"]
    #rhs(du,u,p,t;L=L, func=ode::Function) = func(du,u,p,t;L=L,factors=factors)
    #prob = ODEProblem(rhs, u0, tspan; alg=Tsit5())

    prob = make_ode_problem(ode;
        labels     = labels,
        Ltuple     = Ltuple,
        factors    = factors,
        u0         = u0,
        timepoints = timepoints,
    )

    # find posterior mode
    n_pars = length(chain.info[1])
    _, argmax = findmax(chain[:lp])
    mode_pars = Array(chain[argmax[1], 1:n_pars, argmax[2]])
    p = mode_pars[1:N_pars]
    if inference["bayesian_seed"]
        u0[seed] = chain["seed"][argmax]  
    else
        u0[seed] = inference["seed_value"]  
    end


    # solve ODE
    sol = solve(prob,Tsit5(); p=p, u0=u0, saveat=timepoints, abstol=1e-9, reltol=1e-6)
    sol = Array(sol[sol_idxs,:])

    # plot
    #xticks = ([1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1e-0], [L"$10^{-6}$", L"$10^{-5}$", L"$10^{-4}$", L"$10^{-3}$", L"$10^{-2}$", L"$10^{-1}$", L"$10^0$"])
    #yticks = xticks

    # ticks: predefined for log10, automatic otherwise
    use_log = (plotscale === log10)
    predefined_ticks = ([1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1e0],
                        [L"$10^{-6}$", L"$10^{-5}$", L"$10^{-4}$", L"$10^{-3}$", L"$10^{-2}$", L"$10^{-1}$", L"$10^0$"])
    xticks = use_log ? predefined_ticks : Makie.automatic
    yticks = xticks

    f = CairoMakie.Figure()
    ax = CairoMakie.Axis(f[1,1], title="", ylabel="Predicted", xlabel="Observed", xscale=plotscale, yscale=plotscale, xticks=xticks, yticks=yticks)

    # as we are plotting log-log, we account for zeros in the data
    regions = 1:N
    if length(size(data)) > 2
        data = mean3(data)  # find mean of data 
        # remove regions with less than 3 measuremnts 
        region_idxs = measured_regions(data,3)
        data = data[region_idxs,:,:]
        regions = copy(region_idxs)
    end

    # set x and y ticks
    x = vec(copy(data[:,2:end]))  # skip first time point (model is trivially y(0)=0)
    y = vec(copy(sol[regions,2:end]))
    nonmissing = findall(x .!== missing)
    x = x[nonmissing]
    y = y[nonmissing]
    minxy = min(minimum(x),minimum(y))
    if plotscale==log10 && ((sum(x .== 0) + sum(y .== 0)) > 0)  # if zeros present, add the smallest number in plot
        #minx = minimum(x[x.>0])  # change back to this if plots are weird also see below if statemetn
        #miny = minimum(y[y.>0])
        #minxy = min(minx, miny)
        minxy = minimum(x[x.>0])  # change minimum to minimum of data to avoid super low value i.e e-44 from sims
        x = x .+ minxy
        y = y .+ minxy
    end

    CairoMakie.scatter!(ax,x,y, alpha=0.5)
    maxxy = max(maximum(x), maximum(y))
    CairoMakie.lines!([minxy,maxxy],[minxy,maxxy], color=:grey, alpha=0.5)
    if !isempty(save_path)
        CairoMakie.save(save_path * "/predicted_observed_mode.pdf", f)
    end
    push!(fs,f)

    # plot at different time points
    for i in eachindex(timepoints)
        f = CairoMakie.Figure()
        ax = CairoMakie.Axis(f[1,1], title="t = $(timepoints[i])", ylabel="Predicted", xlabel="Observed", xscale=plotscale, yscale=plotscale, xticks=xticks, yticks=yticks)

        # as we are plotting log-log, we account for zeros in the data
        x = vec(copy(data[:,i]))
        y = vec(copy(sol[regions,i]))
        nonmissing = findall(x .!== missing)
        x = x[nonmissing]
        y = y[nonmissing]
        #if plotscale==log10 && ((sum(x .<= 0) + sum(y .<= 0)) > 0)  # if doesn't work change back to this
        if plotscale==log10 && ((sum(x .<= 1e-8) + sum(y .<= 1e-8)) > 0)  # if zeros (or very small) present, add the smallest number in plot
            x = x .+ minxy
            y = y .+ minxy
        end

        CairoMakie.scatter!(ax,x,y, alpha=0.5)
        CairoMakie.lines!([minxy,maxxy],[minxy,maxxy], color=:grey, alpha=0.5)
        if !isempty(save_path)
            CairoMakie.save(save_path * "/predicted_observed_mode_$(i).pdf", f)
        end
        push!(fs,f)
    end

    return fs
end



#=
plot chains of each parameter from inference
 =#
function plot_chains(inference; save_path="")
    # try creating folder to save in
    try
        mkdir(save_path) 
    catch
    end
    chain = inference["chain"]
    vars = collect(keys(inference["priors"]))
    master_fig = StatsPlots.plot(chain) 
    chain_figs = []
    for (i,var) in enumerate(vars)
        chain_i = StatsPlots.plot(master_fig[i,1], title=var)
        if !isempty(save_path)
            savefig(chain_i, save_path*"/chain_$(var).png")
        end
        push!(chain_figs,chain_i)
    end
    return chain_figs
end

#=
plot posteriors of each parameter from inference
 =#
function plot_posteriors(inference; save_path="")
    # try creating folder to save in
    try
        mkdir(save_path) 
    catch
    end
    chain = inference["chain"]
    vars = collect(keys(inference["priors"]))
    master_fig = StatsPlots.plot(chain) 
    #posterior_figs = []
    for (i,var) in enumerate(vars)
        posterior_i = StatsPlots.plot(master_fig[i,2], title=var)
        if !isempty(save_path)
            savefig(posterior_i, save_path*"/posterior_$(var).png")
        end
        StatsPlots.closeall()
        #push!(posterior_figs,posterior_i)
    end
    #return posterior_figs
    return nothing
end

"""
    plot_retrodiction(inference; save_path=nothing, N_samples=200,
                               show_band=true, band=(0.25, 0.75),
                               data_error=:none)  # :none | :sd | :se

Plots per-region panels with:
  • Model: posterior median trajectory (solid line) and, optionally, a 50% credible band.
  • Data: per-timepoint MEAN only (optionally with ±SD or ±SE error bars).

This avoids spaghetti and raw replicate clutter.

Arguments
- N_samples::Int: number of posterior draws used to summarize trajectories.
- show_band::Bool: draw a credible band between the given quantiles (default 50%).
- band::Tuple: (low, high) quantiles for the band, e.g., (0.25, 0.75).
- data_error::Symbol: :none (default), :sd, or :se to add error bars around the mean.
"""
function plot_retrodiction(inference; save_path=nothing, N_samples=200,
    show_band=true, band=(0.25, 0.75),
    data_style::Symbol=:mean, data_error::Symbol=:none,
    level::Union{Nothing,Float64}=nothing,      # e.g., 0.50 or 0.95; if nothing, use `band`
    interval::Symbol=:process,                  # :process (traj only) or :predictive (traj + Normal noise)
    line_from::Symbol=:process,                 # :process or :predictive median line
    truncate_at_zero::Bool=true,                # clamp band/line & data bars at a floor
    ymin::Float64=0.0,                          # axis lower bound (visual)
    ymax::Union{Nothing,Float64}=nothing,       # fixed common y-max (overrides sync_y)
    sync_y::Bool=false,                         # auto-compute a common y-max from observed data
    y_padding::Float64=0.05,                    # headroom when sync_y is true (top only)
    clamp_floor::Union{Nothing,Float64}=0.0,    # floor for BOTH band/line and data error bars (physical floor)
    lower_headroom_frac::Float64=0.03,           # small visual headroom below clamp_floor
    save_legend::Bool=true,
    legend_filename::String="aretrodiction_legend.pdf"
)

    # --- Unpack ---
    data        = inference["data"]
    chain       = inference["chain"]
    timepoints  = inference["timepoints"]
    seed        = inference["seed_idx"]
    sol_idxs    = inference["sol_idxs"]
    Ltuple      = inference["L"]
    labels      = inference["labels"]
    ks          = collect(keys(inference["priors"]))
    N_pars      = findall(x->x=="σ", ks)[1] - 1
    factors     = [1.0 for _ in 1:N_pars]
    ode         = odes[inference["ode"]]
    N           = size(data, 1)

    # Data summaries
    if ndims(data) == 3
        mean_data = mean3(data)             # (region,time)
        var_data  = var3(data)              # (region,time)
        n_rep     = size(data, 3)
    elseif ndims(data) == 2
        mean_data = data
        var_data  = fill!(similar(data), 0.0)
        n_rep     = 1
    else
        error("Unsupported data array with ndims=$(ndims(data)).")
    end

    # Time grid
    tgrid = collect(range(0, stop=timepoints[end], step=0.1))

    # ODE problem (we pass u0 per-draw to solve)
    prob = make_ode_problem(ode;
        labels     = labels,
        Ltuple     = Ltuple,
        factors    = factors,
        u0         = inference["u0"],
        timepoints = tgrid,
    )

    # Figures & axes
    fs  = Vector{Any}(undef, N)
    axs = Vector{Any}(undef, N)
    for i in 1:N
        f  = CairoMakie.Figure()
        ax = CairoMakie.Axis(
            f[1,1];
            title  = "Region $(i): $(labels[i])",
            xlabel = "Time (months)",
            ylabel = "\u03b1-synuclein pathology (% area)",
            limits = truncate_at_zero ? (nothing, nothing, ymin, nothing) :
                                        (nothing, nothing, nothing, nothing)
        )
        fs[i]  = f
        axs[i] = ax
    end

    # Posterior trajectories on common grid
    posterior_samples = sample(chain, min(N_samples, length(chain[:lp])); replace=false)
    S = size(posterior_samples, 1)
    T = length(tgrid)
    traj = Array{Float64}(undef, N, T, S)

    # Parameter indices
    par_names   = chain.name_map.parameters
    sigma_ch_idx = findfirst(==(Symbol("σ")), par_names)
    sigma_ch_idx === nothing && error("Could not find :σ in chain.name_map.parameters")
    
    seed_ch_idx = nothing
    if get(inference, "bayesian_seed", false)
        seed_ch_idx = findfirst(==(Symbol("seed")), par_names)
        seed_ch_idx === nothing && error("Could not find :seed in chain.name_map.parameters")
    end
    

    # Zero template u0 (only seed nonzero per draw)
    u0_template = fill!(similar(inference["u0"]), 0.0)

    # Solve per-draw (fresh u0; per-draw seed)
    for (s, sample_vec) in enumerate(eachrow(Array(posterior_samples)))
        p    = sample_vec[1:N_pars]
        u0_s = copy(u0_template)
        if get(inference, "bayesian_seed", false)
            u0_s[seed] = sample_vec[seed_ch_idx]
        else
            u0_s[seed] = inference["seed_value"]
        end
        sol = solve(prob, Tsit5(); p=p, u0=u0_s, saveat=tgrid, abstol=1e-9, reltol=1e-6)
        traj[:, :, s] = Array(sol[sol_idxs, :])
    end

    # Quantiles
    lowq, highq = isnothing(level) ? band : ((1 - level)/2, 1 - (1 - level)/2)
    q_med_proc  = mapslices(x -> quantile(x, 0.5),   traj; dims=3)[:, :, 1]
    q_low_proc  = mapslices(x -> quantile(x, lowq),  traj; dims=3)[:, :, 1]
    q_high_proc = mapslices(x -> quantile(x, highq), traj; dims=3)[:, :, 1]

    # Predictive quantiles (simulate Normal noise per draw using that draw's σ)
    q_med_pred = q_low_pred = q_high_pred = nothing
    if interval == :predictive || line_from == :predictive
        ytraj = similar(traj)
        for (s, sample_vec) in enumerate(eachrow(Array(posterior_samples)))
            σ_s = sample_vec[sigma_ch_idx]
            ytraj[:, :, s] = traj[:, :, s] .+ (σ_s .* randn(N, T))
        end
        q_med_pred  = mapslices(x -> quantile(x, 0.5),   ytraj; dims=3)[:, :, 1]
        q_low_pred  = mapslices(x -> quantile(x, lowq),  ytraj; dims=3)[:, :, 1]
        q_high_pred = mapslices(x -> quantile(x, highq), ytraj; dims=3)[:, :, 1]
    end

    # Choose band/line sources
    if interval == :predictive
        q_low_draw, q_high_draw = q_low_pred, q_high_pred
    elseif interval == :process
        q_low_draw, q_high_draw = q_low_proc, q_high_proc
    else
        error("interval must be :process or :predictive")
    end
    q_med_draw = (line_from == :predictive && q_med_pred !== nothing) ? q_med_pred : q_med_proc

    # --- Clamp band/line & data bars to a common floor ---
    if truncate_at_zero
        floor = isnothing(clamp_floor) ? ymin : clamp_floor
        q_low_draw  = clamp.(q_low_draw,  floor, Inf)
        q_high_draw = clamp.(q_high_draw, floor, Inf)
        q_med_draw  = clamp.(q_med_draw,  floor, Inf)
    end

    # Common y-max from observed data only (if requested)
    global_ymax = ymax
    if isnothing(global_ymax) && sync_y
        dvals = collect(skipmissing(vec(data)))
        dvals = Float64.(filter(isfinite, dvals))
        global_ymax = isempty(dvals) ? ymin + 1.0 : maximum(dvals)
        global_ymax = max(global_ymax, ymin) * (1 + y_padding)
        if !isfinite(global_ymax) || !(global_ymax > ymin)
            global_ymax = ymin + 1.0
        end
    end

    # If we have a shared top, compute a shared lower headroom below clamp_floor
    shared_ymin = nothing
    if !isnothing(global_ymax)
        floor = isnothing(clamp_floor) ? ymin : clamp_floor
        shared_ymin = floor - lower_headroom_frac * max(global_ymax - floor, 1e-9)
    end

    # Plot per region
    for i in 1:N
        # Model band + line
        if show_band
            CairoMakie.band!(axs[i], tgrid, q_low_draw[i, :], q_high_draw[i, :]; color=(:gray, 0.25))
        end
        CairoMakie.lines!(axs[i], tgrid, q_med_draw[i, :]; color=:black, linewidth=5)

        # Data
        if data_style == :mean
            nonmissing = findall(mean_data[i, :] .!== missing)
            if !isempty(nonmissing)
                t_i = Float64.(timepoints[nonmissing])
                μ_i = Float64.(mean_data[i, :][nonmissing])
                CairoMakie.scatter!(axs[i], t_i, μ_i;
                    color=RGB(0/255,71/255,171/255),
                    markersize=round(Int, 1.2*18), alpha=0.95)

                if data_error != :none
                    v_i = Float64.(var_data[i, :][nonmissing])
                    v_i = max.(v_i, 0.0)
                    σ_i = sqrt.(v_i)
                    if data_error === :se && n_rep > 1
                        σ_i ./= sqrt(n_rep)
                    end
                    replace!(σ_i, NaN => 0.0)

                    if truncate_at_zero
                        floor = isnothing(clamp_floor) ? ymin : clamp_floor
                        lower_cap = max.(0.0, μ_i .- floor)   # how far down we can go
                        σ_low = map(min, σ_i, lower_cap)      # cap lower whisker
                        σ_up  = σ_i
                        CairoMakie.errorbars!(axs[i], t_i, μ_i, σ_low, σ_up;
                            color=RGB(0/255,71/255,171/255),
                            whiskerwidth=20, alpha=0.35, linewidth=5)
                    else
                        CairoMakie.errorbars!(axs[i], t_i, μ_i, σ_i;
                            color=RGB(0/255,71/255,171/255),
                            whiskerwidth=20, alpha=0.35, linewidth=3)
                    end
                end
            end

        elseif data_style == :all
            for k in axes(data, 3)
                nonmissing = findall(data[i, :, k] .!== missing)
                if !isempty(nonmissing)
                    t_i = Float64.(timepoints[nonmissing]) .+ randn(length(nonmissing)) .* 0.04
                    y_i = Float64.(data[i, :, k][nonmissing])
                    CairoMakie.scatter!(axs[i], t_i, y_i;
                        color=RGB(0/255,71/255,171/255),
                        markersize=round(Int, 0.4*18), alpha=0.35)
                end
            end
        else
            error("data_style must be :mean or :all")
        end

        # Apply y-limits (shared or per-panel with small lower headroom)
        if !isnothing(global_ymax) && isfinite(global_ymax)
            low = (shared_ymin isa Number && isfinite(shared_ymin)) ? shared_ymin : ymin
            Makie.ylims!(axs[i], low, global_ymax)
        else
            # Per-panel top from data (and ensure we don't crop band)
            dvals_i = ndims(data) == 3 ? vec(data[i, :, :]) : vec(data[i, :])
            dvals_i = Float64.(filter(isfinite, collect(skipmissing(dvals_i))))
            panel_data_max = isempty(dvals_i) ? ymin + 1.0 : maximum(dvals_i)
            panel_band_max = show_band ? maximum(q_high_draw[i, :]) : maximum(q_med_draw[i, :])
            panel_top = max(panel_data_max, panel_band_max)

            floor = isnothing(clamp_floor) ? ymin : clamp_floor
            panel_low = floor - lower_headroom_frac * max(panel_top - floor, 1e-9)

            if !(panel_top > panel_low) || !isfinite(panel_top)
                panel_top = floor + 1.0
            end
            Makie.ylims!(axs[i], panel_low, panel_top)
        end

        if save_path !== nothing
            try; mkdir(save_path); catch; end
            CairoMakie.save(joinpath(save_path, "retrodiction_region_$(i).pdf"), fs[i])
        end
    end

    # Link y-axes if we synced / fixed
    if !isnothing(global_ymax) || sync_y
        try
            Makie.linkyaxes!(axs...)
        catch
        end
    end

    # --- Standalone legend figure (optional) ---
    if save_legend
        # Colors/sizes consistent with the panels
        model_line_color   = :black
        model_line_width   = 5
        band_fill_color    = (:gray, 0.25)                       # 25% gray alpha
        data_color         = RGB(0/255, 71/255, 171/255)
        data_marker_size   = round(Int, data_style == :mean ? 1.2*18 : 0.4*18)

        # Build legend entries using explicit elements
        line_el  = CairoMakie.LineElement(color=model_line_color, linewidth=model_line_width)
        band_el  = CairoMakie.PolyElement(color=band_fill_color, strokecolor=:transparent)  # 95% CI patch
        mark_el  = CairoMakie.MarkerElement(marker=:circle, color=data_color, markersize=data_marker_size)

        leg_fig  = CairoMakie.Figure(resolution = (480, 200), figure_padding=20)
        legend   = CairoMakie.Legend(
            leg_fig,
            [line_el, band_el, mark_el],
            ["Model fit", "95% CI", "Experimental data"];
            orientation = :horizontal,
            framevisible = false,
            padding = (10,10,10,10),
            patchsize = (35,18)
        )
        leg_fig[1,1] = legend

        if save_path !== nothing
            try; mkdir(save_path); catch; end
            CairoMakie.save(joinpath(save_path, legend_filename), leg_fig)
        end
    end

    return fs
end

using CairoMakie
using Statistics
using StatsBase

"""
Quantile-binned calibration curve: mean observed vs mean predicted with 95% CI.

- interval: :process (trajectory only) or :predictive (trajectory + Normal noise)
- bins:     number of quantile bins over predicted values
- skip_first_timepoint: often useful if y(0)=0 trivially
- truncate_at_zero & clamp_floor: apply a common physical floor to plotted values

Saves a single PDF if save_path is provided.
"""

function plot_calibration(inference;
    save_path::Union{Nothing,String}=nothing,
    N_samples::Int=200,
    interval::Symbol=:process,          # :process or :predictive
    bins::Int=20,
    skip_first_timepoint::Bool=true,
    truncate_at_zero::Bool=true,
    clamp_floor::Union{Nothing,Float64}=0.0,
    binning::Symbol=:quantile,          # :quantile | :geometric | :adaptive
    min_bin_count::Int=100              # only used for :adaptive
)
    # --- Unpack ---
    data        = inference["data"]
    chain       = inference["chain"]
    timepoints  = inference["timepoints"]
    seed        = inference["seed_idx"]
    sol_idxs    = inference["sol_idxs"]
    Ltuple      = inference["L"]
    labels      = inference["labels"]
    ks          = collect(keys(inference["priors"]))
    N_pars      = findall(x->x=="σ", ks)[1] - 1
    factors     = [1.0 for _ in 1:N_pars]
    ode         = odes[inference["ode"]]
    N           = size(data, 1)

    # Data summaries (match your conventions)
    if ndims(data) == 3
        mean_data = mean3(data)             # (region,time)
        n_rep     = size(data, 3)
    elseif ndims(data) == 2
        mean_data = data
        n_rep     = 1
    else
        error("Unsupported data array with ndims=$(ndims(data)).")
    end

    # We calibrate at the observed timepoints (no dense grid)
    tgrid = timepoints

    # ODE problem (u0 set per draw below)
    prob = make_ode_problem(ode;
        labels     = labels,
        Ltuple     = Ltuple,
        factors    = factors,
        u0         = inference["u0"],
        timepoints = tgrid,
    )

    # Posterior draws
    posterior_samples = sample(chain, min(N_samples, length(chain[:lp])); replace=false)
    S = size(posterior_samples, 1)
    T = length(tgrid)
    traj = Array{Float64}(undef, N, T, S)

    # Parameter indices
    par_names     = chain.name_map.parameters
    sigma_ch_idx  = findfirst(==(Symbol("σ")), par_names)
    sigma_ch_idx === nothing && error("Could not find :σ in chain.name_map.parameters")
    seed_ch_idx = nothing
    if get(inference, "bayesian_seed", false)
        seed_ch_idx = findfirst(==(Symbol("seed")), par_names)
        seed_ch_idx === nothing && error("Could not find :seed in chain.name_map.parameters")
    end

    # Zero template u0 (seed per draw)
    u0_template = fill!(similar(inference["u0"]), 0.0)

    # Solve per draw on observed timepoints
    for (s, sample_vec) in enumerate(eachrow(Array(posterior_samples)))
        p    = sample_vec[1:N_pars]
        u0_s = copy(u0_template)
        if get(inference, "bayesian_seed", false)
            u0_s[seed] = sample_vec[seed_ch_idx]
        else
            u0_s[seed] = inference["seed_value"]
        end
        sol = solve(prob, Tsit5(); p=p, u0=u0_s, saveat=tgrid, abstol=1e-9, reltol=1e-6)
        traj[:, :, s] = Array(sol[sol_idxs, :])
    end

    # Choose process vs predictive
    arr = traj
    if interval == :predictive
        ytraj = similar(traj)
        for (s, sample_vec) in enumerate(eachrow(Array(posterior_samples)))
            σ_s = sample_vec[sigma_ch_idx]
            ytraj[:, :, s] = traj[:, :, s] .+ (σ_s .* randn(N, T))
        end
        arr = ytraj
    elseif interval != :process
        error("interval must be :process or :predictive")
    end

    # Predicted median across draws at each (region, time)
    pred_med = mapslices(x -> quantile(x, 0.5), arr; dims=3)[:, :, 1]   # (N,T)

    # Optionally skip the first timepoint (often trivial at t=0)
    first_col = skip_first_timepoint ? 2 : 1
    if first_col > size(pred_med, 2)
        error("No timepoints left after skipping the first.")
    end

    # Vectorize (region,time) pairs
    p_vec = vec(pred_med[:, first_col:end])     # predicted
    o_vec = vec(mean_data[:, first_col:end])    # observed summary

    # Joint missing/finite mask
    mask = .!ismissing.(o_vec) .& isfinite.(p_vec)
    p_vec = Float64.(p_vec[mask])
    o_vec = Float64.(o_vec[mask])

    if isempty(p_vec)
        error("No data available for calibration after filtering.")
    end

    # Optional truncation/clamping
    if truncate_at_zero
        floor = isnothing(clamp_floor) ? 0.0 : clamp_floor
        p_vec = clamp.(p_vec, floor, Inf)
        o_vec = clamp.(o_vec, floor, Inf)
    end

    # ----------------- Non-uniform bins (robust) -----------------
    bins = max(bins, 2)

    # normalize binning symbol to avoid :Geometric / :GEOMETRIC mismatches
    _binning = Symbol(lowercase(String(binning)))

    # helper: enforce strictly increasing edges
    _monotone!(edges::Vector{Float64}) = begin
        for i in 2:length(edges)
            if edges[i] <= edges[i-1]
                edges[i] = nextfloat(edges[i-1])
            end
        end
        edges
    end

    edges = Float64[]  # will be assigned below

    if _binning === :quantile
        edges = collect(Float64.(quantile(p_vec, range(0, 1; length=bins+1))))
        _monotone!(edges)

    elseif _binning === :geometric || _binning === :adaptive
        pmin, pmax = minimum(p_vec), maximum(p_vec)
        ε = 1e-12
        # start strictly >0 for geometric spacing; if all ~0, create a tiny span
        start = max(ε, min(pmin > 0 ? pmin : ε, pmax))
        stopv = max(pmax, start*10)  # ensure a span if pmax≈start
        ge = 10.0 .^ range(log10(start), log10(stopv); length=bins+1) |> collect
        if pmin == 0.0
            ge[1] = 0.0  # include zeros in the first bin
        end
        edges = _monotone!(Float64.(ge))

        if _binning === :adaptive
            # iteratively merge bins until each has >= min_bin_count points
            while true
                if length(edges) < 3
                    break  # need at least 2 bins
                end
                idx = [min(searchsortedlast(edges, v), length(edges)-1) for v in p_vec]
                counts = [count(==(b), idx) for b in 1:(length(edges)-1)]
                small = findall(<(min_bin_count), counts)  # bins with too few pts
                isempty(small) && break

                new_edges = Float64[edges[1]]
                b = 1
                while b <= length(counts)
                    if counts[b] < min_bin_count
                        if b < length(counts)
                            # merge bin b with b+1 (keep right edge of b+1)
                            push!(new_edges, edges[b+2])
                            b += 2
                        else
                            # last bin small: merge into previous
                            new_edges[end] = edges[end]
                            b += 1
                        end
                    else
                        push!(new_edges, edges[b+1])
                        b += 1
                    end
                end
                edges = _monotone!(new_edges)
            end
        end

    else
        error("binning must be :quantile, :geometric, or :adaptive (got $binning)")
    end

    (length(edges) >= 2) || error("Failed to build bin edges")

    # Assign bin index for each predicted value
    binidx = [min(searchsortedlast(edges, v), length(edges)-1) for v in p_vec]
    # -------------------------------------------------------------
    B = length(edges) - 1
    μ_pred = fill(NaN, B)
    μ_obs  = fill(NaN, B)
    lo95   = fill(NaN, B)
    hi95   = fill(NaN, B)
    m_in_bin = zeros(Int, B)

    for b in 1:B
        I = findall(==(b), binidx)
        m = length(I)
        m_in_bin[b] = m
        if m == 0
            continue
        end
        pv = p_vec[I]
        ov = o_vec[I]
        μ_pred[b] = mean(pv)                        # mean predicted (original scale)
        μ_obs[b]  = mean(ov)                        # mean observed
        s  = std(ov)
        se = m > 1 ? s / sqrt(m) : 0.0
        lo95[b] = μ_obs[b] - 1.96 * se
        hi95[b] = μ_obs[b] + 1.96 * se
        if truncate_at_zero
            floor = isnothing(clamp_floor) ? 0.0 : clamp_floor
            lo95[b] = max(lo95[b], floor)
        end
    end

    # Figure
    f = CairoMakie.Figure()
    ax = CairoMakie.Axis(f[1,1];
        title  = "Calibration curve",
        xlabel = "Predicted (bin mean)",
        ylabel = "Observed (mean ± 95% CI)"
    )

    # y = x guide over the span of μ_pred
    keep_x = .!isnan.(μ_pred)
    if any(keep_x)
        xmin = minimum(μ_pred[keep_x])
        xmax = maximum(μ_pred[keep_x])
        CairoMakie.lines!(ax, [xmin, xmax], [xmin, xmax], color=:grey, alpha=0.6)
    end

    # Error bars + points (only finite bins)
    keep = .!isnan.(μ_pred) .& .!isnan.(μ_obs) .& .!isnan.(lo95) .& .!isnan.(hi95)
    CairoMakie.errorbars!(ax, μ_pred[keep], μ_obs[keep],
                          μ_obs[keep] .- lo95[keep],  # lower whisker
                          hi95[keep] .- μ_obs[keep],  # upper whisker
                          whiskerwidth=20, linewidth=4, alpha=0.7)
    CairoMakie.scatter!(ax, μ_pred[keep], μ_obs[keep]; markersize=20, alpha=1.0)

    if save_path !== nothing
        try; mkdir(save_path); catch; end
        CairoMakie.save(joinpath(save_path, "calibration_curve_$(String(interval))_$(String(binning)).pdf"), f)
    end
    return f
end







#=
plot priors from inference result
 =#
function plot_priors(inference; save_path="")
    # try creating folder to save in
    try
        mkdir(save_path) 
    catch
    end
    # rescale the parameters according to the factor
    priors = inference["priors"]

    i = 1
    for (var, dist) in priors
        prior_i = StatsPlots.plot(dist, title=var, ylabel="Density", xlabel="Sample value", legend=false)
        if !isempty(save_path)
            savefig(prior_i, save_path*"/prior_$(var).png")
        end
        i += 1
        StatsPlots.closeall()
    end
    return nothing 
end

using CairoMakie
using Statistics

using CairoMakie
using Statistics

using CairoMakie
using Statistics

function plot_calibration_cross(inference;
    save_path::Union{Nothing,String}=nothing,
    N_samples::Int=200,
    interval::Symbol=:process,      # y-interval source: :process or :predictive
    x_ci_from::Symbol=:process,     # x-interval source: :process or :predictive
    bins::Int=20,
    binning::Symbol=:geometric,     # :quantile | :geometric | :adaptive
    min_bin_count::Int=100,         # used only for :adaptive
    skip_first_timepoint::Bool=true,
    truncate_at_zero::Bool=true,
    clamp_floor::Union{Nothing,Float64}=0.0,
    v_linewidth::Real=8,
    h_linewidth::Real=8
)
    # --- Unpack ---
    data        = inference["data"]
    chain       = inference["chain"]
    timepoints  = inference["timepoints"]
    seed        = inference["seed_idx"]
    sol_idxs    = inference["sol_idxs"]
    Ltuple      = inference["L"]
    labels      = inference["labels"]
    ks          = collect(keys(inference["priors"]))
    N_pars      = findall(x->x=="σ", ks)[1] - 1
    factors     = [1.0 for _ in 1:N_pars]
    ode         = odes[inference["ode"]]
    N           = size(data, 1)

    # aesthetics
    cross_color = RGBf(0/255, 71/255, 171/255)           # no alpha baked in

    # Data summaries
    mean_data =
        ndims(data) == 3 ? mean3(data) :
        ndims(data) == 2 ? data :
        error("Unsupported data array with ndims=$(ndims(data)).")

    # Observed time grid
    tgrid = timepoints

    # ODE problem
    prob = make_ode_problem(ode;
        labels     = labels,
        Ltuple     = Ltuple,
        factors    = factors,
        u0         = inference["u0"],
        timepoints = tgrid,
    )

    # Posterior draws
    posterior_samples = sample(chain, min(N_samples, length(chain[:lp])); replace=false)
    S = size(posterior_samples, 1)
    T = length(tgrid)
    traj = Array{Float64}(undef, N, T, S)   # process trajectories

    # Parameter indices (robust)
    par_names    = chain.name_map.parameters
    sigma_ch_idx = findfirst(==(Symbol("σ")), par_names)
    sigma_ch_idx === nothing && error("Could not find :σ in chain.name_map.parameters")
    seed_ch_idx = nothing
    if get(inference, "bayesian_seed", false)
        seed_ch_idx = findfirst(==(Symbol("seed")), par_names)
        seed_ch_idx === nothing && error("Could not find :seed in chain.name_map.parameters")
    end

    # Zero template u0
    u0_template = fill!(similar(inference["u0"]), 0.0)

    # Solve per draw
    for (s, sample_vec) in enumerate(eachrow(Array(posterior_samples)))
        p    = sample_vec[1:N_pars]
        u0_s = copy(u0_template)
        if seed_ch_idx === nothing
            u0_s[seed] = inference["seed_value"]
        else
            u0_s[seed] = sample_vec[seed_ch_idx]
        end
        sol = solve(prob, Tsit5(); p=p, u0=u0_s, saveat=tgrid, abstol=1e-9, reltol=1e-6)
        traj[:, :, s] = Array(sol[sol_idxs, :])
    end

    # Predictive (process + Normal noise)
    ytraj = similar(traj)
    for (s, sample_vec) in enumerate(eachrow(Array(posterior_samples)))
        σ_s = sample_vec[sigma_ch_idx]
        ytraj[:, :, s] = traj[:, :, s] .+ (σ_s .* randn(N, T))
    end

    # Choose arrays for vertical (y) and horizontal (x) CIs
    arr_y = (interval  === :predictive) ? ytraj : traj
    arr_x = (x_ci_from === :predictive) ? ytraj : traj

    # Predicted medians at observed grid (from process traj for stable x-centers)
    pred_med = mapslices(x -> quantile(x, 0.5), traj; dims=3)[:, :, 1]   # (N,T)

    # Skip first timepoint if desired
    first_col = skip_first_timepoint ? 2 : 1
    first_col <= size(pred_med, 2) || error("No timepoints left after skipping the first.")

    # --- Work in matrix form to keep original linear indices ---
    Mpred = pred_med[:, first_col:end]          # (Nreg, Tsel)
    Mobs  = mean_data[:, first_col:end]
    maskM = .!ismissing.(Mobs) .& isfinite.(Mpred)
    orig_lin_idx = findall(maskM)               # linear indices into (Nreg, Tsel)

    p_vec = Float64.(Mpred[maskM])              # predicted centers (x)
    o_vec = Float64.(Mobs[maskM])               # observed centers (y)
    isempty(p_vec) && error("No data for calibration after filtering.")

    # Optional clamp
    if truncate_at_zero
        floor = isnothing(clamp_floor) ? 0.0 : clamp_floor
        p_vec = clamp.(p_vec, floor, Inf)
        o_vec = clamp.(o_vec, floor, Inf)
    end

    # ----------------- Non-uniform bins -----------------
    bins = max(bins, 2)
    _binning = Symbol(lowercase(String(binning)))

    _monotone!(edges::Vector{Float64}) = begin
        for i in 2:length(edges)
            if edges[i] <= edges[i-1]
                edges[i] = nextfloat(edges[i-1])
            end
        end
        edges
    end

    edges = Float64[]
    if _binning === :quantile
        edges = collect(Float64.(quantile(p_vec, range(0, 1; length=bins+1))))
        _monotone!(edges)
    elseif _binning === :geometric || _binning === :adaptive
        pmin, pmax = minimum(p_vec), maximum(p_vec)
        ε = 1e-12
        start = max(ε, min(pmin > 0 ? pmin : ε, pmax))
        stopv = max(pmax, start*10)
        ge = 10.0 .^ range(log10(start), log10(stopv); length=bins+1) |> collect
        if pmin == 0.0
            ge[1] = 0.0
        end
        edges = _monotone!(Float64.(ge))
        if _binning === :adaptive
            while true
                if length(edges) < 3; break; end
                idx_tmp = [min(searchsortedlast(edges, v), length(edges)-1) for v in p_vec]
                counts  = [count(==(b), idx_tmp) for b in 1:(length(edges)-1)]
                isempty(findall(b -> counts[b] < min_bin_count, 1:length(counts))) && break
                new_edges = Float64[edges[1]]
                b = 1
                while b <= length(counts)
                    if counts[b] < min_bin_count
                        if b < length(counts)
                            push!(new_edges, edges[b+2]); b += 2
                        else
                            new_edges[end] = edges[end]; b += 1
                        end
                    else
                        push!(new_edges, edges[b+1]); b += 1
                    end
                end
                edges = _monotone!(new_edges)
            end
        end
    else
        error("binning must be :quantile, :geometric, or :adaptive")
    end
    (length(edges) >= 2) || error("Failed to build bin edges")

    # Assign bins in x
    binidx = [min(searchsortedlast(edges, v), length(edges)-1) for v in p_vec]
    B = length(edges) - 1

    # Aggregates per bin
    μ_pred = fill(NaN, B)
    μ_obs  = fill(NaN, B)
    lo95_y = fill(NaN, B)
    hi95_y = fill(NaN, B)
    lo95_x = fill(NaN, B)
    hi95_x = fill(NaN, B)
    m_in   = zeros(Int, B)

    # sizes to decode linear indices
    Nreg, Tsel = size(Mpred)

    for b in 1:B
        I = findall(==(b), binidx)      # indices into p_vec/o_vec (and orig_lin_idx)
        m = length(I)
        m_in[b] = m
        if m == 0; continue; end

        pv = p_vec[I]; ov = o_vec[I]
        μ_pred[b] = mean(pv)
        μ_obs[b]  = mean(ov)

        # Vertical = empirical 95% PI of observed values in the bin
        lo95_y[b] = quantile(ov, 0.025)
        hi95_y[b] = quantile(ov, 0.975)
        if truncate_at_zero
            floor = isnothing(clamp_floor) ? 0.0 : clamp_floor
            lo95_y[b] = max(lo95_y[b], floor)
        end
        if truncate_at_zero
            floor = isnothing(clamp_floor) ? 0.0 : clamp_floor
            lo95_y[b] = max(lo95_y[b], floor)
        end

        # Horizontal CI for predictions from arr_x across draws and the SAME positions
        xsamps = Float64[]
        sizehint!(xsamps, m * size(arr_x, 3))
        for idx_in_vec in I
            lin = orig_lin_idx[idx_in_vec]          # linear index into (Nreg, Tsel)
            ci = CartesianIndices((Nreg, Tsel))[lin]
            i, j = Tuple(ci)
            t = (first_col - 1) + j                 # original time index
            @inbounds for sidx in axes(arr_x, 3)
                push!(xsamps, arr_x[i, t, sidx])
            end
        end
        if !isempty(xsamps)
            lo95_x[b] = quantile(xsamps, 0.025)
            hi95_x[b] = quantile(xsamps, 0.975)
        end
    end

    # Figure
    f = CairoMakie.Figure()
    ax = CairoMakie.Axis(f[1,1];
        title  = "Binned predictive vs observed",
        xlabel = "Predicted (mean ± 95% PI) ",
        ylabel = "Observed (mean ± 95% QI)"
    )

    # y = x guide
    keepx = .!isnan.(μ_pred)
    if any(keepx)
        xmin, xmax = minimum(μ_pred[keepx]), maximum(μ_pred[keepx])
        CairoMakie.lines!(ax, [xmin, xmax], [xmin, xmax], color=:grey, alpha=0.5)
    end

    # Cross error bars
    keep = .!isnan.(μ_pred) .& .!isnan.(μ_obs)
    for b in findall(keep)
        # horizontal (predicted) 95% CI through the dot's y
        #if isfinite(lo95_x[b]) && isfinite(hi95_x[b])
        #    CairoMakie.lines!(ax, [lo95_x[b], hi95_x[b]], [μ_obs[b], μ_obs[b]];
        #                      color=cross_color, alpha=0.75, linewidth=h_linewidth)
        #end
        # horizontal WITH WHISKERS? (predicted) 95% CI through the dot's y
        if isfinite(lo95_x[b]) && isfinite(hi95_x[b])
            CairoMakie.errorbars!(ax,
                [μ_pred[b]], [μ_obs[b]],    # center
                [μ_pred[b] - lo95_x[b]],    # left extent
                [hi95_x[b] - μ_pred[b]];    # right extent
                direction = :x,
                color = (cross_color,0.7),
                whiskerwidth = 15,
                linewidth = h_linewidth
            )
        end

        # vertical (observed-mean) CI through the dot's x
        if isfinite(lo95_y[b]) && isfinite(hi95_y[b])
            CairoMakie.errorbars!(ax, [μ_pred[b]], [μ_obs[b]],
                                  [μ_obs[b] - lo95_y[b]], [hi95_y[b] - μ_obs[b]];
                                  color=(cross_color,0.7), whiskerwidth=15, linewidth=v_linewidth)
        end
        CairoMakie.scatter!(ax, [μ_pred[b]], [μ_obs[b]]; markersize=20, color=(cross_color,0.9), strokecolor=:black, strokewidth=1.5 )
    end

    if save_path !== nothing
        try; mkdir(save_path); catch; end
        CairoMakie.save(joinpath(save_path,
            "calibration_curve_cross_$(String(interval))_x$(String(x_ci_from))_$(String(binning)).pdf"), f)
    end
    return f
end






#=
plot priors and posteriors together from inference result
 =#
function plot_prior_and_posterior(inference; save_path="")
    # try creating folder to save in
    try
        mkdir(save_path) 
    catch
    end
    # rescale the priors
    chain = inference["chain"]
    priors = inference["priors"]
    vars = collect(keys(priors))
    master_fig = StatsPlots.plot(chain) 
    for (i,var) in enumerate(vars)
        plot_i = StatsPlots.plot(master_fig[i,2], title=var)
        StatsPlots.plot!(plot_i, priors[var])
        if !isempty(save_path)
            savefig(plot_i, save_path*"/prior_and_posterior_$(var).png")
        end
        StatsPlots.closeall()
    end
    return nothing
end


function plot_ppc_coverages(inference;
    save_path::Union{Nothing,String}=nothing,
    levels::AbstractVector{<:Real} = [0.50, 0.80, 0.95],
    N_samples::Int = 200,
    skip_first_timepoint::Bool = true,
    truncate_at_zero::Bool = true,
    clamp_floor::Union{Nothing,Float64} = 0.0,
)
    # --- Unpack ---
    data        = inference["data"]
    chain       = inference["chain"]
    timepoints  = inference["timepoints"]
    seed        = inference["seed_idx"]
    sol_idxs    = inference["sol_idxs"]
    Ltuple      = inference["L"]
    labels      = inference["labels"]
    ks          = collect(keys(inference["priors"]))
    N_pars      = findall(x->x=="σ", ks)[1] - 1
    factors     = get(inference, "factors", [1.0 for _ in 1:N_pars])  # fallback
    ode         = odes[inference["ode"]]
    N           = size(data, 1)

    # Observed values to check coverage against
    # If 3D: treat each replicate as a separate observation
    # If 2D: just those values
    if ndims(data) == 3
        obs_mat = data                      # (region, time, rep)
        n_rep   = size(data, 3)
    elseif ndims(data) == 2
        obs_mat = reshape(data, N, size(data,2), 1)
        n_rep   = 1
    else
        error("Unsupported data array with ndims=$(ndims(data)).")
    end

    # Time grid = observed times
    tgrid = timepoints

    # ODE problem definition (we will set u0 per draw)
    prob = make_ode_problem(ode;
        labels     = labels,
        Ltuple     = Ltuple,
        factors    = factors,
        u0         = inference["u0"],
        timepoints = tgrid,
    )

    # Posterior draws
    posterior_samples = sample(chain, min(N_samples, length(chain[:lp])); replace=false)
    S = size(posterior_samples, 1)
    T = length(tgrid)

    # Parameter indices
    par_names    = chain.name_map.parameters
    sigma_ch_idx = findfirst(==(Symbol("σ")), par_names)
    sigma_ch_idx === nothing && error("Could not find :σ in chain.name_map.parameters")
    seed_ch_idx = nothing
    if get(inference, "bayesian_seed", false)
        seed_ch_idx = findfirst(==(Symbol("seed")), par_names)
        seed_ch_idx === nothing && error("Could not find :seed in chain.name_map.parameters")
    end

    # Solve trajectories per draw
    traj = Array{Float64}(undef, N, T, S)   # process trajectories
    u0_template = fill!(similar(inference["u0"]), 0.0)
    for (s, sample_vec) in enumerate(eachrow(Array(posterior_samples)))
        p    = sample_vec[1:N_pars]
        u0_s = copy(u0_template)
        if seed_ch_idx === nothing
            u0_s[seed] = inference["seed_value"]
        else
            u0_s[seed] = sample_vec[seed_ch_idx]
        end
        sol = solve(prob, Tsit5(); p=p, u0=u0_s, saveat=tgrid, abstol=1e-9, reltol=1e-6)
        traj[:, :, s] = Array(sol[sol_idxs, :])
    end

    # Posterior predictive: add Normal noise per draw
    ypred = similar(traj)
    for (s, sample_vec) in enumerate(eachrow(Array(posterior_samples)))
        σ_s = sample_vec[sigma_ch_idx]
        @inbounds ypred[:, :, s] = traj[:, :, s] .+ (σ_s .* randn(N, T))
    end

    # Optionally skip the first timepoint
    first_col = skip_first_timepoint ? 2 : 1
    first_col <= T || error("No timepoints left after skipping the first.")

    # Clamp to floor if requested (both predictive and observed)
    if truncate_at_zero
        floor = isnothing(clamp_floor) ? 0.0 : clamp_floor
        @inbounds ypred .= clamp.(ypred, floor, Inf)
    end

    # Build masks for valid observed entries (exclude missing)
    obs_slice = view(obs_mat, :, first_col:T, :)                       # (N, Tsel, R)
    valid_mask = .!ismissing.(obs_slice)

    # Convert observed to Float64 where valid
    obs_vals = similar(obs_slice, Float64)
    @inbounds for i in axes(obs_slice,1), j in axes(obs_slice,2), k in axes(obs_slice,3)
        if valid_mask[i,j,k]
            v = obs_slice[i,j,k]
            obs_vals[i,j,k] = Float64(v)
        else
            obs_vals[i,j,k] = NaN
        end
    end
    if truncate_at_zero
        floor = isnothing(clamp_floor) ? 0.0 : clamp_floor
        @inbounds obs_vals .= clamp.(obs_vals, floor, Inf)
    end

    # Coverage per level
    levels = collect(levels)
    (all(0 .< levels .< 1)) || error("levels must be in (0,1).")
    Tsel = T - (first_col - 1)

    cover_fracs = Float64[]
    nominal     = Float64[]

    # Pre-allocate buffers for quantiles
    lowq   = similar(obs_vals, Float64, N, Tsel)
    highq  = similar(obs_vals, Float64, N, Tsel)

    for α in levels
        lo = (1 - α)/2
        hi = 1 - lo

        # Compute predictive interval bounds per (i,t)
        @inbounds for i in 1:N, tj in 1:Tsel
            t = (first_col - 1) + tj
            v = view(ypred, i, t, :)
            lowq[i, tj]  = quantile(v, lo)
            highq[i, tj] = quantile(v, hi)
        end

        # Count coverage over all valid observed replicates
        inside = 0
        total  = 0
        @inbounds for i in 1:N, tj in 1:Tsel, k in 1:n_rep
            if valid_mask[i, tj, k]
                y = obs_vals[i, tj, k]
                # skip if y is NaN (shouldn’t happen given mask)
                if isfinite(y)
                    total += 1
                    if lowq[i, tj] <= y <= highq[i, tj]
                        inside += 1
                    end
                end
            end
        end

        push!(nominal, α)
        push!(cover_fracs, total == 0 ? NaN : inside / total)
    end

    # ---- Plot bar chart of coverage vs nominal level ----
    labels = string.(round.(100 .* nominal; digits=0)) .* "%"
    f = CairoMakie.Figure()
    ax = CairoMakie.Axis(f[1,1];
        title  = "Posterior predictive coverage",
        xlabel = "Nominal level",
        ylabel = "Empirical coverage"
    )

    # bars
    xs = 1:length(levels)
    CairoMakie.barplot!(ax, xs, cover_fracs; color=RGBAf(0.2,0.4,0.7,0.8))
    ax.xticks = (xs, labels)

    # reference points/line at y = nominal level (helpful visual target)
    CairoMakie.scatter!(ax, xs, nominal; color=:black, markersize=10)
    CairoMakie.lines!(ax, xs, nominal; color=:black, linestyle=:dash, alpha=0.6)

    # y-limits [0,1] with small headroom
    Makie.ylims!(ax, 0.0, 1.02)
    Makie.xlims!(ax, 0.5, length(xs) + 0.5)

    if save_path !== nothing
        try; mkdir(save_path); catch; end
        CairoMakie.save(joinpath(save_path, "ppc_coverage_levels.pdf"), f)
    end
    return f, (; levels=nominal, coverage=cover_fracs)
end


using CairoMakie, Statistics, Distributions

function plot_ppc_coverage_by_region(inference;
    save_path::Union{Nothing,String}=nothing,
    N_samples::Int=400,
    level::Float64=0.95,
    skip_first_timepoint::Bool=true,
    truncate_at_zero::Bool=true,
    clamp_floor::Union{Nothing,Float64}=0.0,
)
    # --- Unpack ---
    data       = inference["data"]
    chain      = inference["chain"]
    timepoints = inference["timepoints"]
    seed       = inference["seed_idx"]
    sol_idxs   = inference["sol_idxs"]
    Ltuple     = inference["L"]
    labels     = inference["labels"]
    ks         = collect(keys(inference["priors"]))
    N_pars     = findall(x->x=="σ", ks)[1] - 1
    factors    = get(inference, "factors", [1.0 for _ in 1:N_pars])
    ode        = odes[inference["ode"]]

    # Shapes
    R = size(data, 1)
    nd = ndims(data)
    if nd == 3
        # (region, time, replicate)
        nothing
    elseif nd == 2
        # (region, time)
        nothing
    else
        error("Unsupported data dims $(ndims(data)).")
    end

    # Use observed time grid
    tgrid = timepoints
    T     = length(tgrid)                  # ensure consistency with lo/hi shapes
    first_col = skip_first_timepoint ? 2 : 1
    tsel = first_col:T
    isempty(tsel) && error("No timepoints left after skipping the first.")

    # ODE problem template
    prob = make_ode_problem(ode;
        labels     = labels,
        Ltuple     = Ltuple,
        factors    = factors,
        u0         = inference["u0"],
        timepoints = tgrid,
    )

    # Posterior samples
    posterior_samples = sample(chain, min(N_samples, length(chain[:lp])); replace=false)
    S = size(posterior_samples, 1)

    # Parameter indices
    par_names    = chain.name_map.parameters
    sigma_ch_idx = findfirst(==(Symbol("σ")), par_names)
    sigma_ch_idx === nothing && error("Could not find :σ in chain.name_map.parameters")
    seed_ch_idx = nothing
    if get(inference, "bayesian_seed", false)
        seed_ch_idx = findfirst(==(Symbol("seed")), par_names)
        seed_ch_idx === nothing && error("Could not find :seed in chain.name_map.parameters")
    end

    # Simulate process trajectories for each draw at observed timepoints
    traj = Array{Float64}(undef, R, T, S)
    u0_template = fill!(similar(inference["u0"]), 0.0)
    for (s, sample_vec) in enumerate(eachrow(Array(posterior_samples)))
        p    = sample_vec[1:N_pars]
        u0_s = copy(u0_template)
        if seed_ch_idx === nothing
            u0_s[seed] = inference["seed_value"]
        else
            u0_s[seed] = sample_vec[seed_ch_idx]
        end
        sol = solve(prob, Tsit5(); p=p, u0=u0_s, saveat=tgrid, abstol=1e-9, reltol=1e-6)
        traj[:, :, s] = Array(sol[sol_idxs, :])
    end

    # Posterior predictive draws: add Normal noise per draw using that draw’s σ
    ypred = similar(traj)
    for (s, sample_vec) in enumerate(eachrow(Array(posterior_samples)))
        σ_s = sample_vec[sigma_ch_idx]
        ypred[:, :, s] = traj[:, :, s] .+ (σ_s .* randn(R, T))
    end

    # Truncate floor if requested (affects the interval and comparisons)
    if truncate_at_zero
        floor = isnothing(clamp_floor) ? 0.0 : clamp_floor
        ypred = clamp.(ypred, floor, Inf)
    end

    # Predictive interval bounds at each (r,t) across S draws
    α = (1 - level)/2
    lo = mapslices(x -> quantile(x, α),   ypred; dims=3)[:, :, 1]   # (R,T)
    hi = mapslices(x -> quantile(x, 1-α), ypred; dims=3)[:, :, 1]

    # Count per region: numerator = inside PI, denominator = total observed points
    num = zeros(Int, R)
    den = zeros(Int, R)

    if nd == 3
        # data: (r,t,k)
        for r in 1:R
            for t in tsel
                ℓ = lo[r,t]; h = hi[r,t]
                @inbounds for k in axes(data, 3)
                    v = data[r,t,k]
                    if v !== missing
                        y = Float64(v)
                        den[r] += 1
                        if (y >= ℓ) && (y <= h)
                            num[r] += 1
                        end
                    end
                end
            end
        end
    else
        # data: (r,t)
        for r in 1:R
            for t in tsel
                v = data[r,t]
                if v !== missing
                    y = Float64(v)
                    ℓ = lo[r,t]; h = hi[r,t]
                    den[r] += 1
                    if (y >= ℓ) && (y <= h)
                        num[r] += 1
                    end
                end
            end
        end
    end

    frac = fill(NaN, R)
    for r in 1:R
        frac[r] = den[r] > 0 ? num[r] / den[r] : NaN
    end

    # Sanity check
    bad = findall(r -> !isnan(frac[r]) && (frac[r] > 1 + 1e-12 || frac[r] < -1e-12), 1:R)
    if !isempty(bad)
        @warn "Coverage out of bounds detected" bad_regions=bad num=num[bad] den=den[bad]
        frac[bad] .= clamp.(frac[bad], 0.0, 1.0)  # last-resort clamp to render plot
    end

    # --- Plot (keep labels sane for large R) ---
    f = Figure(resolution=(max(1400, 3R), 500))
    ax = Axis(f[1,1];
        title  = "Posterior predictive coverage by region (level=$(round(level*100))%)",
        xlabel = "Region",
        ylabel = "Coverage (fraction in interval)",
    )
    CairoMakie.barplot!(ax, 1:R, frac; color=:steelblue)
    CairoMakie.hlines!(ax, [level]; color=:red, linewidth=3, linestyle=:dash)
    Makie.ylims!(ax, 0, 1.0)

    # Hide labels if too many; otherwise show a subset
    if R > 60
        ax.xticklabelsvisible = false
    else
        k = max(1, ceil(Int, R/20))
        sel = 1:k:R
        Makie.xticks!(ax, (sel, labels[sel]))
        ax.xticklabelrotation = π/2
        ax.xticklabelsize = 8
    end

    if save_path !== nothing
        try; mkdir(save_path); catch; end
        CairoMakie.save(joinpath(save_path, "ppc_coverage_by_region.pdf"), f)
    end

    return f, (; frac, num, den, level, tsel)
end


# ----- PARAM PATH LENGTH
"""
Return unique base names from priors that look like `name[<int>]`, in first-seen order.
"""
function vector_bases_from_priors(inference)::Vector{String}
    bases, seen = String[], Set{String}()
    for v in collect(keys(inference["priors"]))
        m = match(r"^([^\[]+)\[(\d+)\]$", v)  # base[i]
        m === nothing && continue
        b = String(m.captures[1])
        if !(b in seen); push!(bases, b); push!(seen, b); end
    end
    bases
end

"""
Posterior-mean vector μ (length R) for one base, using `priors` order to map base[i] -> p[k].
Assumes chain stores components as p[1] / p__1 / p_1 / p.1 .
"""
function posterior_mean_vector_from_priors(chain, priors, base::String, R::Int)
    # 1) map base[i] -> global index k using priors order up to "σ"
    pri_keys = collect(keys(priors))
    kσ = findfirst(==("σ"), pri_keys)
    N_pars = isnothing(kσ) ? length(pri_keys) : (kσ - 1)
    idxs = fill(0, R)
    for (k, name) in enumerate(pri_keys[1:N_pars])
        m = match(Regex("^" * escape_string(base) * "\\[(\\d+)\\]\$"), name)
        if m !== nothing
            j = parse(Int, m.captures[1])
            (1 ≤ j ≤ R) && (idxs[j] = k)
        end
    end
    any(==(0), idxs) && error("Family '$base' missing some indices 1:$R in priors order.")

    # 2) tiny getter for p[k] with common encodings
    function _pvals(k)
        for sym in (Symbol("p[$k]"), Symbol("p__$(k)"), Symbol("p_$(k)"), Symbol("p.$(k)"))
            try
                return chain[sym]   # returns array of draws
            catch
            end
        end
        error("Could not find chain column for p[$k] (tried [k], __k, _k, .k).")
    end

    # 3) compute means
    μ = Vector{Float64}(undef, R)
    @inbounds for i in 1:R
        μ[i] = mean(vec(_pvals(idxs[i])))
    end
    μ
end

"""
Weighted shortest-path lengths from Laplacian L = inference["L"][1].
edge_cost = :reciprocal -> cost = 1/w (default), or :direct -> cost = w.
"""
function pathlengths_from_L(inference; symmetrize::Bool=false, edge_cost::Symbol=:reciprocal)
    L = Matrix(inference["L"][1])
    W = max.(0.0, -Float64.(L));  W[diagind(W)] .= 0.0
    if symmetrize; W = 0.5 .* (W .+ W'); end
    cost = edge_cost === :reciprocal ? (w->1.0/w) :
            edge_cost === :direct     ? (w->w)    :
            error("edge_cost must be :reciprocal or :direct")
    n = size(W,1)
    g = symmetrize ? SimpleWeightedGraph(n) : SimpleWeightedDiGraph(n)
    if symmetrize
        @inbounds for i in 1:n, j in i+1:n
            (W[i,j] > 0) && add_edge!(g, i, j, cost(W[i,j]))
        end
    else
        @inbounds for i in 1:n, j in 1:n
            (i!=j && W[i,j] > 0) && add_edge!(g, i, j, cost(W[i,j]))
        end
    end
    dijkstra_shortest_paths(g, inference["seed_idx"]).dists
end

"""
Plot posterior-mean of `base[i]` vs weighted path length from seed.
"""
function plot_param_vs_pathlength(inference;
    base::String,
    symmetrize::Bool=false,
    overlay_regression::Bool=false,
    show_r2::Bool=true,
    min_prior_shift::Union{Nothing,Float64}=nothing,  # standardized shift filter (≥ threshold)
    region_idxs::Union{Nothing,AbstractVector{<:Integer}}=nothing,  # <-- NEW: only plot these regions
    save_path::Union{Nothing,String}=nothing,
    filename::String="param_vs_pathlength.pdf"
)
    R = size(inference["data"], 1)
    chain, priors = inference["chain"], inference["priors"]

    # distances (cost = 1/w; you already fixed pathlengths_from_L)
    d = pathlengths_from_L(inference; symmetrize=symmetrize)
    @assert length(d) == R
    keep = .!isinf.(d) .& .!isnan.(d)

    # optional: restrict to a user-provided subset of region indices
    if region_idxs !== nothing
        sel = falses(R)
        @inbounds for i in region_idxs
            if 1 <= i <= R
                sel[i] = true
            end
        end
        keep .&= sel
    end

    # posterior means for base[i] via your priors→p[k] mapping
    y = posterior_mean_vector_from_priors(chain, priors, base, R)

    # single filter: standardized mean shift relative to prior SD
    if min_prior_shift !== nothing
        μprior = similar(d); σprior = similar(d)
        @inbounds for i in 1:R
            pr = priors["$base[$i]"]
            μprior[i] = mean(pr)
            σprior[i] = std(pr)
        end
        δ = abs.(y .- μprior) ./ σprior
        keep .&= (δ .>= min_prior_shift)
    end

    keep .&= .!isnan.(y)

    f = Figure()
    ax = Axis(f[1,1];
        title  = base,
        xlabel = symmetrize ? "Weighted path length (undirected)" : "Weighted path length (directed)",
        ylabel = "Posterior mean of $(base)[i]"
    )
    CairoMakie.scatter!(ax, d[keep], y[keep])

    if overlay_regression && sum(keep) ≥ 2
        X  = hcat(ones(sum(keep)), d[keep]); β = X \ y[keep]
        xs = range(minimum(d[keep]), maximum(d[keep]); length=200)
        lines!(ax, xs, β[1] .+ β[2] .* xs)
        if show_r2 && sum(keep) ≥ 3
            ŷ  = X * β
            r2 = 1 - sum((y[keep] .- ŷ).^2) / sum((y[keep] .- mean(y[keep])).^2)
            ax.title = "$(base)  (R² = $(round(r2, digits=3))), n=$(sum(keep))"
        end
    end

    if save_path !== nothing
        try; mkdir(save_path); catch; end
        CairoMakie.save(joinpath(save_path, filename), f)
    end
    return f
end


using CairoMakie, Statistics, Distributions

# Assumes you already have this mapping helper (from earlier):
# posterior_mean_vector_from_priors(chain, priors, base::String, R::Int)

"""
plot_two_local_params_scatter(inference; bases=nothing, min_prior_shift=nothing,
                              region_idxs=nothing, overlay_regression=false,
                              show_r2=true, save_path=nothing,
                              filename="two_local_params_scatter.pdf")

- If `bases === nothing`, auto-detects exactly TWO local families from keys(priors).
  If it doesn't find exactly two, returns `nothing` and does nothing.
- If `bases` is provided, pass a Tuple like ("beta","gamma").

Filters:
- `region_idxs` :: Vector{Int} — only plot these regions.
- `min_prior_shift` :: Real — keep regions where *either* family moved by at least
  this many prior SDs: |E_post - E_prior| / SD_prior ≥ threshold.

Scatter axes: x = mean of first base, y = mean of second base.
"""
function plot_two_local_params_scatter(inference;
    bases::Union{Nothing,Tuple{String,String}}=nothing,
    min_prior_shift::Union{Nothing,Float64}=nothing,
    region_idxs::Union{Nothing,AbstractVector{<:Integer}}=nothing,
    overlay_regression::Bool=false,
    show_r2::Bool=false,
    save_path::Union{Nothing,String}=nothing,
    filename::String="two_local_params_scatter.pdf",
)
    R       = size(inference["data"], 1)
    chain   = inference["chain"]
    priors  = inference["priors"]
    pk      = collect(keys(priors))

    # Auto-detect two bases from priors if not provided
    if bases === nothing
        found = String[]
        seen  = Set{String}()
        @inbounds for v in pk
            m = match(r"^([^\[]+)\[(\d+)\]$", v)
            if m !== nothing
                b = String(m.captures[1])
                if !(b in seen)
                    push!(found, b); push!(seen, b)
                end
            end
        end
        if length(found) != 2
            @info "plot_two_local_params_scatter: expected exactly 2 local families, found $(length(found)). Doing nothing."
            return nothing
        end
        bases = (found[1], found[2])
    end
    b1, b2 = bases

    # Posterior means per region for both families (via priors→p[k] mapping)
    μx = posterior_mean_vector_from_priors(chain, priors, b1, R)
    μy = posterior_mean_vector_from_priors(chain, priors, b2, R)

    # Build keep mask
    keep = trues(R)
    if region_idxs !== nothing
        sel = falses(R)
        @inbounds for i in region_idxs
            if 1 <= i <= R; sel[i] = true; end
        end
        keep .&= sel
    end

    if min_prior_shift !== nothing
        μp1 = similar(μx); σp1 = similar(μx)
        μp2 = similar(μy); σp2 = similar(μy)
        @inbounds for i in 1:R
            pr1 = priors["$b1[$i]"]; μp1[i] = mean(pr1); σp1[i] = std(pr1)
            pr2 = priors["$b2[$i]"]; μp2[i] = mean(pr2); σp2[i] = std(pr2)
        end
        δ1 = abs.(μx .- μp1) ./ σp1
        δ2 = abs.(μy .- μp2) ./ σp2
        keep .&= (δ1 .>= min_prior_shift) .| (δ2 .>= min_prior_shift)
    end

    keep .&= .!isnan.(μx) .& .!isnan.(μy)

    # Plot
    f = Figure()
    ax = Axis(f[1,1];
        title  = "$(b1) vs $(b2) (posterior means)",
        xlabel = "$(b1) posterior mean",
        ylabel = "$(b2) posterior mean",
    )
    CairoMakie.scatter!(ax, μx[keep], μy[keep])

    if overlay_regression && sum(keep) ≥ 2
        X  = hcat(ones(sum(keep)), μx[keep])
        β  = X \ μy[keep]
        xs = range(minimum(μx[keep]), maximum(μx[keep]); length=200)
        lines!(ax, xs, β[1] .+ β[2] .* xs)
        if show_r2 && sum(keep) ≥ 3
            ŷ  = X * β
            r2 = 1 - sum((μy[keep] .- ŷ).^2) / sum((μy[keep] .- mean(μy[keep])).^2)
            ax.title = @sprintf("%s vs %s  (R² = %.3f, n=%d)", b1, b2, r2, sum(keep))
        end
    end

    if save_path !== nothing
        try; mkdir(save_path); catch; end
        CairoMakie.save(joinpath(save_path, filename), f)
    end
    return f
end



#=
master plotting function (plot everything relevant to inference)
 =#
function plot_inference(inference, save_path; N_samples=300, show_variance=false, nonzero_regions=false)
    # load inference simulation 
    #display(inference["chain"])

    # create folder
    try
        mkdir(save_path);
    catch
    end
    if !inference["bayesian_seed"]
        try
            delete!(inference["priors"], "seed")
        catch
        end
    end

    # find nonzero indices for plotting if needed
    region_idxs = nothing
    if nonzero_regions
        region_idxs = nonzero_regions(inference["data"], eps=0.22)
    end
        
    # plot
    plot_two_local_params_scatter(inference;
    save_path=save_path*"/two_param_scatter",
    )
    # Plot parameters means as function of path length from seed
    for b in vector_bases_from_priors(inference)
        plot_param_vs_pathlength(
            inference;
            base      = b,
            min_prior_shift = nothing,
            region_idxs = region_idxs,
            save_path = save_path*"/path_parameters",
            filename  = "$(b)_vs_pathlength.pdf",
        )
    end

    plot_ppc_coverages(inference;
    save_path = save_path*"/coverage",
    levels = [0.25, 0.5, 0.75, 0.95],
    N_samples = 1000,
    skip_first_timepoint = false,
    truncate_at_zero = true,
    clamp_floor = 0.0
    )
    plot_ppc_coverage_by_region(inference;
    save_path=save_path*"/coverage",
    N_samples=400,
    level=0.75,
    skip_first_timepoint=false
    )
    plot_calibration(inference;
    save_path=save_path*"/calibration",      # directory to save output (or nothing for no file)
    N_samples=200,         # how many posterior samples to draw
    interval=:process,     # :process (traj only) or :predictive (traj + noise)
    binning = :geometric,
    bins=50,               # number of quantile bins
    #binning = :adaptive,
    #min_bin_count = 100,
    skip_first_timepoint=true,
    truncate_at_zero=true,
    clamp_floor=0.0
    )
    plot_calibration_cross(inference;
        save_path = save_path*"/calibration_cross",  # directory to save output (or nothing for no file)
        N_samples = 200,          # how many posterior samples to draw
        interval = :process,      # vertical CI source: :process (traj only) or :predictive (traj + noise)
        x_ci_from = :process,     # horizontal CI source: :process or :predictive
        #binning = :geometric,     # :quantile | :geometric | :adaptive
        #bins = 40,                # number of bins (ignored if :adaptive), we have ~3000 data points -> 50-200 bins
        binning = :adaptive,
        min_bin_count = 3,     # minimum points per bin (adaptive only), 30-100 bins for ~3000 data points
        skip_first_timepoint = true,
        truncate_at_zero = true,
        clamp_floor = 0.0,
        v_linewidth = 5,          # thickness of vertical bars
        h_linewidth = 5           # thickness of horizontal bars
    )
    predicted_observed(inference; save_path=save_path*"/predicted_observed_log10", plotscale=log10);
    predicted_observed(inference; save_path=save_path*"/predicted_observed_id", plotscale=identity);
    plot_retrodiction(inference; save_path=save_path*"/retrodiction",
                           N_samples=1000, level=0.90, interval=:predictive, line_from=:process,
                           data_style=:mean, data_error=:sd,
                           y_padding=0.05#, sync_y=true, ymax=1.0, ymin=-0.05
                           )
    plot_prior_and_posterior(inference; save_path=save_path*"/prior_and_posterior");
    plot_posteriors(inference, save_path=save_path*"/posteriors");
    #plot_chains(inference, save_path=save_path*"/chains");
    #plot_priors(inference; save_path=save_path*"/priors");
    return nothing
end