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
    xticks = ([1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1e-0], [L"$10^{-6}$", L"$10^{-5}$", L"$10^{-4}$", L"$10^{-3}$", L"$10^{-2}$", L"$10^{-1}$", L"$10^0$"])
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

    x = vec(copy(data))
    y = vec(copy(sol[regions,:]))
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
        CairoMakie.save(save_path * "/predicted_observed_mode.png", f)
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
            CairoMakie.save(save_path * "/predicted_observed_mode_$(i).png", f)
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

#=
plot retrodictino from inference result
 =#
function plot_retrodiction(inference; save_path=nothing, N_samples=1, show_variance=false)
    # try creating folder to save in
    try
        mkdir(save_path) 
    catch
    end
    # unload from simulation
    data = inference["data"]
    chain = inference["chain"]
    priors = inference["priors"]
    timepoints = inference["timepoints"]
    seed = inference["seed_idx"]
    sol_idxs = inference["sol_idxs"]
    Ltuple = inference["L"]
    labels = inference["labels"]
    ks = collect(keys(inference["priors"]))
    N_pars = findall(x->x=="σ",ks)[1] - 1
    factors = [1. for _ in 1:N_pars]
    ode = odes[inference["ode"]]
    N = size(data)[1]
    M = length(timepoints)
    par_names = chain.name_map.parameters
    if inference["bayesian_seed"]
        seed_ch_idx = findall(x->x==:seed,par_names)[1]  # TODO find index of chain programmatically
    end
    # if data is 3D, find mean
    if length(size(data)) > 2
        var_data = var3(data)
        mean_data = mean3(data)
    end

    # define ODE problem 
    u0 = inference["u0"]
    tspan = (0, timepoints[end])
    
    prob = make_ode_problem(ode;
        labels     = labels,
        Ltuple     = Ltuple,
        factors    = factors,
        u0         = u0,
        timepoints = timepoints,
    )

    #rhs(du,u,p,t;L=L, func=ode::Function) = func(du,u,p,t;L=L,factors=factors)
    #prob = ODEProblem(rhs, u0, tspan; alg=Tsit5())

    fs = Any[NaN for _ in 1:N]
    axs = Any[NaN for _ in 1:N]
    for i in 1:N
        f = CairoMakie.Figure(fontsize=20)
        ax = CairoMakie.Axis(f[1,1], title="Region $(i): $(labels[i])", ylabel="Percentage area with pathology", xlabel="time (months)", xticks=0:9, limits=(0,9.1,nothing,nothing))
        fs[i] = f
        axs[i] = ax
    end
    posterior_samples = sample(chain, N_samples; replace=false)
    for sample in eachrow(Array(posterior_samples))
        # samples
        p = sample[1:N_pars]  # first index is σ and last index is seed
        if inference["bayesian_seed"]
            u0[seed] = sample[seed_ch_idx]  
        else    
            u0[seed] = inference["seed_value"]
        end
        σ = sample[end-1]
        
        # solve
        sol_p = solve(prob,Tsit5(); p=p, u0=u0, saveat=0.1, abstol=1e-9, reltol=1e-6)
        t = sol_p.t
        sol_p = Array(sol_p[sol_idxs,:])
        for i in 1:N
            lower_bound = sol_p[i,:] .- σ
            upper_bound = sol_p[i,:] .+ σ
            if show_variance
                CairoMakie.band!(axs[i], t, lower_bound, upper_bound; color=(:grey,0.1))
                CairoMakie.lines!(axs[i],t, sol_p[i,:]; alpha=0.9, color=:black)
            end
            CairoMakie.lines!(axs[i],t, sol_p[i,:]; alpha=0.5, color=:grey)
        end
    end

    # Plot simulation and noisy observations.
    # plot mean and variance
    for i in 1:N
        # =-=----
        nonmissing = findall(mean_data[i,:] .!== missing)
        data_i = Float64.(mean_data[i,:][nonmissing])
        timepoints_i = Float64.(timepoints[nonmissing])
        var_data_i = Float64.(var_data[i,:][nonmissing])

        # skip if mean is empty
        if isempty(data_i)
            continue
        end

        indices = findall(x -> isnan(x),var_data_i)
        var_data_i[indices] .= 0
        CairoMakie.scatter!(axs[i], timepoints_i, data_i; color=RGB(0/255, 0/255, 139/255), alpha=1.)  
        # have lower std capped at 0.01 (to be visible in the plots)
        var_data_i_lower = copy(var_data_i)
        for (n,var) in enumerate(var_data_i)
            if sqrt(var) > data_i[n]
                var_data_i_lower[n] = max(data_i[n]^2-1e-5, 0)
                #var_data_i_lower[n] = data_i[n]^2
            end
        end

        #CairoMakie.errorbars!(axs[i], timepoints_i, data_i, sqrt.(var_data_i); color=RGB(0/255, 71/255, 171/255), whiskerwidth=20, alpha=0.2)
        CairoMakie.errorbars!(axs[i], timepoints_i, data_i, sqrt.(var_data_i_lower), sqrt.(var_data_i); color=RGB(0/255, 71/255, 171/255), whiskerwidth=20, alpha=0.2)
        #CairoMakie.errorbars!(axs[i], timepoints_i, data_i, [0. for _ in 1:length(timepoints_i)], sqrt.(var_data_i); color=RGB(0/255, 71/255, 171/255), whiskerwidth=20, alpha=0.2)
    end
    # plot all data points across all samples
    for i in 1:N
        jiggle = rand(Normal(0,0.01),size(data)[3])
        for k in axes(data,3)
            # =-=----
            nonmissing = findall(data[i,:,k] .!== missing)
            data_i = Float64.(data[i,:,k][nonmissing])
            timepoints_i = Float64.(timepoints[nonmissing] .+ jiggle[k])
            CairoMakie.scatter!(axs[i], timepoints_i, data_i; color=RGB(0/255, 71/255, 171/255), alpha=0.4)  
        end
        CairoMakie.save(save_path * "/retrodiction_region_$(i).png", fs[i])
    end

    # we're done
    return fs
end


"""
    plot_retrodiction_meanonly(inference; save_path=nothing, N_samples=200,
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
function plot_retrodiction2(inference; save_path=nothing, N_samples=200,
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
    lower_headroom_frac::Float64=0.03           # small visual headroom below clamp_floor
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
            CairoMakie.save(joinpath(save_path, "retrodiction_datachoice_region_$(i).png"), fs[i])
        end
    end

    # Link y-axes if we synced / fixed
    if !isnothing(global_ymax) || sync_y
        try
            Makie.linkyaxes!(axs...)
        catch
        end
    end

    return fs
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

#=
master plotting function (plot everything relevant to inference)
 =#
function plot_inference(inference, save_path; plotscale=log10, N_samples=300, show_variance=false)
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
    
    # plot
    predicted_observed(inference; save_path=save_path*"/predicted_observed", plotscale=plotscale);
    #plot_retrodiction(inference; save_path=save_path*"/retrodiction", N_samples=N_samples, show_variance=show_variance);
    plot_retrodiction2(inference; save_path=save_path*"/retrodiction",
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