using PathoSpread
using Serialization, Statistics
using PrettyTables, DataFrames
using Printf

# number of samples for WAIC
Sn = 300
group_cells = false  # weight each region x timepoint equally in WAIC

# Read inference results
simulations_list = [
    [
        "simulations/u0_DIFF_EUCL",
        "simulations/u0_DIFF_ANTERO",
        "simulations/u0_DIFF_RETRO",
        "simulations/u0_DIFF_BIDIR",
    ],                     
    [                      
        "simulations/DIFFG_EUCL",
        "simulations/DIFFG_ANTERO",
        "simulations/DIFFG_RETRO",
        "simulations/DIFFG_BIDIR",
    ],               
    [
        "simulations/DIFFGA_EUCL",
        "simulations/DIFFGA_ANTERO",
        "simulations/DIFFGA_RETRO",
        "simulations/DIFFGA_BIDIR",
    ],               
    [                
        "simulations/u0_DIFF_RETRO",
        "simulations/DIFFG_RETRO",
        "simulations/DIFFGA_RETRO",
    ],
    # FLUSH
    [                
        "simulations/FLUSH_DIFF_RETRO",
        "simulations/u0_DIFF_RETRO",
    ],          
    [
        "simulations/FLUSH_DIFFG_RETRO",
        "simulations/DIFFG_RETRO",
    ],
    [
        "simulations/FLUSH_DIFFGA_RETRO_C1",
        "simulations/DIFFGA_RETRO",
    ],

    # HIPPO
    #[
    #    "simulations/igs_hippo_DIFF_EUCL",
    #    "simulations/igs_hippo_DIFF_ANTERO",
    #    "simulations/igs_hippo_DIFF_RETRO",
    #    "simulations/igs_hippo_DIFF_BIDIR",
    #],
    #[
    #    "simulations/hippo_DIFFG_EUCL",
    #    "simulations/hippo_DIFFG_ANTERO",
    #    "simulations/hippo_DIFFG_RETRO",
    #    "simulations/hippo_DIFFG_BIDIR",
    #],
    #[
    #    "simulations/hippo_DIFFGA_EUCL",
    #    "simulations/hippo_DIFFGA_ANTERO",
    #    "simulations/hippo_DIFFGA_RETRO_CUT",
    #    "simulations/hippo_DIFFGA_BIDIR",
    #],
    [
        "simulations/hippo_DIFF_RETRO",
        "simulations/hippo_DIFFGA_RETRO_posterior_prior",
        "simulations/hippo_DIFFGA_RETRO_posterior_prior",
    ],
]

model_names_list = [
    [
        "euclidean",
        "anterograde",
        "retrograde",
        "bidirectional",
    ],
    [
        "euclidean",
        "anterograde",
        "retrograde",
        "bidirectional",
    ],
    [
        "euclidean",
        "anterograde",
        "retrograde",
        "bidirectional",
    ],
    [
        "DIFF",
        "DIFFG",
        "DIFFGA",
    ],
    [
        "DIFF flush",
        "DIFF retrograde",
    ],
    [
        "DIFFG flush",
        "DIFFG retrograde",
    ],
    [
        "DIFFGA flush",
        "DIFFGA retrograde",
    ],
    # HIPPO
    #[
    #    "euclidean",
    #    "anterograde",
    #    "retrograde",
    #    "bidirectional",
    #],
    #[
    #    "euclidean",
    #    "anterograde",
    #    "retrograde",
    #    "bidirectional",
    #],
    #[
    #    "euclidean",
    #    "anterograde",
    #    "retrograde",
    #    "bidirectional",
    #],
    [
        "DIFF" 
        "DIFFG" 
        "DIFFGA" 
    ],
]

fig_prefixes = [
    "striatum_DIFF",
    "striatum_DIFFG",
    "striatum_DIFFGA",
    "striatum_RETRO_ONLY",
    "striatum_FLUSH_DIFF",
    "striatum_FLUSH_DIFFG",
    "striatum_FLUSH_DIFFGA",
    #"hippo_DIFF",
    #"hippo_DIFFG",
    #"hippo_DIFFGA",
    "hippo_RETRO_ONLY",
]

fig_titles = [
    "DIFF",
    "DIFFG",
    "DIFFGA",
    "Retrograde transport",
    "",
    "",
    "",
    #"DIFF (hippocampal)",
    #"DIFFG (hippocampal)",
    #"DIFFGA (hippocampal)",
    "Retrograde transport (hippocampal)",
]

#model_names_list = [model_names_list[1]]
#simulations_list = [simulations_list[1]]
#fig_prefixes = [fig_prefixes[1]]

for (i,(simulations, model_names, prefix, fig_title)) in enumerate(zip(simulations_list, model_names_list, fig_prefixes, fig_titles))
    # file name
    fig_file = "figures/model_comparison/$(prefix)_WAIC.pdf"
    
    println("\n──────────────────────────────────────────────")
    println("🧩 Batch $i → $(fig_file)")
    for (m, s) in zip(model_names, simulations)
        println("  • $m → $s")
    end
    println("──────────────────────────────────────────────")


    # load inferences, skip if doesn't exist
    # --- Filter out missing simulations ---
    pairs = [(m, s) for (m, s) in zip(model_names, simulations) if isfile(s * ".jls")]

    if isempty(pairs)
        @warn "No valid inference files found for this batch; skipping."
        continue
    end

    # overwrite the same names
    unzip(v) = (getindex.(v, 1), getindex.(v, 2))
    model_names, simulations = unzip(pairs)

    # --- Load remaining inferences ---
    inferences = [load_inference(s * ".jls") for s in simulations]

    # Compute WAIC, AIC, BIC, MSE, and Frobenius covariance norm for models
    waic_vals = Float64[]
    se_waic_vals = Float64[]
    aic_vals  = Float64[]
    bic_vals  = Float64[]
    mse_vals  = Float64[]
    covnorm_vals = Float64[]
    regcov = []

    # needed for paired WAIC scores
    waic_i_list = Vector{Vector{Float64}}()
    n_useds     = Int[]
    pwaic_vals  = Float64[]   # optional: if you returned p_waic

    for inference in inferences
        waic, se_waic, waic_i, lppd, p_waic, n_used = compute_waic(inference; S=Sn, group_cells=group_cells, ignore_seed=true)
        push!(waic_vals, waic)
        push!(se_waic_vals, se_waic)
        push!(waic_i_list, waic_i)
        push!(n_useds, n_used)
        aic, bic = compute_aic_bic(inference)
        push!(aic_vals, aic)
        push!(bic_vals, bic)
        mse = compute_mse_mc(inference)
        push!(mse_vals, mse)
        regional_cov = compute_regional_correlations(inference)
        #covnorm = mean(abs.(regional_cov))  # avg |r|
        covnorm = mean((regional_cov).^2)  # avg R^2
        push!(covnorm_vals, covnorm)
        push!(regcov, regional_cov)
    end

    # Compute delta metrics relative to the best (lowest) value
    min_waic = minimum(waic_vals)
    min_aic  = minimum(aic_vals)
    min_bic  = minimum(bic_vals)
    min_mse  = minimum(mse_vals)

    # Handle the case where all covnorm_vals are NaN
    valid_cov = filter(!isnan, covnorm_vals)  # values are already ≥0; no need for abs here
    if isempty(valid_cov)
        min_cov   = NaN
        delta_cov = fill(NaN, length(covnorm_vals))
    else
        min_cov   = minimum(valid_cov)
        delta_cov = [c - min_cov for c in covnorm_vals]  # NaN stays NaN here automatically
    end

    delta_waic = [w - min_waic for w in waic_vals]
    delta_aic  = [a - min_aic for a in aic_vals]
    delta_bic  = [b - min_bic for b in bic_vals]
    delta_mse  = [m - min_mse for m in mse_vals]
    delta_cov  = [c - min_cov for c in covnorm_vals]

    # Build a DataFrame to display the results
    df = DataFrame(
        Model   = model_names,
        WAIC    = round.(waic_vals, digits=0),
        ∆WAIC   = round.(delta_waic, digits=0),
        AIC     = round.(aic_vals, digits=0),
        ∆AIC    = round.(delta_aic, digits=0),
        BIC     = round.(bic_vals, digits=0),
        ∆BIC    = round.(delta_bic, digits=0),
        MSE     = round.(mse_vals, digits=6),
        ∆MSE    = round.(delta_mse, digits=6),
        ParCor = round.(covnorm_vals, digits=4),
        ∆ParCor   = round.(delta_cov, digits=4)
    )
    display(df)

    # Print LaTeX table
    pretty_table(df; formatters = ft_printf("%5d"), backend = Val(:latex))

    # ─── reorder DataFrame so deltas come last ─────────────────────────────────────
    ordered = [
    :Model,
    :WAIC, :AIC, :BIC,      # main big metrics
    :MSE,  :ParCor,         # main small metrics
    :∆WAIC, :∆AIC, :∆BIC,   # deltas for the big metrics
    :∆MSE,  :∆ParCor        # deltas for the small metrics
    ]
    df2 = df[:, ordered]

    # ─── print LaTeX table with mixed formatting ───────────────────────────────────
    #pretty_table(
    #df2;
    #formatters = (
    #    ft_printf("%s",        1),     # Model (string)
    #    ft_printf("%7.0f",   2:4),     # WAIC, AIC, BIC  (zero‑decimal floats)
    #    ft_printf("%.2e",    5:6),     # MSE, ParCor     (sci‑notation 6 d.p.)
    #    ft_printf("%7.0f",   7:9),     # ∆WAIC, ∆AIC, ∆BIC (zero‑decimal floats)
    #    ft_printf("%.2e",     10),     # ∆MSE           (sci‑notation 6 d.p.)
    #    ft_printf("%.2e",     11)      # ∆ParCor        (sci‑notation 4 d.p.)
    #),
    #backend = Val(:latex),
    #)
    open("figures/model_comparison/$(prefix)_table.txt", "w") do io
        pretty_table(io, df2;
            formatters = (
                ft_printf("%s",        1),     # Model
                ft_printf("%7.0f",   2:4),     # WAIC, AIC, BIC
                ft_printf("%.2e",    5:6),     # MSE, ParCor
                ft_printf("%7.0f",   7:9),     # ΔWAIC, ΔAIC, ΔBIC
                ft_printf("%.2e",     10),     # ΔMSE
                ft_printf("%.2e",     11)      # ΔParCor
            ),
            backend = Val(:latex),
        )
    end



    # PAIRED WAIC COMPARISON
    # ─── Paired ΔWAIC ± SE(Δ) vs the best model ───────────────────────────────────
    # 1. FILTER OUT INVALID WAICs BEFORE ANYTHING ELSE
    valid_mask = .!([isnan(w) || isinf(w) for w in waic_vals])
    if !all(valid_mask)
        @warn "Ignoring models with Inf/NaN WAIC" model_names[.!valid_mask]
    end

    # Keep only valid models and their WAIC_i lists
    waic_vals   = waic_vals[valid_mask]
    waic_i_list = waic_i_list[valid_mask]
    model_names = model_names[valid_mask]

    # 2. CONTINUE WITH YOUR EXISTING CODE
    best_ix = argmin(waic_vals)
    ref_waic_i = waic_i_list[best_ix]
    n = length(ref_waic_i)
    @assert all(length(wi) == n for wi in waic_i_list)

    delta_waic_paired = Float64[]
    se_delta_waic     = Float64[]
    for j in eachindex(waic_i_list)
        if j == best_ix
            push!(delta_waic_paired, 0.0)
            push!(se_delta_waic, 0.0)
        else
            d_i = waic_i_list[j] .- ref_waic_i
            mask = .!(isnan.(d_i) .| isinf.(d_i))
            if count(mask) == 0
                @warn "Model $(model_names[j]) has all invalid ΔWAIC_i; skipping"
                push!(delta_waic_paired, NaN)
                push!(se_delta_waic, NaN)
                continue
            end
            d_i = d_i[mask]
            Δ   = sum(d_i)
            SEΔ = sqrt(length(d_i) * var(d_i))
            push!(delta_waic_paired, Δ)
            push!(se_delta_waic, SEΔ)
        end
    end


    # Build with simple column names, then set pretty headers in pretty_table
    df_pairs = DataFrame(
        Model = model_names,
        ΔWAIC_vs_best = round.(delta_waic_paired, digits=1),
        SE_ΔWAIC      = round.(se_delta_waic, digits=1),
    )


    pretty_table(
        df_pairs;
        header = ["Model", "ΔWAIC (vs best)", "SE(ΔWAIC)"],
        backend = Val(:latex),
    )


    # ─── PLOTTING PAIRED ΔWAIC ± 2SE ─────────────────────────────────────────────
    using CairoMakie, Statistics, Printf, Colors
    setup_plot_theme!()  # set plotting settings

    # classify with 95% bands
    low  = delta_waic_paired .- 2 .* se_delta_waic
    high = delta_waic_paired .+ 2 .* se_delta_waic
    best_ix = argmin(waic_vals)

    classes = map(1:length(model_names)) do i
        if i == best_ix
            :best
        elseif low[i] <= 0.0 <= high[i]
            :tied
        else
            :worse
        end
    end

    # aesthetics
    blue  = RGBf(0/255,71/255,171/255);
    red   = RGBf(185/255,40/255,40/255);
    grayc = RGBf(0.35,0.35,0.35);

    marker_for(c) = c===:best ? :star5 : :circle
    color_for(c)  = c===:best ? blue : (c===:tied ? grayc : red)

    # order rows by Δ (best first if ties exist)
    ord = sortperm(delta_waic_paired)
    ys  = collect(1:length(ord))

    fig = Figure(size=(1200, 300 + 30*length(model_names)), figure_padding = (25, 25, 25, 25));
    ax  = Axis(fig[1,1];
        title=fig_title,
        titlesize=38,
        xlabel="ΔWAIC",
        #xlabelsize=24,
        xlabelsize=35,
        xticklabelsize=32,
        yticklabelsize=35,
        yticks=(ys, model_names[ord]),
    )

    # reference at zero
    vlines!(ax, [0.0]; color=:grey, linestyle=:dash, linewidth=8, alpha=0.8)

    for (row, idx) in enumerate(ord)
        Δ   = delta_waic_paired[idx]
        SE2 = 2*se_delta_waic[idx]
        c   = classes[idx]
        y   = ys[row]

        # draw error bar for non-best only
        if c !== :best
            errorbars!(ax, [Δ], [y], [SE2], [SE2];
                direction=:x, whiskerwidth=25, linewidth=10, color=color_for(c), alpha=0.9)
        end

        # point (best = big blue star only)
        CairoMakie.scatter!(ax, [Δ], [y];
            markersize = c===:best ? 45 : 25,
            marker     = marker_for(c),
            color      = color_for(c),
            strokecolor=:black, strokewidth=1.2)

        # label "Δ ± 2SE" offset up & right (to avoid overlap)
        if c !== :best
            txt = @sprintf("%.0f ± %.0f", Δ, SE2)
            text!(ax, Δ, y;
                text=txt,
                align=(:left,:bottom),
                offset=(14, 10),          # ← right & up
                fontsize=30,
                color=:black,
                )
        end
    end

    # Add headroom inside the axis so top text isn’t cut off
    Makie.ylims!(ax, 0.5, length(model_names) + 0.9)

    # compute tight bounds from the drawn bars
    # OLD
    #xmax = maximum(delta_waic_paired .+ 2 .* se_delta_waic)
    #xmin = min(0.0, minimum(delta_waic_paired .- 2 .* se_delta_waic))
    ## add a sensible right pad (>= 50 units or 6% of span)
    #span = max(xmax - xmin, 1e-9)
    #pad  = max(0.1 * span, 50.0)
    #Makie.xlims!(ax, xmin-0.15*pad, xmax + pad)

    # NEW
    xmax = maximum(delta_waic_paired .+ 2 .* se_delta_waic)
    xmin = min(0.0, minimum(delta_waic_paired .- 2 .* se_delta_waic))
    txt      = @sprintf("%.0f ± %.0f", xmax, 2 * maximum(se_delta_waic))
    char_pad = 0.012 * length(txt) * (xmax - xmin)
    Makie.xlims!(ax, xmin-0.05*(xmax-xmin), xmax + char_pad)
    
    # save figure
    fig
    save(fig_file, fig)
end
