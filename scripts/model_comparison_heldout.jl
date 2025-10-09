using Serialization
using PathoSpread
using PrettyTables, DataFrames, Statistics
using CairoMakie, Printf, Colors

# -------------------- parameter for the add-on --------------------
const eps = 0.3   # threshold for PathoSpread.nonzero_regions

# ------------------------------------------------------------------
# 0) Load your inferences (same as in your WAIC script)
# ------------------------------------------------------------------
Tmins = [1,2,3]
for Tmin in Tmins
    display("Plotting for Tmin = $Tmin")
    simulations = [
        "simulations/DIFF_RETRO_T-$(Tmin)",
        "simulations/DIFFG_T-$(Tmin)",
        "simulations/DIFFGA_RETRO_T-$(Tmin)",
        "simulations/DIFFGAM_RETRO_T-$(Tmin)",
    ]
    model_names = [
        "DIFF T-$(Tmin)",
        "DIFFG T-$(Tmin)",
        "DIFFGA T-$(Tmin)",
        "DIFFGAM T-$(Tmin)",
    ]
    inferences = [deserialize(sim * ".jls") for sim in simulations]

    # get full data (needed for held out time points)
    data_full, timepoints_full = PathoSpread.process_pathology(
        "data/total_path.csv"; W_csv="data/W_labeled_filtered.csv"
    )
    # keep raw replicates; we aggregate only in scoring summaries

    # ------------------------------------------------------------------
    # 1) Compute held-out scores for each model
    # ------------------------------------------------------------------
    S_draws = 400
    scores_list = [
        PathoSpread.compute_heldout_scores(inf;
            data_full       = data_full,          # R×T×K (or R×T)
            timepoints_full = timepoints_full,
            S               = S_draws
        ) for inf in inferences
    ]

    # sanity: all models scored on same # held-out points
    n_points = unique([s.n_points for s in scores_list])
    @assert length(n_points) == 1
    npts = n_points[1]

    # ------------------------------------------------------------------
    # 2) Reconstruct (rs, ts) in the SAME order as compute_heldout_scores,
    #    then aggregate ELPD/CRPS per (region,time) before any averaging.
    # ------------------------------------------------------------------
    heldout_t = [j for (j,t) in enumerate(timepoints_full)
                 if !any(isapprox(t, τ; atol=1e-8, rtol=1e-8)
                         for τ in inferences[1]["timepoints"])]

    rs, ts = Int[], Int[]
    if ndims(data_full) == 3
        R, _, K = size(data_full)
        for r in 1:R, tj in heldout_t, k in 1:K
            y = data_full[r, tj, k]
            if !(ismissing(y))
                push!(rs, r); push!(ts, tj)
            end
        end
    else
        R, _ = size(data_full)
        for r in 1:R, tj in heldout_t
            y = data_full[r, tj]
            if !(ismissing(y))
                push!(rs, r); push!(ts, tj)
            end
        end
    end
    @assert length(rs) == npts == length(ts)

    function aggregate_by_region_time(values::Vector{Float64},
                                      rs::Vector{Int}, ts::Vector{Int})
        pairs = unique(zip(rs, ts))
        [mean(values[(rs .== r) .& (ts .== t)]) for (r, t) in pairs], pairs
    end

    elpd_i_all_bal = Vector{Vector{Float64}}(undef, length(scores_list))
    crps_i_all_bal = Vector{Vector{Float64}}(undef, length(scores_list))
    pairs_ref = nothing

    for (j, s) in enumerate(scores_list)
        elpd_bal, pairs = aggregate_by_region_time(s.elpd_i, rs, ts)
        crps_bal, _     = aggregate_by_region_time(s.crps_i, rs, ts)
        elpd_i_all_bal[j] = elpd_bal
        crps_i_all_bal[j] = crps_bal
        if pairs_ref === nothing
            pairs_ref = pairs   # Vector{Tuple{Int,Int}} of (region, time) cells
        end
    end
    n_cells = length(pairs_ref)

    # Balanced per-model means and SEs (equal weight per (region,time))
    elpd_means = [mean(v) for v in elpd_i_all_bal]
    crps_means = [mean(v) for v in crps_i_all_bal]
    elpd_se    = [std(v)/sqrt(length(v)) for v in elpd_i_all_bal]
    crps_se    = [std(v)/sqrt(length(v)) for v in crps_i_all_bal]

    # ------------------------------------------------------------------
    # 3) Ranking, deltas vs best, and table (balanced, ALL regions)
    # ------------------------------------------------------------------
    best_elpd_ix = argmax(elpd_means)      # higher is better
    best_crps_ix = argmin(crps_means)      # lower is better

    delta_elpd = elpd_means .- maximum(elpd_means)       # ≤ 0 except best
    delta_crps = crps_means .- minimum(crps_means)       # ≥ 0 except best

    df = DataFrame(
        Model          = model_names,
        n_cells        = fill(n_cells, length(model_names)),
        ELPD_mean      = elpd_means,
        ELPD_SE        = elpd_se,
        CRPS_mean      = crps_means,
        CRPS_SE        = crps_se,
        ΔELPD_to_best  = delta_elpd,
        ΔCRPS_to_best  = delta_crps,
    )
    sort!(df, [:ΔCRPS_to_best, :Model])

    pretty_table(
        df;
        header = ["Model","n_cells","ELPD (balanced)","SE","CRPS (balanced)","SE","ΔELPD","ΔCRPS"],
        formatters = (
            ft_printf("%s", 1),
            ft_printf("%d",  2),
            ft_printf("%.4f", 3),
            ft_printf("%.4f", 4),
            ft_printf("%.4f", 5),
            ft_printf("%.4f", 6),
            ft_printf("%+.4f", 7),
            ft_printf("%+.4f", 8),
        ),
        backend = Val(:latex),
    )

    # ------------------------------------------------------------------
    # 4) Δ vs best plots (balanced, ALL regions)
    # ------------------------------------------------------------------
    function paired_delta_and_se(vs::Vector{Vector{Float64}}, ref_ix::Int)
        ref = vs[ref_ix]
        @assert all(length(v) == length(ref) for v in vs)
        n = length(ref)
        Δ   = Float64[]
        SEΔ = Float64[]
        for (j, v) in enumerate(vs)
            if j == ref_ix
                push!(Δ, 0.0); push!(SEΔ, 0.0)
            else
                d  = v .- ref
                push!(Δ,  sum(d))
                push!(SEΔ, sqrt(n * var(d)))
            end
        end
        return Δ, SEΔ
    end

    ΔELPD, SEΔELPD = paired_delta_and_se(elpd_i_all_bal, best_elpd_ix)
    ΔCRPS, SEΔCRPS = paired_delta_and_se(crps_i_all_bal, best_crps_ix)

    function classify(Δ, SEΔ, best_ix; better=:higher)
        low  = Δ .- 2 .* SEΔ
        high = Δ .+ 2 .* SEΔ
        map(1:length(Δ)) do i
            if i == best_ix
                :best
            else
                tied = (low[i] <= 0.0 <= high[i])
                tied ? :tied :
                    ((better == :higher && Δ[i] > 0) || (better == :lower && Δ[i] < 0) ? :better : :worse)
            end
        end
    end

    classes_elpd = classify(ΔELPD, SEΔELPD, best_elpd_ix; better=:higher)
    classes_crps = classify(ΔCRPS, SEΔCRPS, best_crps_ix; better=:lower)

    blue  = RGBf(0/255,71/255,171/255);  red = RGBf(185/255,40/255,40/255)
    grayc = RGBf(0.35,0.35,0.35);        green = RGBf(0.15,0.55,0.25)

    marker_for(c) = c===:best ? :star5 : :circle
    color_for_elpd(c) = c===:best ? blue  : (c===:better ? green : (c===:tied ? grayc : red))
    color_for_crps(c) = c===:best ? blue  : (c===:better ? green : (c===:tied ? grayc : red))

    function delta_plot!(ax, Δ, SEΔ, classes; label_left="Δ", colorsym=color_for_elpd)
        ord = sortperm(Δ)
        ys  = collect(1:length(ord))
        for (row, idx) in enumerate(ord)
            Δi, se2, cls = Δ[idx], 2SEΔ[idx], classes[idx]
            y = ys[row]
            if cls !== :best
                errorbars!(ax, [Δi], [y], [se2], [se2];
                    direction=:x, whiskerwidth=25, linewidth=10, color=colorsym(cls), alpha=0.9)
            end
            CairoMakie.scatter!(ax, [Δi], [y];
                markersize = cls===:best ? 45 : 25,
                marker     = marker_for(cls),
                color      = colorsym(cls),
                strokecolor=:black, strokewidth=1.2)
            txt = @sprintf("%.0f ± %.0f", Δi, 2se2)
            text!(ax, Δi, y; text=txt, align=(:left,:bottom), offset=(14,10), fontsize=24, color=:black)
        end
        vlines!(ax, [0.0]; color=:gray, linestyle=:dash, linewidth=4)
        Makie.ylims!(ax, 0.5, length(model_names) + 0.5)
        xmax = maximum(Δ .+ 2 .* SEΔ); xmin = min(0.0, minimum(Δ .- 2 .* SEΔ))
        span = max(xmax - xmin, 1e-9); pad = max(0.2*span, 50.0)
        Makie.xlims!(ax, xmin - 0.1pad, xmax + pad)
        return nothing
    end

    # Plots (ALL regions, balanced)
    fig1 = Figure(resolution=(1100, 350 + 44length(model_names)), figure_padding = (20,20,20,20));
    ax1  = Axis(fig1[1,1];
        title="Held-out ΔELPD (balanced) ± 2·SE vs best",
        titlesize=25, xlabel="ΔELPD (sum over region×time cells)",
        xlabelsize=24, xticklabelsize=24, yticklabelsize=26,
        yticks=(1:length(model_names), model_names));
    delta_plot!(ax1, ΔELPD, SEΔELPD, classes_elpd; colorsym=color_for_elpd)
    save("figures/model_comparison/heldout/heldout_delta_elpd_vs_best_T-$(Tmin).pdf", fig1)

    fig2 = Figure(resolution=(1100, 350 + 44length(model_names)), figure_padding = (20,20,20,20));
    ax2  = Axis(fig2[1,1];
        title="Held-out ΔCRPS (balanced) ± 2·SE vs best",
        titlesize=25, xlabel="ΔCRPS (sum over region×time cells)",
        xlabelsize=24, xticklabelsize=24, yticklabelsize=26,
        yticks=(1:length(model_names), model_names));
    delta_plot!(ax2, ΔCRPS, SEΔCRPS, classes_crps; colorsym=color_for_crps);
    #save("figures/model_comparison/heldout/heldout_delta_crps_vs_best_T-$(Tmin).pdf", fig2)

    println("Balanced table/plots done (cells = $n_cells; raw n = $npts).")

    # ------------------------------------------------------------------
    # --- NONZERO ADD-ON: limit to regions with any mean > eps ----------
    # ------------------------------------------------------------------
    nz_regions = PathoSpread.nonzero_regions(data_full; eps=eps)
    # mask pairs_ref (Vector{Tuple{r,t}}) by region membership
    idx_keep = [r in nz_regions for (r, _) in pairs_ref]
    n_cells_nz = count(idx_keep)

    if n_cells_nz == 0
        @warn "No nonzero regions found at eps=$(eps). Skipping nonzero-only outputs."
        continue
    end

    # Filter balanced vectors
    elpd_i_all_nz = [v[idx_keep] for v in elpd_i_all_bal]
    crps_i_all_nz = [v[idx_keep] for v in crps_i_all_bal]

    # Means/SEs on the restricted set
    elpd_means_nz = [mean(v) for v in elpd_i_all_nz]
    crps_means_nz = [mean(v) for v in crps_i_all_nz]
    elpd_se_nz    = [std(v)/sqrt(length(v)) for v in elpd_i_all_nz]
    crps_se_nz    = [std(v)/sqrt(length(v)) for v in crps_i_all_nz]

    best_elpd_ix_nz = argmax(elpd_means_nz)
    best_crps_ix_nz = argmin(crps_means_nz)

    delta_elpd_nz = elpd_means_nz .- maximum(elpd_means_nz)
    delta_crps_nz = crps_means_nz .- minimum(crps_means_nz)

    df_nz = DataFrame(
        Model          = model_names,
        n_cells        = fill(n_cells_nz, length(model_names)),
        ELPD_mean      = elpd_means_nz,
        ELPD_SE        = elpd_se_nz,
        CRPS_mean      = crps_means_nz,
        CRPS_SE        = crps_se_nz,
        ΔELPD_to_best  = delta_elpd_nz,
        ΔCRPS_to_best  = delta_crps_nz,
    )
    sort!(df_nz, [:ΔCRPS_to_best, :Model])

    pretty_table(
        df_nz;
        header = ["Model","n_cells(>$(eps))","ELPD (bal, nz)","SE","CRPS (bal, nz)","SE","ΔELPD","ΔCRPS"],
        formatters = (
            ft_printf("%s", 1),
            ft_printf("%d",  2),
            ft_printf("%.4f", 3),
            ft_printf("%.4f", 4),
            ft_printf("%.4f", 5),
            ft_printf("%.4f", 6),
            ft_printf("%+.4f", 7),
            ft_printf("%+.4f", 8),
        ),
        backend = Val(:latex),
    )

    ΔELPD_nz, SEΔELPD_nz = paired_delta_and_se(elpd_i_all_nz, best_elpd_ix_nz)
    ΔCRPS_nz, SEΔCRPS_nz = paired_delta_and_se(crps_i_all_nz, best_crps_ix_nz)

    classes_elpd_nz = classify(ΔELPD_nz, SEΔELPD_nz, best_elpd_ix_nz; better=:higher)
    classes_crps_nz = classify(ΔCRPS_nz, SEΔCRPS_nz, best_crps_ix_nz; better=:lower)

    # Plots (NONZERO regions only, balanced)
    fig1nz = Figure(resolution=(1100, 350 + 44length(model_names)), figure_padding = (20,20,20,20));
    ax1nz  = Axis(fig1nz[1,1];
        title="Held-out ΔELPD (balanced, regions mean>$(eps)) ± 2·SE",
        titlesize=25, xlabel="ΔELPD (sum over region×time cells)",
        xlabelsize=24, xticklabelsize=24, yticklabelsize=26,
        yticks=(1:length(model_names), model_names));
    delta_plot!(ax1nz, ΔELPD_nz, SEΔELPD_nz, classes_elpd_nz; colorsym=color_for_elpd)
    #save("figures/model_comparison/heldout/heldout_delta_elpd_vs_best_T-$(Tmin)_nonzero_eps$(eps).pdf", fig1nz)

    fig2nz = Figure(resolution=(1100, 350 + 44length(model_names)), figure_padding = (20,20,20,20));
    ax2nz  = Axis(fig2nz[1,1];
        title="Held-out ΔCRPS (balanced, regions mean>$(eps)) ± 2·SE",
        titlesize=25, xlabel="ΔCRPS (sum over region×time cells)",
        xlabelsize=24, xticklabelsize=24, yticklabelsize=26,
        yticks=(1:length(model_names), model_names));
    delta_plot!(ax2nz, ΔCRPS_nz, SEΔCRPS_nz, classes_crps_nz; colorsym=color_for_crps)
    #save("figures/model_comparison/heldout/heldout_delta_crps_vs_best_T-$(Tmin)_nonzero_eps$(eps).pdf", fig2nz)

    println("Nonzero-only table/plots done (eps=$(eps); cells = $n_cells_nz).")


    # --- quick region printout (put this after rs/ts are built) ---
    r = 2  # region index you want to inspect

    function region_cell_stats(s, rs::Vector{Int}, ts::Vector{Int}, r::Int)
        idx_r = findall(rs .== r)
        if isempty(idx_r)
            return (n_cells=0, elpd_mean=NaN, elpd_sum=NaN, crps_mean=NaN, crps_sum=NaN)
        end
        times = unique(ts[idx_r])
        elpd_cells = [mean(s.elpd_i[(rs .== r) .& (ts .== t)]) for t in times]
        crps_cells = [mean(s.crps_i[(rs .== r) .& (ts .== t)]) for t in times]
        return (n_cells=length(times),
                elpd_mean=mean(elpd_cells), elpd_sum=sum(elpd_cells),
                crps_mean=mean(crps_cells), crps_sum=sum(crps_cells))
    end

    println("\n=== Region $r held-out scores (cell-balanced within region) ===")
    for (name, s) in zip(model_names, scores_list)
        st = region_cell_stats(s, rs, ts, r)
        @printf("%-12s | n_cells=%d  ELPD(mean)=%.4f  ELPD(sum)=%.2f  CRPS(mean)=%.4f  CRPS(sum)=%.2f\n",
                name, st.n_cells, st.elpd_mean, st.elpd_sum, st.crps_mean, st.crps_sum)
    end
    # --- per-region summaries and pairwise differences -----------------
    function region_cell_stat_vec(s, rs::Vector{Int}, ts::Vector{Int}; r::Int)
        idx_r = findall(rs .== r)
        if isempty(idx_r)
            return (n_cells=0, elpd=NaN, crps=NaN)
        end
        times = unique(ts[idx_r])
        elpd_cells = [mean(s.elpd_i[(rs .== r) .& (ts .== t)]) for t in times]
        crps_cells = [mean(s.crps_i[(rs .== r) .& (ts .== t)]) for t in times]
        return (n_cells=length(times), elpd=mean(elpd_cells), crps=mean(crps_cells))
    end

    R_total = size(data_full, 1)
    # table: one row per region, with balanced ELPD/CRPS per model
    rows = Vector{Any}()
    for r in 1:R_total
        stats = [region_cell_stat_vec(s, rs, ts; r=r) for s in scores_list]
        if all(st.n_cells == 0 for st in stats); continue; end
        push!(rows, (
            region = r,
            n_cells = stats[1].n_cells,    # same across models
            elpd_1 = stats[1].elpd, crps_1 = stats[1].crps,  # DIFF
            elpd_2 = stats[2].elpd, crps_2 = stats[2].crps,  # DIFFGA
            elpd_3 = stats[3].elpd, crps_3 = stats[3].crps   # DIFFGAM
        ))
    end

    using DataFrames
    regdf = DataFrame(rows)
    rename!(regdf, Dict(
        :elpd_1=>"ELPD_DIFF", :crps_1=>"CRPS_DIFF",
        :elpd_2=>"ELPD_DIFFGA", :crps_2=>"CRPS_DIFFGA",
        :elpd_3=>"ELPD_DIFFGAM", :crps_3=>"CRPS_DIFFGAM"
    ))

    # pairwise ELPD deltas (positive means right-hand model is better)
    regdf.:ΔELPD_DIFFGA_vs_DIFF  = regdf.ELPD_DIFFGA  .- regdf.ELPD_DIFF
    regdf.:ΔELPD_DIFFGAM_vs_DIFF = regdf.ELPD_DIFFGAM .- regdf.ELPD_DIFF
    regdf.:ΔELPD_DIFFGAM_vs_DIFFGA = regdf.ELPD_DIFFGAM .- regdf.ELPD_DIFFGA

    # quick overall counts
    println("\nPer-region ELPD advantage counts (held-out, balanced within region):")
    println("  DIFFGA > DIFF  : ", count(>(0), regdf.:ΔELPD_DIFFGA_vs_DIFF),  " / ", nrow(regdf))
    println("  DIFFGAM > DIFF : ", count(>(0), regdf.:ΔELPD_DIFFGAM_vs_DIFF), " / ", nrow(regdf))
    println("  DIFFGAM > DIFFGA: ", count(>(0), regdf.:ΔELPD_DIFFGAM_vs_DIFFGA), " / ", nrow(regdf))

    # Top/bottom regions by ΔELPD (DIFFGA vs DIFF)
    top10_gain = first(sort(regdf, :ΔELPD_DIFFGA_vs_DIFF, rev=true), 10)
    top10_loss = first(sort(regdf, :ΔELPD_DIFFGA_vs_DIFF, rev=false), 10)

    # Minimal columns that matter
    cols = [:region, :n_cells, :ELPD_DIFF, :ELPD_DIFFGA, :ELPD_DIFFGAM,
            :ΔELPD_DIFFGA_vs_DIFF, :ΔELPD_DIFFGAM_vs_DIFF, :ΔELPD_DIFFGAM_vs_DIFFGA]

    println("\nTop 10 regions where DIFFGA beats DIFF (by ELPD):")
    pretty_table(top10_gain[:, cols]; backend=Val(:text))

    println("\nTop 10 regions where DIFF beats DIFFGA (by ELPD):")
    pretty_table(top10_loss[:, cols]; backend=Val(:text))


end
