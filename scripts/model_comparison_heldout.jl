using Serialization
using PathoSpread
using PrettyTables, DataFrames, Statistics
using CairoMakie, Printf, Colors

# ------------------------------------------------------------------
# 0) Load your inferences (same as in your WAIC script)
# ------------------------------------------------------------------
simulations = [
    "simulations/DIFFGA_RETRO_T-1",
    "simulations/DIFFGAM_RETRO_T-1",
]
model_names = [
    "DIFFGA T-1",
    "DIFFGAM T-1",
]
inferences = [deserialize(sim * ".jls") for sim in simulations]

# get full data (needed for held out time points)
data_full, timepoints_full = PathoSpread.process_pathology("data/total_path.csv", W_csv="data/W_labeled_filtered.csv")

# ------------------------------------------------------------------
# 1) Compute held-out scores for each model
#    Expect fields: elpd_mean, elpd_i, crps_mean, crps_i, n_points
# ------------------------------------------------------------------
S_draws = 400  # posterior draws per point used to score
scores_list = [
    PathoSpread.compute_heldout_scores(inf;
        data_full       = data_full,          # R×T_full or R×T_full×K
        timepoints_full = timepoints_full,
        S               = S_draws
    ) for inf in inferences
]

# sanity: all models scored on same # held-out points
n_points = unique([s.n_points for s in scores_list])
@assert length(n_points) == 1
npts = n_points[1]

# ------------------------------------------------------------------
# 2) Summaries (per-point means ± SE) and deltas vs best
# ------------------------------------------------------------------
elpd_means = [s.elpd_mean for s in scores_list]   # per-point
crps_means = [s.crps_mean for s in scores_list]   # per-point
elpd_i_all = [s.elpd_i    for s in scores_list]   # vectors length npts
crps_i_all = [s.crps_i    for s in scores_list]

# standard errors of the per-point means
elpd_se = [std(v)/sqrt(length(v)) for v in elpd_i_all]
crps_se = [std(v)/sqrt(length(v)) for v in crps_i_all]

# identify best per metric (ELPD higher is better; CRPS lower is better)
best_elpd_ix = argmax(elpd_means)
best_crps_ix = argmin(crps_means)

delta_elpd = elpd_means .- maximum(elpd_means)  # ≤ 0 except best
delta_crps = crps_means .- minimum(crps_means)  # ≥ 0 except best

df = DataFrame(
    Model          = model_names,
    n_points       = fill(npts, length(model_names)),
    ELPD_mean      = elpd_means,
    ELPD_SE        = elpd_se,
    CRPS_mean_pp   = crps_means,      # percentage points
    CRPS_SE_pp     = crps_se,         # percentage points
    ΔELPD_to_best  = delta_elpd,
    ΔCRPS_to_best  = delta_crps,
)

# nice ordering: best first by each metric (choose one to sort by)
sort!(df, [:ΔCRPS_to_best, :Model])  # primarily by CRPS

# Print LaTeX (compact but informative)
pretty_table(
    df;
    header = ["Model","n","ELPD (per pt)","SE","CRPS (pp)","SE (pp)","ΔELPD","ΔCRPS"],
    formatters = (
        ft_printf("%s", 1),
        ft_printf("%d",  2),
        ft_printf("%.4f", 3),
        ft_printf("%.4f", 4),
        ft_printf("%.4f", 5),
        ft_printf("%.4f", 6),
        ft_printf("%+.4f", 9),
        ft_printf("%+.4f", 10),
    ),
    backend = Val(:latex),
)

# ------------------------------------------------------------------
# 3) Paired Δ vs best plots (like your ΔWAIC ± 2SE figure)
#    For ELPD: Δ = sum(elpd_i_model - elpd_i_best); higher is better
#    For CRPS: Δ = sum(crps_i_model - crps_i_best); lower is better
#    SE(Δ) = sqrt(n * var(d_i)) with d_i = per-point differences
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

ΔELPD, SEΔELPD = paired_delta_and_se(elpd_i_all, best_elpd_ix)
ΔCRPS, SEΔCRPS = paired_delta_and_se(crps_i_all, best_crps_ix)

# classify points (best/tied/worse) using 95% bands
function classify(Δ, SEΔ, best_ix; better=:higher)
    low  = Δ .- 2 .* SEΔ
    high = Δ .+ 2 .* SEΔ
    map(1:length(Δ)) do i
        if i == best_ix
            :best
        else
            # contains 0 ⇒ statistically tied at ~95%
            tied = (low[i] <= 0.0 <= high[i])
            if tied
                :tied
            else
                # sign direction depends on metric
                (better == :higher && Δ[i] > 0) || (better == :lower && Δ[i] < 0) ? :better : :worse
            end
        end
    end
end

classes_elpd = classify(ΔELPD, SEΔELPD, best_elpd_ix; better=:higher)
classes_crps = classify(ΔCRPS, SEΔCRPS, best_crps_ix; better=:lower)

# aesthetics
blue  = RGBf(0/255,71/255,171/255);
red   = RGBf(185/255,40/255,40/255);
grayc = RGBf(0.35,0.35,0.35);
green = RGBf(0.15,0.55,0.25);

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

# --- ΔELPD ± 2SE vs best (higher is better) ---
fig1 = Figure(resolution=(1100, 350 + 44length(model_names)), figure_padding = (20,20,20,20));
ax1  = Axis(fig1[1,1];
    title="Held-out ΔELPD ± 2·SE(Δ) vs best",
    titlesize=25, xlabel="ΔELPD (sum over held-out points)",
    xlabelsize=24, xticklabelsize=24, yticklabelsize=26,
    yticks=(1:length(model_names), model_names));
delta_plot!(ax1, ΔELPD, SEΔELPD, classes_elpd; colorsym=color_for_elpd)
save("figures/model_comparison/heldout_delta_elpd_vs_best.pdf", fig1)

# --- ΔCRPS ± 2SE vs best (lower is better; left is better) ---
fig2 = Figure(resolution=(1100, 350 + 44length(model_names)), figure_padding = (20,20,20,20));
ax2  = Axis(fig2[1,1];
    title="Held-out ΔCRPS ± 2·SE(Δ) vs best",
    titlesize=25, xlabel="ΔCRPS (sum over held-out points)",
    xlabelsize=24, xticklabelsize=24, yticklabelsize=26,
    yticks=(1:length(model_names), model_names));
delta_plot!(ax2, ΔCRPS, SEΔCRPS, classes_crps; colorsym=color_for_crps);
save("figures/model_comparison/heldout_delta_crps_vs_best.pdf", fig2)

println("\nTable and plots done for held-out data (n = $npts).")
