using PathoSpread

# Load inferences (adjust paths if your filenames differ)
inf_T3   = load_inference("simulations/DIFFGA_RETRO_T-3.jls")
inf_T2   = load_inference("simulations/DIFFGA_RETRO_T-2.jls")
inf_T1   = load_inference("simulations/DIFFGA_RETRO_T-1.jls")
inf_Full = load_inference("simulations/DIFFGA_RETRO.jls")

infs  = [inf_T3, inf_T2, inf_T1, inf_Full]
names = ["T-3", "T-2", "T-1", "Full"]

# 1) Global parameters (few of them)
PathoSpread.plot_global_posterior_slope(infs, names; save_path="figures/posterior_comparison")

# 2) Local families of interest
bases = ["beta", "gamma"]

# (a) Heatmaps of effect-size shifts across adjacent truncations up to Full
#     Pairs: T-3→T-2, T-2→T-1, T-1→Full
for base in bases
    PathoSpread.plot_local_delta_heatmap(
        infs, names;
        base=base,
        compare_pairs=[(1,2), (2,3), (3,4)],
        order=:pathlength,      # or :by_median_delta
        cap=2.0,
        save_path="figures/posterior_comparison"
    )
end

# (b) Stability scatters vs Full (each truncated fit against Full)
for base in bases
    PathoSpread.plot_local_scatter_vs_full(
        inf_Full, inf_T1; base=base,
        other_name="T-1", full_name="Full", alpha=0.25,
        save_path="figures/posterior_comparison"
    )
    PathoSpread.plot_local_scatter_vs_full(
        inf_Full, inf_T2; base=base,
        other_name="T-2", full_name="Full", alpha=0.25,
        save_path="figures/posterior_comparison"
    )
    PathoSpread.plot_local_scatter_vs_full(
        inf_Full, inf_T3; base=base,
        other_name="T-3", full_name="Full", alpha=0.25,
        save_path="figures/posterior_comparison"
    )
end
