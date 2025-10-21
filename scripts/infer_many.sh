#!/usr/bin/env bash
set -euo pipefail  # exit on first error

# DIFFG test
#julia --project=. scripts/infer_this_main.jl DIFF data/W_labeled_filtered.csv  data/hippocampal/hippocampal_syn_only.csv --retrograde=true --n_chains=1 --seed_indices='[53,54,56]' --out_file=simulations/DIFF_hippo.jls 
julia --project=. scripts/infer_this_main.jl DIFFGA data/W_labeled_filtered.csv  data/total_path.csv --retrograde=true --n_chains=1 --seed_indices='[74]' --out_file=simulations/DIFFGA.jls 

# run inference for all models with retrograde transport
#julia --project=. scripts/infer_this_main.jl DIFF  data/W_labeled_filtered.csv  data/total_path.csv --retrograde=true --n_chains=4 --out_file=DIFF_RETRO.jls
#julia --project=. scripts/infer_this_main.jl DIFFG data/W_labeled_filtered.csv  data/total_path.csv --retrograde=true --n_chains=4 --out_file=DIFFG_RETRO.jls
#julia --project=. scripts/infer_this_main.jl DIFFGA data/W_labeled_filtered.csv  data/total_path.csv --retrograde=true --n_chains=4 --out_file=DIFFGA_RETRO.jls
#julia --project=. scripts/infer_this_main.jl DIFFGAM data/W_labeled_filtered.csv  data/total_path.csv --retrograde=true --n_chains=4 --out_file=DIFFGAM_RETRO.jls
#
## ...with anterograde transport
#julia --project=. scripts/infer_this_main.jl DIFF  data/W_labeled_filtered.csv  data/total_path.csv --retrograde=false --n_chains=4 --out_file=DIFF_ANTERO.jls
#julia --project=. scripts/infer_this_main.jl DIFFG data/W_labeled_filtered.csv  data/total_path.csv --retrograde=false --n_chains=4 --out_file=DIFFG_ANTERO.jls
#julia --project=. scripts/infer_this_main.jl DIFFGA data/W_labeled_filtered.csv  data/total_path.csv --retrograde=false --n_chains=4 --out_file=DIFFGA_ANTERO.jls
#julia --project=. scripts/infer_this_main.jl DIFFGAM data/W_labeled_filtered.csv  data/total_path.csv --retrograde=false --n_chains=4 --out_file=DIFFGAM_ANTERO.jls
#
## ... and bidirectional (retrograde + anterograde)
#julia --project=. scripts/infer_this_main.jl DIFF_bidirectional  data/W_labeled_filtered.csv  data/total_path.csv --retrograde=true --n_chains=4 --out_file=DIFF_BIDIR.jls
#julia --project=. scripts/infer_this_main.jl DIFFG_bidirectional data/W_labeled_filtered.csv  data/total_path.csv --retrograde=true --n_chains=4 --out_file=DIFFG_BIDIR.jls
#julia --project=. scripts/infer_this_main.jl DIFFGA_bidirectional data/W_labeled_filtered.csv  data/total_path.csv --retrograde=true --n_chains=4 --out_file=DIFFGA_BIDIR.jls
#julia --project=. scripts/infer_this_main.jl DIFFGAM_bidirectional data/W_labeled_filtered.csv  data/total_path.csv --retrograde=true --n_chains=4 --out_file=DIFFGAM_BIDIR.jls
#
## ... and Euclidean transport
#julia --project=. scripts/infer_this_main.jl DIFF  data/Euclidean_distance_matrix_filtered.csv  data/total_path.csv --retrograde=true --n_chains=4 --out_file=DIFF_EUCL.jls
#julia --project=. scripts/infer_this_main.jl DIFFG data/Euclidean_distance_matrix_filtered.csv  data/total_path.csv --retrograde=true --n_chains=4 --out_file=DIFFG_EUCL.jls
#julia --project=. scripts/infer_this_main.jl DIFFGA data/Euclidean_distance_matrix_filtered.csv  data/total_path.csv --retrograde=true --n_chains=4 --out_file=DIFFGA_EUCL.jls
#julia --project=. scripts/infer_this_main.jl DIFFGAM data/Euclidean_distance_matrix_filtered.csv  data/total_path.csv --retrograde=true --n_chains=4 --out_file=DIFFGAM_EUCL.jls

# Hold out last time points
#julia --project=. scripts/infer_this_main.jl DIFFGA data/W_labeled_filtered.csv  data/total_path.csv --retrograde=true --n_chains=4 --out_file=DIFFGA_RETRO_T-1.jls --holdout_last=1
#julia --project=. scripts/infer_this_main.jl DIFFGA data/W_labeled_filtered.csv  data/total_path.csv --retrograde=true --n_chains=4 --out_file=DIFFGA_RETRO_T-2.jls --holdout_last=2
#julia --project=. scripts/infer_this_main.jl DIFFGA data/W_labeled_filtered.csv  data/total_path.csv --retrograde=true --n_chains=4 --out_file=DIFFGA_RETRO_T-3.jls --holdout_last=3
#
#julia --project=. scripts/infer_this_main.jl DIFFGAM data/W_labeled_filtered.csv  data/total_path.csv --retrograde=true --n_chains=4 --out_file=DIFFGAM_RETRO_T-1.jls --holdout_last=1
#julia --project=. scripts/infer_this_main.jl DIFFGAM data/W_labeled_filtered.csv  data/total_path.csv --retrograde=true --n_chains=4 --out_file=DIFFGAM_RETRO_T-2.jls --holdout_last=2
#julia --project=. scripts/infer_this_main.jl DIFFGAM data/W_labeled_filtered.csv  data/total_path.csv --retrograde=true --n_chains=4 --out_file=DIFFGAM_RETRO_T-3.jls --holdout_last=3