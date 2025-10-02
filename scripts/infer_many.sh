#!/usr/bin/env bash
set -e  # exit on first error

# run inference for all models with retrograde transport
#julia --project=. scripts/infer_this_main.jl DIFF  data/W_labeled_filtered.csv  data/total_path.csv --retrograde=true --n_chains=4 --out_file=DIFF_RETRO.jl
#julia --project=. scripts/infer_this_main.jl DIFFG data/W_labeled_filtered.csv  data/total_path.csv --retrograde=true --n_chains=4 --out_file=DIFFG_RETRO.jl
#julia --project=. scripts/infer_this_main.jl DIFFGA data/W_labeled_filtered.csv  data/total_path.csv --retrograde=true --n_chains=4 --out_file=DIFFGA_RETRO.jl
#julia --project=. scripts/infer_this_main.jl DIFFGAM data/W_labeled_filtered.csv  data/total_path.csv --retrograde=true --n_chains=4 --out_file=DIFFGAM_RETRO.jl
#
## ...with anterograde transport
#julia --project=. scripts/infer_this_main.jl DIFF  data/W_labeled_filtered.csv  data/total_path.csv --retrograde=false --n_chains=4 --out_file=DIFF_ANTERO.jl
#julia --project=. scripts/infer_this_main.jl DIFFG data/W_labeled_filtered.csv  data/total_path.csv --retrograde=false --n_chains=4 --out_file=DIFFG_ANTERO.jl
#julia --project=. scripts/infer_this_main.jl DIFFGA data/W_labeled_filtered.csv  data/total_path.csv --retrograde=false --n_chains=4 --out_file=DIFFGA_ANTERO.jl
#julia --project=. scripts/infer_this_main.jl DIFFGAM data/W_labeled_filtered.csv  data/total_path.csv --retrograde=false --n_chains=4 --out_file=DIFFGAM_ANTERO.jl
#
## ... and bidirectional (retrograde + anterograde)
#julia --project=. scripts/infer_this_main.jl DIFF_bidirectional  data/W_labeled_filtered.csv  data/total_path.csv --retrograde=true --n_chains=4 --out_file=DIFF_BIDIR.jl
#julia --project=. scripts/infer_this_main.jl DIFFG_bidirectional data/W_labeled_filtered.csv  data/total_path.csv --retrograde=true --n_chains=4 --out_file=DIFFG_BIDIR.jl
#julia --project=. scripts/infer_this_main.jl DIFFGA_bidirectional data/W_labeled_filtered.csv  data/total_path.csv --retrograde=true --n_chains=4 --out_file=DIFFGA_BIDIR.jl
#julia --project=. scripts/infer_this_main.jl DIFFGAM_bidirectional data/W_labeled_filtered.csv  data/total_path.csv --retrograde=true --n_chains=4 --out_file=DIFFGAM_BIDIR.jl
#
## ... and Euclidean transport
#julia --project=. scripts/infer_this_main.jl DIFF  data/Euclidean_distance_matrix_filtered.csv  data/total_path.csv --retrograde=true --n_chains=4 --out_file=DIFF_EUCL.jl
#julia --project=. scripts/infer_this_main.jl DIFFG data/Euclidean_distance_matrix_filtered.csv  data/total_path.csv --retrograde=true --n_chains=4 --out_file=DIFFG_EUCL.jl
#julia --project=. scripts/infer_this_main.jl DIFFGA data/Euclidean_distance_matrix_filtered.csv  data/total_path.csv --retrograde=true --n_chains=4 --out_file=DIFFGA_EUCL.jl
#julia --project=. scripts/infer_this_main.jl DIFFGAM data/Euclidean_distance_matrix_filtered.csv  data/total_path.csv --retrograde=true --n_chains=4 --out_file=DIFFGAM_EUCL.jl

# Hold out last time points
julia --project=. scripts/infer_this_main.jl DIFF data/W_labeled_filtered.csv  data/total_path.csv --retrograde=true --n_chains=4 --out_file=DIFF_RETRO_T-1.jls --holdout_last=1
julia --project=. scripts/infer_this_main.jl DIFF data/W_labeled_filtered.csv  data/total_path.csv --retrograde=true --n_chains=4 --out_file=DIFF_RETRO_T-2.jls --holdout_last=2
julia --project=. scripts/infer_this_main.jl DIFF data/W_labeled_filtered.csv  data/total_path.csv --retrograde=true --n_chains=4 --out_file=DIFF_RETRO_T-3.jls --holdout_last=3

julia --project=. scripts/infer_this_main.jl DIFFG data/W_labeled_filtered.csv  data/total_path.csv --retrograde=true --n_chains=4 --out_file=DIFFG_RETRO_T-1.jls --holdout_last=1
julia --project=. scripts/infer_this_main.jl DIFFG data/W_labeled_filtered.csv  data/total_path.csv --retrograde=true --n_chains=4 --out_file=DIFFG_RETRO_T-2.jls --holdout_last=2
julia --project=. scripts/infer_this_main.jl DIFFG data/W_labeled_filtered.csv  data/total_path.csv --retrograde=true --n_chains=4 --out_file=DIFFG_RETRO_T-3.jls --holdout_last=3

#julia --project=. scripts/infer_this_main.jl DIFFGA data/W_labeled_filtered.csv  data/total_path.csv --retrograde=true --n_chains=4 --out_file=DIFFGA_RETRO_T-1.jl --holdout_last=1
#julia --project=. scripts/infer_this_main.jl DIFFGA data/W_labeled_filtered.csv  data/total_path.csv --retrograde=true --n_chains=4 --out_file=DIFFGA_RETRO_T-2.jl --holdout_last=2
#julia --project=. scripts/infer_this_main.jl DIFFGA data/W_labeled_filtered.csv  data/total_path.csv --retrograde=true --n_chains=4 --out_file=DIFFGA_RETRO_T-3.jl --holdout_last=3
#
#julia --project=. scripts/infer_this_main.jl DIFFGAM data/W_labeled_filtered.csv  data/total_path.csv --retrograde=true --n_chains=4 --out_file=DIFFGAM_RETRO_T-1.jl --holdout_last=1
#julia --project=. scripts/infer_this_main.jl DIFFGAM data/W_labeled_filtered.csv  data/total_path.csv --retrograde=true --n_chains=4 --out_file=DIFFGAM_RETRO_T-2.jl --holdout_last=2
#julia --project=. scripts/infer_this_main.jl DIFFGAM data/W_labeled_filtered.csv  data/total_path.csv --retrograde=true --n_chains=4 --out_file=DIFFGAM_RETRO_T-3.jl --holdout_last=3