#!/usr/bin/env bash
set -e  # exit on first error

# run inference for all models with retrograde transport
julia --project=. scripts/infer_this_main.jl DIFF  data/W_labeled.csv  data/total_path.csv --retrograde=true --n_chains=4 --out_file=DIFF_RETRO.jl
julia --project=. scripts/infer_this_main.jl DIFFG data/W_labeled.csv  data/total_path.csv --retrograde=true --n_chains=4 --out_file=DIFFG_RETRO.jl
julia --project=. scripts/infer_this_main.jl DIFFGA data/W_labeled.csv  data/total_path.csv --retrograde=true --n_chains=4 --out_file=DIFFGA_RETRO.jl
julia --project=. scripts/infer_this_main.jl DIFFGAM data/W_labeled.csv  data/total_path.csv --retrograde=true --n_chains=4 --out_file=DIFFGAM_RETRO.jl

# ...with anterograde transport
julia --project=. scripts/infer_this_main.jl DIFF  data/W_labeled.csv  data/total_path.csv --retrograde=false --n_chains=4 --out_file=DIFF_ANTERO.jl
julia --project=. scripts/infer_this_main.jl DIFFG data/W_labeled.csv  data/total_path.csv --retrograde=false --n_chains=4 --out_file=DIFFG_ANTERO.jl
julia --project=. scripts/infer_this_main.jl DIFFGA data/W_labeled.csv  data/total_path.csv --retrograde=false --n_chains=4 --out_file=DIFFGA_ANTERO.jl
julia --project=. scripts/infer_this_main.jl DIFFGAM data/W_labeled.csv  data/total_path.csv --retrograde=false --n_chains=4 --out_file=DIFFGAM_ANTERO.jl
