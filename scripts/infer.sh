#!/usr/bin/env bash
set -euo pipefail  # exit on first error

# run inference!
julia --project=. scripts/infer_this_main.jl DIFFGA data/W_labeled_filtered.csv  data/total_path.csv --retrograde=true --n_chains=1 --seed_indices="[74]" --out_file=simulations/test.jls  --target_acceptance=0.9
