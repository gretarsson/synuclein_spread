#!/usr/bin/env bash
set -euo pipefail  # exit on first error

# run inference!
#julia --project=. scripts/infer_this_main.jl DIFFGA data/W_labeled_filtered.csv  data/total_path.csv --retrograde=true --n_chains=1 --seed_indices="[74]" --out_file=simulations/test.jls --posterior_priors simulations/DIFFGA_RETRO.jls 
julia --project=. scripts/infer_this_main.jl DIFFGA data/W_labeled_filtered.csv data/hippocampal/hippocampal_syn_only.csv --seed_indices='[53,55,56]' --retrograde=true --n_chains=1  --posterior_priors='simulations/DIFFGA_RETRO.jls' --out_file=simulations/global_hippo_DIFFGA_RETRO_posterior_prior.jls
