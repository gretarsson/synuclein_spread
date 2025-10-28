#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$HOME/synuclein_spread"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

# --------------------------------------------------
# Define base (model + args) jobs
# --------------------------------------------------
declare -A BASE_JOBS
# --------------------------------
# iCP injection site
# --------------------------------
# DIFF
BASE_JOBS["DIFF_RETRO"]="DIFF data/W_labeled_filtered.csv data/total_path.csv --retrograde=true --n_chains=1"
BASE_JOBS["DIFF_ANTERO"]="DIFF data/W_labeled_filtered.csv data/total_path.csv --retrograde=false --n_chains=1"
BASE_JOBS["DIFF_BIDIR"]="DIFF_bidirectional data/W_labeled_filtered.csv data/total_path.csv --n_chains=1"
BASE_JOBS["DIFF_EUCL"]="DIFF data/Euclidean_distance_matrix_filtered.csv data/total_path.csv --n_chains=1"

# DIFFG
BASE_JOBS["DIFFG_RETRO"]="DIFFG data/W_labeled_filtered.csv data/total_path.csv --retrograde=true --n_chains=1"
BASE_JOBS["DIFFG_ANTERO"]="DIFFG data/W_labeled_filtered.csv data/total_path.csv --retrograde=false --n_chains=1"
BASE_JOBS["DIFFG_BIDIR"]="DIFFG_bidirectional data/W_labeled_filtered.csv data/total_path.csv --n_chains=1"
BASE_JOBS["DIFFG_EUCL"]="DIFFG data/Euclidean_distance_matrix_filtered.csv data/total_path.csv --n_chains=1"

# DIFFGA
BASE_JOBS["DIFFGA_RETRO"]="DIFFGA data/W_labeled_filtered.csv data/total_path.csv --retrograde=true --n_chains=1"
BASE_JOBS["DIFFGA_ANTERO"]="DIFFGA data/W_labeled_filtered.csv data/total_path.csv --retrograde=false --n_chains=1"
BASE_JOBS["DIFFGA_BIDIR"]="DIFFGA_bidirectional data/W_labeled_filtered.csv data/total_path.csv --n_chains=1"
BASE_JOBS["DIFFGA_EUCL"]="DIFFGA data/Euclidean_distance_matrix_filtered.csv data/total_path.csv --n_chains=1"

# DIFFGAM
BASE_JOBS["DIFFGAM_RETRO"]="DIFFGAM data/W_labeled_filtered.csv data/total_path.csv --retrograde=true --n_chains=1"
BASE_JOBS["DIFFGAM_ANTERO"]="DIFFGAM data/W_labeled_filtered.csv data/total_path.csv --retrograde=false --n_chains=1"
BASE_JOBS["DIFFGAM_BIDIR"]="DIFFGAM_bidirectional data/W_labeled_filtered.csv data/total_path.csv --n_chains=1"
BASE_JOBS["DIFFGAM_EUCL"]="DIFFGAM data/Euclidean_distance_matrix_filtered.csv data/total_path.csv --n_chains=1"

# HELD OUT LAST TIME POINTS
BASE_JOBS["DIFF_T1"]="DIFF data/W_labeled_filtered.csv data/total_path.csv --retrograde=true --holdout_last=1"
BASE_JOBS["DIFF_T2"]="DIFF data/W_labeled_filtered.csv data/total_path.csv --retrograde=true --holdout_last=2"
BASE_JOBS["DIFF_T3"]="DIFF data/W_labeled_filtered.csv data/total_path.csv --retrograde=true --holdout_last=3"

BASE_JOBS["DIFFG_T1"]="DIFFG data/W_labeled_filtered.csv data/total_path.csv --retrograde=true --holdout_last=1"
BASE_JOBS["DIFFG_T2"]="DIFFG data/W_labeled_filtered.csv data/total_path.csv --retrograde=true --holdout_last=2"
BASE_JOBS["DIFFG_T3"]="DIFFG data/W_labeled_filtered.csv data/total_path.csv --retrograde=true --holdout_last=3"

BASE_JOBS["DIFFGA_T1"]="DIFFGA data/W_labeled_filtered.csv data/total_path.csv --retrograde=true --holdout_last=1"
BASE_JOBS["DIFFGA_T2"]="DIFFGA data/W_labeled_filtered.csv data/total_path.csv --retrograde=true --holdout_last=2"
BASE_JOBS["DIFFGA_T3"]="DIFFGA data/W_labeled_filtered.csv data/total_path.csv --retrograde=true --holdout_last=3"

BASE_JOBS["DIFFGAM_T1"]="DIFFGAM data/W_labeled_filtered.csv data/total_path.csv --retrograde=true --holdout_last=1"
BASE_JOBS["DIFFGAM_T2"]="DIFFGAM data/W_labeled_filtered.csv data/total_path.csv --retrograde=true --holdout_last=2"
BASE_JOBS["DIFFGAM_T3"]="DIFFGAM data/W_labeled_filtered.csv data/total_path.csv --retrograde=true --holdout_last=3"
# --------------------------------
# HIPPOCAMPUS INJECTION SITE
# --------------------------------
# DIFF
BASE_JOBS["hippo_DIFF_RETRO"]="DIFF data/W_labeled_filtered.csv data/hippocampal/hippocampal_syn_only.csv --seed_indices='[53,55,56]' --retrograde=true --n_chains=1"
BASE_JOBS["hippo_DIFF_ANTERO"]="DIFF data/W_labeled_filtered.csv data/hippocampal/hippocampal_syn_only.csv  --seed_indices='[53,55,56]' --retrograde=false --n_chains=1"
BASE_JOBS["hippo_DIFF_BIDIR"]="DIFF_bidirectional data/W_labeled_filtered.csv data/hippocampal/hippocampal_syn_only.csv  --seed_indices='[53,55,56]' --n_chains=1"
BASE_JOBS["hippo_DIFF_EUCL"]="DIFF data/Euclidean_distance_matrix_filtered.csv data/hippocampal/hippocampal_syn_only.csv --seed_indices='[53,55,56]' --n_chains=1"

# DIFFG
BASE_JOBS["hippo_DIFFG_RETRO"]="DIFFG data/W_labeled_filtered.csv data/hippocampal/hippocampal_syn_only.csv --seed_indices='[53,55,56]' --retrograde=true --n_chains=1"
BASE_JOBS["hippo_DIFFG_ANTERO"]="DIFFG data/W_labeled_filtered.csv data/hippocampal/hippocampal_syn_only.csv --seed_indices='[53,55,56]' --retrograde=false --n_chains=1"
BASE_JOBS["hippo_DIFFG_BIDIR"]="DIFFG_bidirectional data/W_labeled_filtered.csv data/hippocampal/hippocampal_syn_only.csv --seed_indices='[53,55,56]' --n_chains=1"
BASE_JOBS["hippo_DIFFG_EUCL"]="DIFFG data/Euclidean_distance_matrix_filtered.csv data/hippocampal/hippocampal_syn_only.csv --seed_indices='[53,55,56]' --n_chains=1"

# DIFFGA
BASE_JOBS["hippo_DIFFGA_RETRO"]="DIFFGA data/W_labeled_filtered.csv data/hippocampal/hippocampal_syn_only.csv --seed_indices='[53,55,56]' --retrograde=true --n_chains=1"
BASE_JOBS["hippo_DIFFGA_ANTERO"]="DIFFGA data/W_labeled_filtered.csv data/hippocampal/hippocampal_syn_only.csv --seed_indices='[53,55,56]' --retrograde=false --n_chains=1"
BASE_JOBS["hippo_DIFFGA_BIDIR"]="DIFFGA_bidirectional data/W_labeled_filtered.csv data/hippocampal/hippocampal_syn_only.csv --seed_indices='[53,55,56]' --n_chains=1"
BASE_JOBS["hippo_DIFFGA_EUCL"]="DIFFGA data/Euclidean_distance_matrix_filtered.csv data/hippocampal/hippocampal_syn_only.csv --seed_indices='[53,55,56]' --n_chains=1"

# DIFFGAM
BASE_JOBS["hippo_DIFFGAM_RETRO"]="DIFFGAM data/W_labeled_filtered.csv data/hippocampal/hippocampal_syn_only.csv --seed_indices='[53,55,56]' --retrograde=true --n_chains=1"
BASE_JOBS["hippo_DIFFGAM_ANTERO"]="DIFFGAM data/W_labeled_filtered.csv data/hippocampal/hippocampal_syn_only.csv --seed_indices='[53,55,56]' --retrograde=false --n_chains=1"
BASE_JOBS["hippo_DIFFGAM_BIDIR"]="DIFFGAM_bidirectional data/W_labeled_filtered.csv data/hippocampal/hippocampal_syn_only.csv --seed_indices='[53,55,56]' --n_chains=1"
BASE_JOBS["hippo_DIFFGAM_EUCL"]="DIFFGAM data/Euclidean_distance_matrix_filtered.csv data/hippocampal/hippocampal_syn_only.csv --seed_indices='[53,55,56]' --n_chains=1"

# HELD OUT LAST TIME POINTS
BASE_JOBS["hippo_DIFF_T1"]="DIFF data/W_labeled_filtered.csv data/hippocampal/hippocampal_syn_only.csv --seed_indices='[53,55,56]' --retrograde=true --holdout_last=1"
BASE_JOBS["hippo_DIFF_T2"]="DIFF data/W_labeled_filtered.csv data/hippocampal/hippocampal_syn_only.csv --seed_indices='[53,55,56]' --retrograde=true --holdout_last=2"
BASE_JOBS["hippo_DIFF_T3"]="DIFF data/W_labeled_filtered.csv data/hippocampal/hippocampal_syn_only.csv --seed_indices='[53,55,56]' --retrograde=true --holdout_last=3"

BASE_JOBS["hippo_DIFFG_T1"]="DIFFG data/W_labeled_filtered.csv data/hippocampal/hippocampal_syn_only.csv --seed_indices='[53,55,56]' --retrograde=true --holdout_last=1"
BASE_JOBS["hippo_DIFFG_T2"]="DIFFG data/W_labeled_filtered.csv data/hippocampal/hippocampal_syn_only.csv --seed_indices='[53,55,56]' --retrograde=true --holdout_last=2"
BASE_JOBS["hippo_DIFFG_T3"]="DIFFG data/W_labeled_filtered.csv data/hippocampal/hippocampal_syn_only.csv --seed_indices='[53,55,56]' --retrograde=true --holdout_last=3"

BASE_JOBS["hippo_DIFFGA_T1"]="DIFFGA data/W_labeled_filtered.csv data/hippocampal/hippocampal_syn_only.csv --seed_indices='[53,55,56]' --retrograde=true --holdout_last=1"
BASE_JOBS["hippo_DIFFGA_T2"]="DIFFGA data/W_labeled_filtered.csv data/hippocampal/hippocampal_syn_only.csv --seed_indices='[53,55,56]' --retrograde=true --holdout_last=2"
BASE_JOBS["hippo_DIFFGA_T3"]="DIFFGA data/W_labeled_filtered.csv data/hippocampal/hippocampal_syn_only.csv --seed_indices='[53,55,56]' --retrograde=true --holdout_last=3"

BASE_JOBS["hippo_DIFFGAM_T1"]="DIFFGAM data/W_labeled_filtered.csv data/hippocampal/hippocampal_syn_only.csv --seed_indices='[53,55,56]' --retrograde=true --holdout_last=1"
BASE_JOBS["hippo_DIFFGAM_T2"]="DIFFGAM data/W_labeled_filtered.csv data/hippocampal/hippocampal_syn_only.csv --seed_indices='[53,55,56]' --retrograde=true --holdout_last=2"
BASE_JOBS["hippo_DIFFGAM_T3"]="DIFFGAM data/W_labeled_filtered.csv data/hippocampal/hippocampal_syn_only.csv --seed_indices='[53,55,56]' --retrograde=true --holdout_last=3"

# HIPPO WITH PRIORS FROM STRIATUM
BASE_JOBS["hippo_DIFF_RETRO_posterior_prior"]="DIFF data/W_labeled_filtered.csv data/hippocampal/hippocampal_syn_only.csv --seed_indices='[53,55,56]' --retrograde=true --n_chains=1 --posterior_priors='simulations/DIFF_RETRO.jls'"
BASE_JOBS["hippo_DIFFG_RETRO_posterior_prior"]="DIFFG data/W_labeled_filtered.csv data/hippocampal/hippocampal_syn_only.csv --seed_indices='[53,55,56]' --retrograde=true --n_chains=1  --posterior_priors='simulations/DIFFG_RETRO.jls'"
BASE_JOBS["hippo_DIFFGA_RETRO_posterior_prior"]="DIFFGA data/W_labeled_filtered.csv data/hippocampal/hippocampal_syn_only.csv --seed_indices='[53,55,56]' --retrograde=true --n_chains=1  --posterior_priors='simulations/DIFFGA_RETRO.jls'"
BASE_JOBS["hippo_DIFFGAM_RETRO_posterior_prior"]="DIFFGAM data/W_labeled_filtered.csv data/hippocampal/hippocampal_syn_only.csv --seed_indices='[53,55,56]' --retrograde=true --n_chains=1  --posterior_priors='simulations/DIFFGAM_RETRO.jls'"

BASE_JOBS["hippo_DIFF_ANTERO_posterior_prior"]="DIFF data/W_labeled_filtered.csv data/hippocampal/hippocampal_syn_only.csv --seed_indices='[53,55,56]' --retrograde=false --n_chains=1 --posterior_priors='simulations/DIFF_ANTERO.jls'"
BASE_JOBS["hippo_DIFFG_ANTERO_posterior_prior"]="DIFFG data/W_labeled_filtered.csv data/hippocampal/hippocampal_syn_only.csv --seed_indices='[53,55,56]' --retrograde=false --n_chains=1  --posterior_priors='simulations/DIFFG_ANTERO.jls'"
BASE_JOBS["hippo_DIFFGA_ANTERO_posterior_prior"]="DIFFGA data/W_labeled_filtered.csv data/hippocampal/hippocampal_syn_only.csv --seed_indices='[53,55,56]' --retrograde=false --n_chains=1  --posterior_priors='simulations/DIFFGA_ANTERO.jls'"
BASE_JOBS["hippo_DIFFGAM_ANTERO_posterior_prior"]="DIFFGAM data/W_labeled_filtered.csv data/hippocampal/hippocampal_syn_only.csv --seed_indices='[53,55,56]' --retrograde=false --n_chains=1  --posterior_priors='simulations/DIFFGAM_ANTERO.jls'"

BASE_JOBS["hippo_DIFF_BIDIR_posterior_prior"]="DIFF_bidirectional data/W_labeled_filtered.csv data/hippocampal/hippocampal_syn_only.csv --seed_indices='[53,55,56]' --retrograde=false --n_chains=1 --posterior_priors='simulations/DIFF_BIDIR.jls'"
BASE_JOBS["hippo_DIFFG_BIDIR_posterior_prior"]="DIFFG_bidirectional data/W_labeled_filtered.csv data/hippocampal/hippocampal_syn_only.csv --seed_indices='[53,55,56]' --retrograde=false --n_chains=1  --posterior_priors='simulations/DIFFG_BIDIR.jls'"
BASE_JOBS["hippo_DIFFGA_BIDIR_posterior_prior"]="DIFFGA_bidirectional data/W_labeled_filtered.csv data/hippocampal/hippocampal_syn_only.csv --seed_indices='[53,55,56]' --retrograde=false --n_chains=1  --posterior_priors='simulations/DIFFGA_BIDIR.jls'"
BASE_JOBS["hippo_DIFFGAM_BIDIR_posterior_prior"]="DIFFGAM_bidirectional data/W_labeled_filtered.csv data/hippocampal/hippocampal_syn_only.csv --seed_indices='[53,55,56]' --retrograde=false --n_chains=1  --posterior_priors='simulations/DIFFGAM_BIDIR.jls'"

BASE_JOBS["hippo_DIFF_EUCL_posterior_prior"]="DIFF data/W_labeled_filtered.csv data/hippocampal/hippocampal_syn_only.csv --seed_indices='[53,55,56]' --retrograde=false --n_chains=1 --posterior_priors='simulations/DIFF_EUCL.jls'"
BASE_JOBS["hippo_DIFFG_EUCL_posterior_prior"]="DIFFG data/W_labeled_filtered.csv data/hippocampal/hippocampal_syn_only.csv --seed_indices='[53,55,56]' --retrograde=false --n_chains=1  --posterior_priors='simulations/DIFFG_EUCL.jls'"
BASE_JOBS["hippo_DIFFGA_EUCL_posterior_prior"]="DIFFGA data/W_labeled_filtered.csv data/hippocampal/hippocampal_syn_only.csv --seed_indices='[53,55,56]' --retrograde=false --n_chains=1  --posterior_priors='simulations/DIFFGA_EUCL.jls'"
BASE_JOBS["hippo_DIFFGAM_EUCL_posterior_prior"]="DIFFGAM data/W_labeled_filtered.csv data/hippocampal/hippocampal_syn_only.csv --seed_indices='[53,55,56]' --retrograde=false --n_chains=1  --posterior_priors='simulations/DIFFGAM_EUCL.jls'"

# --------------------------------------------------
# For each base job, submit 4 independent chains
# --------------------------------------------------
for JOBNAME in "${!BASE_JOBS[@]}"; do
  CMD_BASE="${BASE_JOBS[$JOBNAME]}"
  for CHAIN in {1..4}; do
    FULL_JOBNAME="${JOBNAME}_C${CHAIN}"
    OUT_FILE="simulations/${FULL_JOBNAME}.jls"

    # Skip if output file already exists
    if [[ -f "$OUT_FILE" ]]; then
      echo "Skipping $FULL_JOBNAME (output exists: $OUT_FILE)"
      continue
    fi

    echo "Submitting job: $FULL_JOBNAME"
    sbatch <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=$FULL_JOBNAME
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --partition=long
#SBATCH --time=72-00:00:00
#SBATCH --chdir=$PROJECT_DIR
#SBATCH --output=$LOG_DIR/${FULL_JOBNAME}-%j.out
#SBATCH --error=$LOG_DIR/${FULL_JOBNAME}-%j.err
#SBATCH --hint=nomultithread
#SBATCH --signal=B:USR1@300

set -euo pipefail
module purge
module load julia

# ---------- added fixes ----------
# Lift CPU-time cap if allowed
ulimit -t unlimited || true

# Force single-threaded math
export JULIA_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
# (Optional) confirm:
echo "[CPU limit (s)] \$(ulimit -t)"
# ---------------------------------

trap 'echo "[trap] SIGUSR1 received; attempting clean exit"; pkill -USR1 -P $$ || true' USR1

echo "[\$(date)] Launching Julia job $FULL_JOBNAME"
exec stdbuf -oL -eL julia --project=. scripts/infer_this_main.jl $CMD_BASE --n_chains=1 --out_file=$OUT_FILE
EOF

  done
done

echo "All jobs submitted. Use 'squeue -u \$USER' to monitor."


