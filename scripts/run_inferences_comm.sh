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
# DIFFG
BASE_JOBS["DIFFG_comm_in_RETRO"]="DIFFG_comm_in data/W_labeled_filtered.csv data/total_path.csv --retrograde=true --n_chains=1"
BASE_JOBS["DIFFG_comm_out_RETRO"]="DIFFG_comm_out data/W_labeled_filtered.csv data/total_path.csv --retrograde=true --n_chains=1"

# DIFFGA
BASE_JOBS["DIFFGA_comm_in_RETRO"]="DIFFGA_comm_in data/W_labeled_filtered.csv data/total_path.csv --retrograde=true --n_chains=1"
BASE_JOBS["DIFFGA_comm_out_RETRO"]="DIFFGA_comm_out data/W_labeled_filtered.csv data/total_path.csv --retrograde=true --n_chains=1"

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
#SBATCH --partition=all
#SBATCH --time=2-00:00:00
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


