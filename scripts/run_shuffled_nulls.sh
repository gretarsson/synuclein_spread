#!/usr/bin/env bash
#SBATCH --job-name=NULLS
#SBATCH --array=1-100%100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=all
#SBATCH --time=48:00:00
#SBATCH --chdir=/cbica/home/alexanderc/synuclein_spread
#SBATCH --output=logs/nulls/%x_%A_%a.out
#SBATCH --error=logs/nulls/%x_%A_%a.err
#SBATCH --hint=nomultithread
#SBATCH --signal=B:USR1@300

# ============================================================
# Parallel randomized null inference runs for PathoSpread models
# ============================================================

set -euo pipefail
module purge
module load julia

PROJECT_DIR="$HOME/synuclein_spread"
LOG_DIR="$PROJECT_DIR/logs/nulls"
mkdir -p "$LOG_DIR" "$PROJECT_DIR/simulations"

# -----------------------------
# Model configuration
# -----------------------------
MODEL="DIFFGA"         # <<< CHANGE MODEL NAME HERE
DATA_W="data/W_labeled_filtered.csv"
DATA_PATH="data/total_path.csv"

# -----------------------------
# Environment hygiene
# -----------------------------
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export JULIA_NUM_THREADS=1

trap 'echo "[trap] SIGUSR1 received; attempting clean exit"; pkill -USR1 -P $$ || true' USR1

# -----------------------------
# Per-job parameters
# -----------------------------
i=$SLURM_ARRAY_TASK_ID
JOBNAME="${MODEL}_NULL_${i}"
OUT_FILE="simulations/${MODEL}_shuffle_${i}.jls"

echo "[ $(date) ] Starting null inference $i"
echo "→ Model:  $MODEL"
echo "→ Output: $OUT_FILE"

# -----------------------------
# Run the Julia inference
# -----------------------------
stdbuf -oL -eL julia --project=. scripts/infer_this_main.jl \
    $MODEL $DATA_W $DATA_PATH \
    --retrograde=true \
    --n_chains=1 \
    --out_file=$OUT_FILE \
    --shuffle

echo "[ $(date) ] Completed null inference $i"
