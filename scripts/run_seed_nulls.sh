#!/usr/bin/env bash
#SBATCH --job-name=ALL_SEEDS
#SBATCH --array=1-412%100          # run up to 100 seeds concurrently
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --partition=long
#SBATCH --time=72:00:00
#SBATCH --chdir=/cbica/home/alexanderc/synuclein_spread
#SBATCH --output=logs/seeds/%x_%A_%a.out
#SBATCH --error=logs/seeds/%x_%A_%a.err
#SBATCH --hint=nomultithread
#SBATCH --signal=B:USR1@300

# ============================================================
# Parallel inference runs for PathoSpread across all seed sites
# ============================================================

set -euo pipefail
module purge
module load julia

PROJECT_DIR="$HOME/synuclein_spread"
LOG_DIR="$PROJECT_DIR/logs/seeds"
mkdir -p "$LOG_DIR" "$PROJECT_DIR/simulations"

# -----------------------------
# Model configuration
# -----------------------------
MODEL="DIFFGA"              # <<< change model name here
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
# Per-seed parameters
# -----------------------------
i=$SLURM_ARRAY_TASK_ID
JOBNAME="${MODEL}_SEED_${i}"
OUT_FILE="simulations/${MODEL}_seed_${i}.jls"

echo "[ $(date) ] Starting seed inference $i"
echo "→ Model:  $MODEL"
echo "→ Seed index: $i"
echo "→ Output: $OUT_FILE"

# -----------------------------
# Run the Julia inference
# -----------------------------
stdbuf -oL -eL julia --project=. scripts/infer_this_main.jl \
    $MODEL $DATA_W $DATA_PATH \
    --retrograde=true \
    --n_chains=1 \
    --seed_index=$i \
    --infer_seed=true \
    --out_file=$OUT_FILE

echo "[ $(date) ] Completed seed inference $i"
