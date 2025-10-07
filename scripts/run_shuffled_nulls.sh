#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Launch parallel randomized null inference runs for all models
# ============================================================

MODELS=("DIFF" "DIFFG" "DIFFGA" "DIFFGAM")
N_NULLS=100        # number of random shuffles per model

for MODEL in "${MODELS[@]}"; do
  echo "[ $(date) ] Submitting randomized nulls for $MODEL"
  
  sbatch <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${MODEL}_NULLS
#SBATCH --array=1-${N_NULLS}%100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --partition=long
#SBATCH --time=21-00:00:00
#SBATCH --chdir=/cbica/home/alexanderc/synuclein_spread
#SBATCH --output=logs/nulls/%x_%A_%a.out
#SBATCH --error=logs/nulls/%x_%A_%a.err
#SBATCH --hint=nomultithread
#SBATCH --signal=B:USR1@300

set -euo pipefail
module purge
module load julia

PROJECT_DIR="\$HOME/synuclein_spread"
LOG_DIR="\$PROJECT_DIR/logs/nulls"
mkdir -p "\$LOG_DIR" "\$PROJECT_DIR/simulations"

# -----------------------------
# Environment hygiene
# -----------------------------
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export JULIA_NUM_THREADS=1

trap 'echo "[trap] SIGUSR1 received; attempting clean exit"; pkill -USR1 -P \$\$ || true' USR1

# -----------------------------
# Per-job parameters
# -----------------------------
i=\$SLURM_ARRAY_TASK_ID
OUT_FILE="simulations/${MODEL}_shuffle_\${i}.jls"

echo "[ \$(date) ] Starting randomized null run \$i for ${MODEL}"
stdbuf -oL -eL julia --project=. scripts/infer_this_main.jl \
    ${MODEL} data/W_labeled_filtered.csv data/total_path.csv \
    --retrograde=true \
    --n_chains=1 \
    --out_file=\$OUT_FILE \
    --shuffle

echo "[ \$(date) ] Completed randomized null run \$i for ${MODEL}"
EOF

done
