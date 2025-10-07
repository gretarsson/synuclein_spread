#!/usr/bin/env bash
set -euo pipefail

MODELS=("DIFFGA")
N_SEEDS=412

for MODEL in "${MODELS[@]}"; do
  echo "[ $(date) ] Submitting all seeds for $MODEL"
  
  sbatch <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${MODEL}_ALL_SEEDS
#SBATCH --array=1-${N_SEEDS}%100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1           # match single-threaded Julia
#SBATCH --mem=32G
#SBATCH --partition=all
#SBATCH --time=2-00:00:00
#SBATCH --chdir=/cbica/home/alexanderc/synuclein_spread
#SBATCH --output=logs/seeds/%x_%A_%a.out
#SBATCH --error=logs/seeds/%x_%A_%a.err
#SBATCH --hint=nomultithread
#SBATCH --signal=B:USR1@300

set -euo pipefail
module purge
module load julia

PROJECT_DIR="\$HOME/synuclein_spread"
LOG_DIR="\$PROJECT_DIR/logs/seeds"
mkdir -p "\$LOG_DIR" "\$PROJECT_DIR/simulations"

# -----------------------------
# Environment hygiene + CPU limit fix
# -----------------------------
# Lift per-process CPU-time cap (prevents 0:24 / 24:0)
ulimit -t unlimited || true

# Force single-threaded math
export JULIA_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Optional diagnostic
echo "[CPU limit (s)] \$(ulimit -t)"

trap 'echo "[trap] SIGUSR1 received; attempting clean exit"; pkill -USR1 -P \$\$ || true' USR1

# -----------------------------
# Per-job parameters
# -----------------------------
i=\$SLURM_ARRAY_TASK_ID
OUT_FILE="simulations/${MODEL}_seed_\${i}.jls"

echo "[ \$(date) ] Starting seed \$i for ${MODEL}"
exec stdbuf -oL -eL julia --project=. scripts/infer_this_main.jl \
    ${MODEL} data/W_labeled_filtered.csv data/total_path.csv \
    --retrograde=true \
    --n_chains=1 \
    --seed_index=\$i \
    --infer_seed=true \
    --out_file=\$OUT_FILE
EOF

done
