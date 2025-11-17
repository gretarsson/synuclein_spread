#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Launch randomized null inference runs only for missing shuffles
# ============================================================

MODELS=("DIFFGA")
N_NULLS=100        # number of random shuffles per model

for MODEL in "${MODELS[@]}"; do
  echo "[ $(date) ] Checking completed shuffled nulls for $MODEL"

  PROJECT_DIR="$HOME/synuclein_spread"
  SIM_DIR="$PROJECT_DIR/simulations"

  # --- identify missing shuffles ---
  missing=()
  for ((i=1; i<=N_NULLS; i++)); do
      out_file="$SIM_DIR/${MODEL}_shuffle_${i}.jls"
      if [[ ! -f "$out_file" ]]; then
          missing+=("$i")
      fi
  done

  if [[ ${#missing[@]} -eq 0 ]]; then
      echo "All ${N_NULLS} shuffled nulls completed for $MODEL. Skipping."
      continue
  fi

  echo "Found ${#missing[@]} missing shuffled nulls for $MODEL."

  # --- create temp list of indices ---
  idx_file=$(mktemp)
  printf "%s\n" "${missing[@]}" > "$idx_file"

  echo "[ $(date) ] Submitting remaining ${#missing[@]} shuffled nulls for $MODEL"

  sbatch <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${MODEL}_MISSING_NULLS
#SBATCH --array=1-${#missing[@]}%100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --partition=long
#SBATCH --time=5-00:00:00
#SBATCH --chdir=$PROJECT_DIR
#SBATCH --output=logs/nulls/%x_%A_%a.out
#SBATCH --error=logs/nulls/%x_%A_%a.err
#SBATCH --hint=nomultithread
#SBATCH --signal=B:USR1@300

set -euo pipefail
module purge
module load julia

ulimit -t unlimited || true

export JULIA_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

trap 'echo "[trap] SIGUSR1 received; attempting clean exit"; pkill -USR1 -P \$\$ || true' USR1

# Map SLURM_ARRAY_TASK_ID -> actual shuffle index
shuffle_list=($(cat $idx_file))
i=\${shuffle_list[\$((SLURM_ARRAY_TASK_ID-1))]}

OUT_FILE="simulations/${MODEL}_shuffle_\${i}.jls"
echo "[ \$(date) ] Starting randomized null run \$i for ${MODEL}"

exec stdbuf -oL -eL julia --project=. scripts/infer_this_main.jl \
    ${MODEL} data/W_labeled_filtered.csv data/total_path.csv \
    --retrograde=true \
    --n_chains=1 \
    --out_file=\$OUT_FILE \
    --shuffle
EOF

done
