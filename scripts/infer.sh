#!/usr/bin/env bash
set -euo pipefail

# Adjust to your cluster path
PROJECT_DIR="$HOME/synuclein_spread"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

sbatch <<'EOF'
#!/usr/bin/env bash
#SBATCH --job-name=DIFFG_TEST
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=all
#SBATCH --time=24:00:00
#SBATCH --chdir=/home/yourusername/synuclein_spread
#SBATCH --output=/home/yourusername/synuclein_spread/logs/%x-%j.out
#SBATCH --error=/home/yourusername/synuclein_spread/logs/%x-%j.err
#SBATCH --hint=nomultithread
#SBATCH --signal=B:USR1@300

set -euo pipefail

# Avoid BLAS oversubscription
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Let Julia use the cores you asked for
export JULIA_NUM_THREADS=16

# Graceful shutdown if cluster preempts job
trap 'echo "[trap] SIGUSR1 received; attempting clean exit"; pkill -USR1 -P $$ || true' USR1

echo "[`date`] Launching Julia job"
exec julia --project=. scripts/infer_this_main.jl \
  DIFFG data/W_labeled_filtered.csv data/total_path.csv \
  --retrograde=true --n_chains=1 --out_file=simulations/DIFFG_TEST.jls
EOF
