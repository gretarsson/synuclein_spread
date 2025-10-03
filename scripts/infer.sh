#!/usr/bin/env bash
set -euo pipefail

# Adjust to your cluster path
PROJECT_DIR="$HOME/synuclein_spread"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

# --- INTERACTIVE MODE: run with a real TTY (progress shows) ---
if [[ "${1:-}" == "-i" || "${1:-}" == "--interactive" ]]; then
  salloc -N1 -n1 -c1 --mem=32G -p all -t 24:00:00 --hint=nomultithread \
         --chdir="$PROJECT_DIR" \
    bash -lc '
      set -euo pipefail
      module purge
      module load julia
      echo "Julia: $(which julia)"
      julia -v

      export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 JULIA_NUM_THREADS=4

      echo "[${HOSTNAME}] starting Julia with a TTY (progress will render)"
      julia --project=. scripts/infer_this_main.jl \
        DIFFG data/W_labeled_filtered.csv data/total_path.csv \
        --retrograde=true --n_chains=1 --out_file=simulations/DIFFG_TEST.jls
    '
  exit
fi




sbatch <<'EOF'
#!/usr/bin/env bash
#SBATCH --job-name=DIFFG_TEST
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --partition=all
#SBATCH --time=24:00:00
#SBATCH --chdir=/cbica/home/alexanderc/synuclein_spread
#SBATCH --output=/cbica/home/alexanderc/synuclein_spread/logs/%x-%j.out
#SBATCH --error=/cbica/home/alexanderc/synuclein_spread/logs/%x-%j.err
#SBATCH --hint=nomultithread
#SBATCH --signal=B:USR1@300

set -euo pipefail

# Load julia on compute node
module purge
module load julia
echo "Julia: \$(which julia)"
julia -v

# reuse precompiled cache from home
#export JULIA_DEPOT_PATH="$HOME/.julia:$JULIA_DEPOT_PATH"
#export JULIA_PKG_PRECOMPILE_AUTO=0
# first run only
#julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile'

# Avoid BLAS oversubscription
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Let Julia use the cores you asked for
export JULIA_NUM_THREADS=4

# Graceful shutdown if cluster preempts job
trap 'echo "[trap] SIGUSR1 received; attempting clean exit"; pkill -USR1 -P $$ || true' USR1

echo "[`date`] Launching Julia job"
exec stdbuf -oL -eL julia --project=. scripts/infer_this_main.jl \
  DIFFG data/W_labeled_filtered.csv data/total_path.csv \
  --retrograde=true --n_chains=1 --out_file=simulations/DIFFG_TEST.jls
#exec script -q -e -f -c "julia --project=. scripts/test.jl" /dev/stdout 
EOF
