#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Run batch plotting of merged inference results
# ============================================================
PROJECT_DIR="$HOME/synuclein_spread"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

JOBNAME="plot_all_inferences"
LOG_OUT="$LOG_DIR/${JOBNAME}-%j.out"
LOG_ERR="$LOG_DIR/${JOBNAME}-%j.err"

echo "Submitting job: $JOBNAME"

sbatch <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=$JOBNAME
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH --chdir=$PROJECT_DIR
#SBATCH --output=$LOG_OUT
#SBATCH --error=$LOG_ERR
#SBATCH --hint=nomultithread
#SBATCH --signal=B:USR1@300

set -euo pipefail
module purge
module load julia

# ---------- Environment hygiene ----------
ulimit -t unlimited || true
export JULIA_NUM_THREADS=2
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
echo "[CPU limit (s)] \$(ulimit -t)"
# ----------------------------------------

trap 'echo "[trap] SIGUSR1 received; attempting clean exit"; pkill -USR1 -P $$ || true' USR1

echo "[\$(date)] Launching Julia plotting job"
exec stdbuf -oL -eL julia --project=. scripts/analyze_inference_all.jl
EOF

echo "✅ Plotting job submitted. Use 'squeue -u \$USER' to monitor."
