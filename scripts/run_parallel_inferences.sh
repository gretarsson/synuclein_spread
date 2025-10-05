#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$HOME/synuclein_spread"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

# Define (job name, args) pairs
declare -A JOBS
JOBS["DIFF_T1"]="DIFF data/W_labeled_filtered.csv data/total_path.csv --retrograde=true --n_chains=4 --out_file=simulations/DIFF_RETRO_T-1.jls --holdout_last=1"
JOBS["DIFF_T2"]="DIFF data/W_labeled_filtered.csv data/total_path.csv --retrograde=true --n_chains=4 --out_file=simulations/DIFF_RETRO_T-2.jls --holdout_last=2"
JOBS["DIFF_T3"]="DIFF data/W_labeled_filtered.csv data/total_path.csv --retrograde=true --n_chains=4 --out_file=simulations/DIFF_RETRO_T-3.jls --holdout_last=3"
JOBS["DIFFG_T1"]="DIFFG data/W_labeled_filtered.csv data/total_path.csv --retrograde=true --n_chains=4 --out_file=simulations/DIFFG_RETRO_T-1.jls --holdout_last=1"
JOBS["DIFFG_T2"]="DIFFG data/W_labeled_filtered.csv data/total_path.csv --retrograde=true --n_chains=4 --out_file=simulations/DIFFG_RETRO_T-2.jls --holdout_last=2"
JOBS["DIFFG_T3"]="DIFFG data/W_labeled_filtered.csv data/total_path.csv --retrograde=true --n_chains=4 --out_file=simulations/DIFFG_RETRO_T-3.jls --holdout_last=3"

# Submit each job
for JOBNAME in "${!JOBS[@]}"; do
  CMD="${JOBS[$JOBNAME]}"
  echo "Submitting job: $JOBNAME"
  sbatch <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=$JOBNAME
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=all
#SBATCH --time=48:00:00
#SBATCH --chdir=$PROJECT_DIR
#SBATCH --output=$LOG_DIR/${JOBNAME}-%j.out
#SBATCH --error=$LOG_DIR/${JOBNAME}-%j.err
#SBATCH --hint=nomultithread
#SBATCH --signal=B:USR1@300

set -euo pipefail
module purge
module load julia

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export JULIA_NUM_THREADS=1

trap 'echo "[trap] SIGUSR1 received; attempting clean exit"; pkill -USR1 -P $$ || true' USR1

echo "[\$(date)] Launching Julia job $JOBNAME"
exec stdbuf -oL -eL julia --project=. scripts/infer_this_main.jl $CMD
EOF
done

echo "All jobs submitted. Use 'squeue -u \$USER' to monitor."
