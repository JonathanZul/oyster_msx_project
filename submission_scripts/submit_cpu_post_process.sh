#!/bin/bash
#SBATCH --time=0-01:00:00        # 1 hour
#SBATCH --account=def-agodbout
#SBATCH --cpus-per-task=8
#SBATCH --mem=8000M
#SBATCH --job-name=oyster-post-process
#SBATCH --output=hpc_outputs/%x-%j_cpu_post_process.out
#SBATCH --mail-user=jezulluna@upei.ca
#SBATCH --mail-type=ALL

# Usage:
#   sbatch submit_cpu_post_process.sh config_hpc.yaml [--force] [--include-incomplete]
#
# Options:
#   --force              : Re-process slides even if GeoJSON already exists
#   --include-incomplete : Process slides even if inference wasn't fully completed
#
# Note: By default, only processes slides with a .completed marker and skips
# slides that already have GeoJSON output.

CONFIG_FILE=${1:-config_hpc.yaml}
shift  # Remove config file from arguments
EXTRA_ARGS="$@"  # Pass remaining args to Python script

# --- Environment Setup ---
echo "CPU Post-processing Job Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Config: $CONFIG_FILE"
echo "Extra args: $EXTRA_ARGS"

module load gcc cuda opencv python scipy-stack
source .venv/bin/activate

# --- Run Final CPU Script ---
echo "Step 04: Formatting Predictions for QuPath..."
python -m src.main_scripts.04_format_predictions --config ${CONFIG_FILE} ${EXTRA_ARGS}
EXIT_CODE=$?
echo "Step 04 finished with exit code: $EXIT_CODE"

echo "CPU Post-processing Job Finished: $(date)"
exit $EXIT_CODE
