#!/bin/bash
#SBATCH --time=0-02:00:00
#SBATCH --account=def-agodbout
#SBATCH --cpus-per-task=8
#SBATCH --mem=12000M
#SBATCH --job-name=oyster-post-all
#SBATCH --output=hpc_outputs/%x-%j_cpu_post_process_all.out
#SBATCH --mail-user=jezulluna@upei.ca
#SBATCH --mail-type=END,FAIL

# Usage:
#   sbatch submission_scripts/submit_cpu_post_process_all.sh config_hpc.yaml
#
# This script forces post-processing on all slide prediction directories:
#   --force              : overwrite existing GeoJSON outputs
#   --include-incomplete : include slides without .completed marker

CONFIG_FILE=${1:-config_hpc.yaml}

echo "CPU Post-processing (ALL slides) Job Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Config: $CONFIG_FILE"

module load gcc cuda opencv python scipy-stack
source .venv/bin/activate

echo "Step 04: Formatting Predictions for QuPath (FORCE ALL)..."
python -m src.main_scripts.04_format_predictions --config "${CONFIG_FILE}" --force --include-incomplete
EXIT_CODE=$?
echo "Step 04 finished with exit code: $EXIT_CODE"

echo "CPU Post-processing (ALL slides) Job Finished: $(date)"
exit $EXIT_CODE
