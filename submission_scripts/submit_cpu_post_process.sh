#!/bin/bash
#SBATCH --time=0-01:00:00        # 1 hour
#SBATCH --account=def-agodbout
#SBATCH --cpus-per-task=8
#SBATCH --mem=8000M
#SBATCH --job-name=oyster-post-process
#SBATCH --output=hpc_outputs/%x-%j_cpu_post_process.out
#SBATCH --mail-user=jezulluna@upei.ca
#SBATCH --mail-type=ALL

CONFIG_FILE=$1

# --- Environment Setup ---
echo "CPU Post-processing Job Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"

module load gcc cuda opencv python scipy-stack
source .venv/bin/activate

# --- Run Final CPU Script ---
echo "Step 04: Formatting Predictions for QuPath..."
python -m src.main_scripts.04_format_predictions --config ${CONFIG_FILE}
echo "Step 04 finished with exit code: $?"

echo "CPU Post-processing Job Finished: $(date)"
