#!/bin/bash
#SBATCH --time=0-01:00:00        # 1 hour
#SBATCH --account=def-agodbout
#SBATCH --cpus-per-task=16       # Use more CPUs as we are CPU-bound
#SBATCH --mem=32000M             # More RAM is fine for CPU nodes
#SBATCH --job-name=oyster-pre-process
#SBATCH --output=hpc_outputs/%x-%j_cpu_preprocessing.out
#SBATCH --mail-user=jezulluna@upei.ca
#SBATCH --mail-type=ALL

CONFIG_FILE=$1

# --- Environment Setup ---
echo "CPU Pre-processing Job Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"

module load gcc cuda opencv python scipy-stack
source .venv/bin/activate

# --- Run CPU-bound Scripts ---
echo "Step 00: Segmenting Oysters..."
python -m src.main_scripts.00_segment_oysters --config ${CONFIG_FILE}
echo "Step 00 finished with exit code: $?"

echo "Step 01: Creating YOLO Dataset..."
python -m src.main_scripts.01_create_dataset
echo "Step 01 finished with exit code: $?"

echo "CPU Pre-processing Job Finished: $(date)"
