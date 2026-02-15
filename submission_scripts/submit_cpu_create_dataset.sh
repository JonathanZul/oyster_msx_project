#!/bin/bash
#SBATCH --time=0-03:00:00
#SBATCH --account=def-agodbout
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000M
#SBATCH --job-name=oyster-create-ds
#SBATCH --output=hpc_outputs/%x-%j_cpu_create_dataset.out
#SBATCH --mail-user=jezulluna@upei.ca
#SBATCH --mail-type=END,FAIL

# Usage:
#   sbatch submit_cpu_create_dataset.sh config_hpc.yaml

CONFIG_FILE=${1:-config_hpc.yaml}

echo "CPU dataset creation job started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Config: $CONFIG_FILE"

module load gcc cuda opencv python scipy-stack
source .venv/bin/activate

echo "Step 01: Creating YOLO dataset..."
python -m src.main_scripts.01_create_dataset --config "${CONFIG_FILE}"
EXIT_CODE=$?
echo "Step 01 finished with exit code: ${EXIT_CODE}"
echo "CPU dataset creation job finished: $(date)"
exit ${EXIT_CODE}
