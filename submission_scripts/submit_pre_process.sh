#!/bin/bash
# Legacy script: retained for compatibility. For faster HPC runs, prefer:
# 1) submission_scripts/submit_gpu_segment.sh
# 2) submission_scripts/submit_cpu_create_dataset.sh
#SBATCH --time=0-05:00:00        # 3 hours
#SBATCH --account=def-agodbout
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4       # Use more CPUs as we are CPU-bound
#SBATCH --mem=32000M             # More RAM is fine for CPU nodes
#SBATCH --job-name=oyster-pre-process
#SBATCH --output=hpc_outputs/%x-%j_preprocessing.out
#SBATCH --mail-user=jezulluna@upei.ca
#SBATCH --mail-type=END,FAIL

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
python -m src.main_scripts.s00_segment_with_sam --config ${CONFIG_FILE}
STEP00_EXIT=$?
echo "Step 00 finished with exit code: ${STEP00_EXIT}"
if [ ${STEP00_EXIT} -ne 0 ]; then
    echo "ERROR: Step 00 failed. Aborting preprocessing job."
    exit ${STEP00_EXIT}
fi

echo "Step 01: Creating YOLO Dataset..."
python -m src.main_scripts.01_create_dataset --config ${CONFIG_FILE}
STEP01_EXIT=$?
echo "Step 01 finished with exit code: ${STEP01_EXIT}"
if [ ${STEP01_EXIT} -ne 0 ]; then
    echo "ERROR: Step 01 failed."
    exit ${STEP01_EXIT}
fi

echo "CPU Pre-processing Job Finished: $(date)"
