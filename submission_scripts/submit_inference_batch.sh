#!/bin/bash
#SBATCH --time=0-02:00:00
#SBATCH --account=def-agodbout
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16000M
#SBATCH --job-name=oyster-infer
#SBATCH --output=hpc_outputs/%x-%j_%a.out
#SBATCH --mail-user=jezulluna@upei.ca
#SBATCH --mail-type=END,FAIL

# Usage:
#   sbatch --array=0-N submit_inference_batch.sh config_hpc.yaml [slides_per_job]
#
# Examples:
#   # Process 27 slides, 1 slide per job (27 jobs):
#   sbatch --array=0-26 submit_inference_batch.sh config_hpc.yaml 1
#
#   # Process 27 slides, 3 slides per job (9 jobs):
#   sbatch --array=0-8 submit_inference_batch.sh config_hpc.yaml 3
#
#   # Process slides 10-14 only (5 jobs):
#   sbatch --array=10-14 submit_inference_batch.sh config_hpc.yaml 1
#
# To see which slides need processing:
#   python -m src.main_scripts.03_run_inference --config config_hpc.yaml --list-slides

CONFIG_FILE=${1:-config_hpc.yaml}
SLIDES_PER_JOB=${2:-1}

echo "=== Inference Batch Job ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Started: $(date)"
echo "Host: $(hostname)"
echo "Config: $CONFIG_FILE"
echo "Slides per job: $SLIDES_PER_JOB"
echo "=========================="

module load gcc cuda opencv python scipy-stack
source .venv/bin/activate

echo "Python: $(which python)"
echo "PyTorch CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
nvidia-smi

python -m src.main_scripts.03_run_inference \
    --config ${CONFIG_FILE} \
    --slide-index ${SLURM_ARRAY_TASK_ID} \
    --slides-per-job ${SLIDES_PER_JOB}

EXIT_CODE=$?
echo "=== Job Finished: $(date) ==="
echo "Exit code: $EXIT_CODE"
exit $EXIT_CODE
