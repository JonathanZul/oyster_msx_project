#!/bin/bash
#SBATCH --time=0-04:00:00
#SBATCH --account=def-agodbout
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24000M
#SBATCH --job-name=oyster-segment
#SBATCH --output=hpc_outputs/%x-%j_gpu_segment.out
#SBATCH --mail-user=jezulluna@upei.ca
#SBATCH --mail-type=ALL

# Usage:
#   sbatch submit_gpu_segment.sh config_hpc.yaml

CONFIG_FILE=${1:-config_hpc.yaml}

echo "GPU segmentation job started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Config: $CONFIG_FILE"

module load gcc cuda opencv python scipy-stack
source .venv/bin/activate

echo "Python executable: $(which python)"
echo "PyTorch CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
nvidia-smi

echo "Step 00: Segmenting oysters with SAM..."
python -m src.main_scripts.s00_segment_with_sam --config "${CONFIG_FILE}"
EXIT_CODE=$?
echo "Step 00 finished with exit code: ${EXIT_CODE}"
echo "GPU segmentation job finished: $(date)"
exit ${EXIT_CODE}
