#!/bin/bash
#SBATCH --time=0-04:00:00
#SBATCH --account=def-agodbout
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=16000M
#SBATCH --job-name=oyster-gpu-pipeline
#SBATCH --output=hpc_outputs/%x-%j_gpu_train_infer.out
#SBATCH --mail-user=jezulluna@upei.ca
#SBATCH --mail-type=ALL

# Usage:
#   sbatch submit_gpu_train_infer.sh config_hpc.yaml [--skip-train] [--skip-infer]
#
# Options:
#   --skip-train : Skip training, only run inference
#   --skip-infer : Skip inference, only run training

CONFIG_FILE=${1:-config_hpc.yaml}
SKIP_TRAIN=false
SKIP_INFER=false

# Parse optional flags
for arg in "$@"; do
    case $arg in
        --skip-train) SKIP_TRAIN=true ;;
        --skip-infer) SKIP_INFER=true ;;
    esac
done

# --- Environment Setup ---
echo "GPU Job Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Config: $CONFIG_FILE"
echo "Skip training: $SKIP_TRAIN"
echo "Skip inference: $SKIP_INFER"

module load gcc cuda opencv python scipy-stack
source .venv/bin/activate

# --- Sanity Checks ---
echo "Python executable: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available to PyTorch: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Visible CUDA devices: $CUDA_VISIBLE_DEVICES"
echo "--- GPU Information ---"
nvidia-smi
echo "---------------------"

TRAIN_EXIT=0
INFER_EXIT=0

# --- Step 02: Training ---
if [ "$SKIP_TRAIN" = false ]; then
    echo "Step 02: Training YOLO Model..."
    python -m src.main_scripts.02_train_yolo --config ${CONFIG_FILE}
    TRAIN_EXIT=$?
    echo "Step 02 finished with exit code: $TRAIN_EXIT"

    if [ $TRAIN_EXIT -ne 0 ]; then
        echo "ERROR: Training failed. Skipping inference."
        echo "GPU Job Finished with errors: $(date)"
        exit $TRAIN_EXIT
    fi
else
    echo "Step 02: SKIPPED (--skip-train flag)"
fi

# --- Step 03: Inference ---
if [ "$SKIP_INFER" = false ]; then
    echo "Step 03: Running Inference..."
    echo "NOTE: Inference will automatically skip completed slides."
    python -m src.main_scripts.03_run_inference --config ${CONFIG_FILE}
    INFER_EXIT=$?
    echo "Step 03 finished with exit code: $INFER_EXIT"
else
    echo "Step 03: SKIPPED (--skip-infer flag)"
fi

echo "GPU Job Finished: $(date)"
echo "Summary: Training=$TRAIN_EXIT, Inference=$INFER_EXIT"

# Exit with non-zero if either step failed
if [ $TRAIN_EXIT -ne 0 ] || [ $INFER_EXIT -ne 0 ]; then
    exit 1
fi
