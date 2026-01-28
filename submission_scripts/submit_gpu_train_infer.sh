#!/bin/bash
#SBATCH --time=0-02:00:00
#SBATCH --account=def-agodbout
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=16000M
#SBATCH --job-name=oyster-gpu-pipeline
#SBATCH --output=hpc_outputs/%x-%j_gpu_train_infer.out
#SBATCH --mail-user=jezulluna@upei.ca
#SBATCH --mail-type=ALL

CONFIG_FILE=$1

# --- Environment Setup ---
echo "GPU Job Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"

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

# --- Run GPU-bound Scripts ---
echo "Step 02: Training YOLO Model..."
python -m src.main_scripts.02_train_yolo --config ${CONFIG_FILE}
echo "Step 02 finished with exit code: $?"

echo "Step 03: Running Inference..."
python -m src.main_scripts.03_run_inference --config ${CONFIG_FILE}
echo "Step 03 finished with exit code: $?"

echo "GPU Job Finished: $(date)"
