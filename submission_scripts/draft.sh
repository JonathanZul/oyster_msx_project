#!/bin/bash
#SBATCH --time=0-02:00:00                
#SBATCH --account=def-agodbout
#SBATCH --gpus-per-node=v100:1            # Request 1 V100 GPU
#SBATCH --cpus-per-task=12                # Number of CPU cores
#SBATCH --mem=16000M                      # Request 16GB of RAM
#SBATCH --job-name=oyster-yolo-train      # A descriptive job name
#SBATCH --output=hpc_outputs/%x-%j.out    # Standard output and error log (%x=job name, %j=job ID)
#SBATCH --mail-user=jezulluna@upei.ca    
#SBATCH --mail-type=ALL

# --- Environment Setup ---
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"

# 1. Load necessary software modules
module load gcc cuda opencv python scipy-stack 

# 2. Activate your virtual environment
echo "Activating Python virtual environment..."
source .venv/bin/activate

# --- Sanity Checks ---
echo "Python executable: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available to PyTorch: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Visible CUDA devices: $CUDA_VISIBLE_DEVICES"
echo "--- GPU Information ---"
nvidia-smi
echo "---------------------"

# --- Run the Training Script ---
echo "Starting YOLO model training..."

# The main command to execute your training script
python -m src.main_scripts.02_train_yolo

echo "Python script finished with exit code: $?"
echo "Job finished at: $(date)"
