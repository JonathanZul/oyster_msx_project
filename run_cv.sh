#!/bin/bash
#SBATCH --time=0-12:00:00        # Request 12 hours, a safe buffer for 5 folds
#SBATCH --account=def-agodbout   # Use your specific account
#SBATCH --gres=gpu:1             # Request 1 GPU
#SBATCH --cpus-per-task=8        # Number of CPU cores
#SBATCH --mem=16000M             # Request 16GB of memory
#SBATCH --job-name=oyster-cv     # Job name for easy identification
#SBATCH --output=hpc_outputs/%x-%j.out # Standard output and error file
#SBATCH --mail-user=jezulluna@upei.ca
#SBATCH --mail-type=ALL

# --- Job Configuration ---
# This script accepts the path to the config file as its first argument.
# Example usage: sbatch run_cv.sh config.yaml
CONFIG_FILE=${1:-config.yaml} # Default to config.yaml if no argument is provided

# --- Environment Setup ---
echo "================================================================="
echo "Cross-Validation Job Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Running on host: $(hostname)"
echo "Running on node: $SLURMD_NODENAME"
echo "GPU(s) assigned: $CUDA_VISIBLE_DEVICES"
echo "Working directory: $(pwd)"
echo "Using config file: ${CONFIG_FILE}"
echo "================================================================="

# Load the necessary modules for your HPC environment.
# These modules provide access to compilers, CUDA, and pre-built libraries.
# The specific names might differ slightly on your cluster.
echo "Loading modules..."
module load gcc cuda opencv python scipy-stack

# Activate your project's Python virtual environment.
echo "Activating virtual environment..."
source .venv/bin/activate
echo "Python interpreter: $(which python)"

# --- Run the Cross-Validation Script ---
echo "Starting the cross-validation master script..."
python -m src.tools.run_cross_validation_master --config ${CONFIG_FILE}

# Capture the exit code of the Python script. 0 means success.
EXIT_CODE=$?
echo "Python script finished with exit code: $EXIT_CODE"

# --- Job Completion ---
echo "================================================================="
echo "Cross-Validation Job Finished: $(date)"
echo "================================================================="

# Exit with the same code as the Python script, so Slurm knows if the job failed.
exit $EXIT_CODE
