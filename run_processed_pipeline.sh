#!/bin/bash

# ====================================================================
#     Master Submission Script for the PROCESSED Oyster MSX Pipeline
# ====================================================================
# This script submits the training, inference, and post-processing
# stages of the pipeline to the Slurm scheduler.
#
# !! PREREQUISITES !!
# This script ASSUMES that the pre-processing scripts (00 and 01)
# have already been run successfully, and that a complete YOLO dataset
# exists in your scratch space at the path specified in config.yaml.
#
# Usage:
#   ./run_processed_pipeline.sh [path/to/your/config.yaml]
#
# Make sure this script is executable:
#   chmod +x run_processed_pipeline.sh
# ====================================================================

echo "🚀 Starting the PROCESSED data pipeline (skipping pre-processing)..."

CONFIG_FILE=${1:-"config.yaml"} # Use the first argument ($1), or "config.yaml" if it's not set.
echo "Using configuration file: ${CONFIG_FILE}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ ERROR: Configuration file not found at '${CONFIG_FILE}'"
    exit 1
fi

# --- Step 1: Submit the GPU training/inference job ---
# This is now the first step in our chain.
echo "Submitting Step 1: GPU Training & Inference (Scripts 02 & 03)..."
job1_output=$(sbatch submission_scripts/submit_gpu_train_infer.sh ${CONFIG_FILE})
job1_id=$(echo $job1_output | awk '{print $4}')

# --- Robustness Check ---
if [[ -z "$job1_id" ]]; then
    echo "❌ ERROR: Failed to submit the GPU job. Aborting."
    exit 1
fi
echo "✅ GPU job submitted successfully. Job ID: ${job1_id}"


# --- Step 2: Submit the CPU post-processing job ---
# This job waits for the GPU job to complete successfully.
echo "Submitting Step 2: CPU Post-processing (Script 04)..."
job2_output=$(sbatch --dependency=afterok:${job1_id} submission_scripts/submit_cpu_post_process.sh ${CONFIG_FILE})
job2_id=$(echo $job2_output | awk '{print $4}')

# --- Robustness Check ---
if [[ -z "$job2_id" ]]; then
    echo "❌ ERROR: Failed to submit the post-processing job. Aborting."
    # If this fails, we should cancel the GPU job we just submitted.
    scancel ${job1_id}
    exit 1
fi
echo "✅ Post-processing job submitted successfully. Job ID: ${job2_id}. Will run after job ${job1_id} completes."

echo "🎉 Processed data pipeline submitted successfully!"
