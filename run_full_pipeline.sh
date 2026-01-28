#!/bin/bash

# ====================================================================
#          Master Submission Script for the Oyster MSX Pipeline
# ====================================================================
# This script submits the entire three-stage pipeline to the Slurm
# scheduler, creating a dependency chain.
#
# Usage:
#   ./run_full_pipeline.sh [path/to/your/config.yaml]
#
# If no config path is provided, it defaults to 'config.yaml'.
# Make sure this script is executable: chmod +x run_full_pipeline.sh
# ====================================================================

echo "🚀 Starting the full Oyster MSX pipeline..."

CONFIG_FILE=${1:-"config.yaml"} # Use the first argument ($1), or "config.yaml" if it's not set.
echo "Using configuration file: ${CONFIG_FILE}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ ERROR: Configuration file not found at '${CONFIG_FILE}'"
    exit 1
fi

# --- Step 1: Submit the CPU pre-processing job ---
echo "Submitting Step 1: CPU Pre-processing..."
job1_output=$(sbatch submission_scripts/submit_pre_process.sh ${CONFIG_FILE})
job1_id=$(echo $job1_output | awk '{print $4}')

if [[ -z "$job1_id" ]]; then
    echo "❌ ERROR: Failed to submit the pre-processing job. Aborting."
    exit 1
fi
echo "✅ Pre-processing job submitted. Job ID: ${job1_id}"


# --- Step 2: Submit the GPU training/inference job ---
echo "Submitting Step 2: GPU Training & Inference..."
job2_output=$(sbatch --dependency=afterok:${job1_id} submission_scripts/submit_gpu_train_infer.sh ${CONFIG_FILE})
job2_id=$(echo $job2_output | awk '{print $4}')

if [[ -z "$job2_id" ]]; then
    echo "❌ ERROR: Failed to submit the GPU job. Aborting."
    scancel ${job1_id}
    exit 1
fi
echo "✅ GPU job submitted. Job ID: ${job2_id}. Will run after job ${job1_id} completes."


# --- Step 3: Submit the CPU post-processing job ---
echo "Submitting Step 3: CPU Post-processing..."
job3_output=$(sbatch --dependency=afterok:${job2_id} submission_scripts/submit_cpu_post_process.sh ${CONFIG_FILE})
job3_id=$(echo $job3_output | awk '{print $4}')

if [[ -z "$job3_id" ]]; then
    echo "❌ ERROR: Failed to submit the post-processing job. Aborting."
    scancel ${job1_id} ${job2_id}
    exit 1
fi
echo "✅ Post-processing job submitted. Job ID: ${job3_id}. Will run after job ${job2_id} completes."

echo "🎉 Full pipeline submitted successfully!"
