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

echo "Starting the full Oyster MSX pipeline..."

CONFIG_FILE=${1:-"config.yaml"} # Use the first argument ($1), or "config.yaml" if it's not set.
SLIDES_PER_JOB=${2:-2}
echo "Using configuration file: ${CONFIG_FILE}"
echo "Slides per inference array task: ${SLIDES_PER_JOB}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Configuration file not found at '${CONFIG_FILE}'"
    exit 1
fi

# --- Step 1: Submit the CPU pre-processing job ---
echo "Submitting Step 1: CPU Pre-processing..."
job1_output=$(sbatch submission_scripts/submit_pre_process.sh ${CONFIG_FILE})
job1_id=$(echo $job1_output | awk '{print $4}')

if [[ -z "$job1_id" ]]; then
    echo "ERROR: Failed to submit the pre-processing job. Aborting."
    exit 1
fi
echo "Pre-processing job submitted. Job ID: ${job1_id}"


# --- Step 2: Submit the GPU training job ---
echo "Submitting Step 2: GPU Training..."
job2_output=$(sbatch --dependency=afterok:${job1_id} submission_scripts/submit_gpu_train_infer.sh ${CONFIG_FILE} --skip-infer)
job2_id=$(echo $job2_output | awk '{print $4}')

if [[ -z "$job2_id" ]]; then
    echo "ERROR: Failed to submit the GPU training job. Aborting."
    scancel ${job1_id}
    exit 1
fi
echo "GPU training job submitted. Job ID: ${job2_id}. Will run after job ${job1_id} completes."

# --- Step 3: Submit array-based inference jobs ---
PENDING_COUNT=0
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate >/dev/null 2>&1
fi
PENDING_COUNT=$(python -m src.main_scripts.03_run_inference --config ${CONFIG_FILE} --list-slides 2>/dev/null | grep -c "\[PENDING\]" || true)

POSTPROCESS_DEP_JOB=${job2_id}
if [ "${PENDING_COUNT}" -gt 0 ]; then
    ARRAY_MAX=$(( (PENDING_COUNT + SLIDES_PER_JOB - 1) / SLIDES_PER_JOB - 1 ))
    echo "Submitting Step 3: GPU Inference Array (${PENDING_COUNT} pending slides, index 0-${ARRAY_MAX})..."
    job3_output=$(sbatch --array=0-${ARRAY_MAX} --dependency=afterok:${job2_id} submission_scripts/submit_inference_batch.sh ${CONFIG_FILE} ${SLIDES_PER_JOB})
    job3_id=$(echo $job3_output | awk '{print $4}')

    if [[ -z "$job3_id" ]]; then
        echo "ERROR: Failed to submit inference array job. Aborting."
        scancel ${job1_id} ${job2_id}
        exit 1
    fi
    echo "GPU inference array submitted. Job ID: ${job3_id}. Will run after job ${job2_id} completes."
    POSTPROCESS_DEP_JOB=${job3_id}
else
    echo "No pending slides found for inference. Skipping Step 3 inference submission."
fi

# --- Step 4: Submit the CPU post-processing job ---
echo "Submitting Step 4: CPU Post-processing..."
job4_output=$(sbatch --dependency=afterok:${POSTPROCESS_DEP_JOB} submission_scripts/submit_cpu_post_process.sh ${CONFIG_FILE})
job4_id=$(echo $job4_output | awk '{print $4}')

if [[ -z "$job4_id" ]]; then
    echo "ERROR: Failed to submit the post-processing job. Aborting."
    scancel ${job1_id} ${job2_id} ${job3_id}
    exit 1
fi
echo "Post-processing job submitted. Job ID: ${job4_id}. Will run after job ${POSTPROCESS_DEP_JOB} completes."

echo "Full pipeline submitted successfully."
