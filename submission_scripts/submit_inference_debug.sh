#!/bin/bash
#SBATCH --time=0-00:20:00
#SBATCH --account=def-agodbout
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --job-name=oyster-infer-debug
#SBATCH --output=hpc_outputs/%x-%j.out
#SBATCH --mail-user=jezulluna@upei.ca
#SBATCH --mail-type=END,FAIL

# Usage:
#   sbatch submission_scripts/submit_inference_debug.sh config_hpc.yaml [include_annotated]
#
# Examples:
#   sbatch submission_scripts/submit_inference_debug.sh config_hpc.yaml
#   sbatch submission_scripts/submit_inference_debug.sh config_hpc.yaml 1

CONFIG_FILE=${1:-config_hpc.yaml}
INCLUDE_ANNOTATED=${2:-0}

echo "Inference debug job started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Config: $CONFIG_FILE"
if [ "$INCLUDE_ANNOTATED" = "1" ]; then
    echo "Include annotated slides: enabled"
fi

module load gcc cuda opencv python scipy-stack
source .venv/bin/activate

CMD=(python -m src.main_scripts.03_run_inference --config "${CONFIG_FILE}" --debug-slide-selection)
if [ "$INCLUDE_ANNOTATED" = "1" ]; then
    CMD+=(--include-annotated)
fi

"${CMD[@]}"
EXIT_CODE=$?
echo "Inference debug job finished: $(date)"
echo "Exit code: $EXIT_CODE"
exit $EXIT_CODE
