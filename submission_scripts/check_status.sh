#!/bin/bash
# check_status.sh - Check pipeline status and suggest next steps
#
# Usage: ./check_status.sh [config_file]

CONFIG_FILE=${1:-config_hpc.yaml}

echo "=== MSX Pipeline Status Check ==="
echo "Config: $CONFIG_FILE"
echo "Date: $(date)"
echo ""

# Load environment if on HPC
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate 2>/dev/null
fi

# Extract paths from config (basic parsing)
YOLO_DATASET=$(grep -A5 "^paths:" "$CONFIG_FILE" | grep "yolo_dataset:" | awk '{print $2}' | tr -d '"')
MODEL_DIR=$(grep -A10 "^paths:" "$CONFIG_FILE" | grep "model_output_dir:" | awk '{print $2}' | tr -d '"')
INFERENCE_DIR=$(grep -A10 "^paths:" "$CONFIG_FILE" | grep "inference_results:" | awk '{print $2}' | tr -d '"')

echo "=== Dataset Status ==="
if [ -d "$YOLO_DATASET" ]; then
    TRAIN_COUNT=$(find "$YOLO_DATASET/images/train" -name "*.png" 2>/dev/null | wc -l)
    VAL_COUNT=$(find "$YOLO_DATASET/images/val" -name "*.png" 2>/dev/null | wc -l)
    TEST_COUNT=$(find "$YOLO_DATASET/images/test" -name "*.png" 2>/dev/null | wc -l)
    echo "  Train images: $TRAIN_COUNT"
    echo "  Val images:   $VAL_COUNT"
    echo "  Test images:  $TEST_COUNT"

    if [ "$VAL_COUNT" -eq 0 ]; then
        echo "  WARNING: No validation images! Training will fail."
        echo "  -> Run: python -m src.main_scripts.01_create_dataset --config $CONFIG_FILE"
    fi
else
    echo "  Dataset directory not found: $YOLO_DATASET"
    echo "  -> Run preprocessing steps first"
fi

echo ""
echo "=== Training Status ==="
if [ -d "$MODEL_DIR" ]; then
    LATEST_RUN=$(ls -td "$MODEL_DIR"/*/ 2>/dev/null | head -1)
    if [ -n "$LATEST_RUN" ]; then
        echo "  Latest run: $(basename $LATEST_RUN)"
        if [ -f "$LATEST_RUN/weights/best.pt" ]; then
            echo "  Best model: EXISTS"
            BEST_MODEL="$LATEST_RUN/weights/best.pt"
        else
            echo "  Best model: NOT FOUND"
        fi
        if [ -f "$LATEST_RUN/weights/last.pt" ]; then
            echo "  Last checkpoint: EXISTS (can resume)"
        fi
        # Check training progress
        if [ -f "$LATEST_RUN/results.csv" ]; then
            EPOCHS_DONE=$(tail -1 "$LATEST_RUN/results.csv" | cut -d',' -f1)
            echo "  Epochs completed: $EPOCHS_DONE"
        fi
    else
        echo "  No training runs found"
    fi
else
    echo "  Model directory not found: $MODEL_DIR"
fi

echo ""
echo "=== Inference Status ==="
if [ -d "$INFERENCE_DIR" ]; then
    TOTAL_SLIDES=$(ls -d "$INFERENCE_DIR"/*/ 2>/dev/null | wc -l)
    COMPLETED_SLIDES=$(find "$INFERENCE_DIR" -name ".completed" 2>/dev/null | wc -l)
    PENDING=$((TOTAL_SLIDES - COMPLETED_SLIDES))

    echo "  Total slide directories: $TOTAL_SLIDES"
    echo "  Completed (with marker): $COMPLETED_SLIDES"
    echo "  Pending/In-progress:     $PENDING"

    if [ "$PENDING" -gt 0 ]; then
        echo ""
        echo "  Pending slides:"
        for dir in "$INFERENCE_DIR"/*/; do
            if [ ! -f "$dir/.completed" ]; then
                echo "    - $(basename $dir)"
            fi
        done
    fi
else
    echo "  Inference results directory not found: $INFERENCE_DIR"
fi

echo ""
echo "=== Suggested Next Steps ==="

# Check if validation set is empty
if [ "$VAL_COUNT" -eq 0 ] 2>/dev/null; then
    echo "1. CRITICAL: Recreate dataset with validation split:"
    echo "   python -m src.main_scripts.01_create_dataset --config $CONFIG_FILE"
    echo ""
fi

# Check if training is needed
if [ ! -f "$BEST_MODEL" ] 2>/dev/null; then
    echo "2. Train the model:"
    echo "   sbatch submission_scripts/submit_gpu_train_infer.sh $CONFIG_FILE --skip-infer"
    echo ""
fi

# Check if inference is needed
if [ "$PENDING" -gt 0 ] 2>/dev/null; then
    echo "3. Run inference on remaining slides (using batch jobs):"
    # Get actual count of unannotated slides
    PENDING_COUNT=$PENDING
    echo "   # Option A: Single job (may timeout for many slides)"
    echo "   sbatch submission_scripts/submit_gpu_train_infer.sh $CONFIG_FILE --skip-train"
    echo ""
    echo "   # Option B: Batch jobs (recommended, 2 slides per job)"
    MAX_IDX=$(( (PENDING_COUNT + 1) / 2 - 1 ))
    if [ $MAX_IDX -lt 0 ]; then MAX_IDX=0; fi
    echo "   sbatch --array=0-$MAX_IDX submission_scripts/submit_inference_batch.sh $CONFIG_FILE 2"
    echo ""
    echo "   # Option C: One slide per job (most parallel)"
    MAX_IDX=$((PENDING_COUNT - 1))
    if [ $MAX_IDX -lt 0 ]; then MAX_IDX=0; fi
    echo "   sbatch --array=0-$MAX_IDX submission_scripts/submit_inference_batch.sh $CONFIG_FILE 1"
fi

echo ""
echo "=== Done ==="
