# src/tools/run_cross_validation_master.py
"""
Performs a K-fold cross-validation experiment to compare the performance
of different oyster segmentation methods (e.g., U-Net, SAM, Watershed).

This script automates the entire process:
1.  Splitting the dataset into K folds.
2.  Iterating through each fold, using it as a test set while training on the rest.
3.  For each fold, running each specified segmentation pipeline.
4.  Evaluating the generated predictions against ground truth annotations.
5.  Aggregating the results across all folds and printing a final summary report
    comparing the methods' average performance and stability.
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm

# Add src to path to allow for relative imports
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.file_handling import load_config
from src.utils.logging_config import setup_logging, log_config
from src.utils.evaluation_utils import calculate_segmentation_metrics, intelligently_match_masks
from src.utils.wsi_utils import find_matching_wsi_path, get_wsi_level0_dimensions

# Import the callable pipeline functions from our refactored scripts
from src.main_scripts.s00_segment_with_sam import run_sam_pipeline
from archive.unet_00b_train_model import run_unet_training
from archive.unet_00_run_inference_callable import run_unet_inference
from archive.s00_segment_oysters_classical import run_watershed_pipeline

# Import the rasterization logic from the evaluation script
from src.tools.evaluate_segmentation import rasterize_ground_truth


def evaluate_predictions_for_fold(pred_dir: Path, gt_dir: Path, test_stems: list[str], config: dict, logger,
                                  fold_num: int, method_name: str) -> tuple[dict, pd.DataFrame]:
    """Evaluates segmentation predictions for a single fold against ground truth.

    This function iterates through the test set of a fold, loads the predicted
    masks and corresponding ground truth GeoJSONs, and calculates segmentation
    metrics. It includes logic to handle cases where a method fails to produce
    a mask and an "intelligent matching" system to correctly pair predicted
    masks with ground truth masks.

    Args:
        pred_dir: Directory containing the prediction outputs for a method.
        gt_dir: Directory containing the ground truth GeoJSON annotations.
        test_stems: A list of slide filename stems in the current test set.
        config: The main configuration dictionary.
        logger: The logger instance for logging messages.
        fold_num: The current fold number (for logging purposes).
        method_name: The name of the segmentation method being evaluated.

    Returns:
        A tuple containing:
        - A dictionary with the average metrics (IoU, Dice, J&F) for the fold.
        - A pandas DataFrame with detailed, per-prediction metrics.
    """
    logger.info(f"Evaluating predictions in: {pred_dir}")
    wsi_dir = Path(config["paths"]["raw_wsis"])
    class_map = config["ml_segmentation"]["classes"]  # U-Net class map

    per_prediction_results = []

    eval_debug_dir = Path("verification_outputs/eval_debug") / f"fold_{fold_num}" / method_name
    eval_debug_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving evaluation debug images to: {eval_debug_dir.resolve()}")

    for stem in test_stems:
        gt_geojson_path = gt_dir / f"{stem}.geojson"
        if not gt_geojson_path.exists():
            logger.warning(f"Ground truth file not found for {stem}, skipping evaluation for this slide.")
            continue

        wsi_path = find_matching_wsi_path(gt_geojson_path, wsi_dir)
        if not wsi_path: continue
        wsi_dims = get_wsi_level0_dimensions(wsi_path, logger)
        if not wsi_dims: continue

        pred_mask_1_raw = cv2.imread(str(pred_dir / stem / "oyster_1_mask.png"), cv2.IMREAD_GRAYSCALE)
        pred_mask_2_raw = cv2.imread(str(pred_dir / stem / "oyster_2_mask.png"), cv2.IMREAD_GRAYSCALE)

        if pred_mask_1_raw is None or pred_mask_2_raw is None:
            logger.warning(f"Missing prediction masks for {stem}. Scoring as zero.")

            per_prediction_results.append({
                "slide": stem,
                "oyster": "oyster_1",
                "dice": 0.0,
                "iou": 0.0,
                "j_and_f": 0.0,
                "status": "FAILED"
            })
            per_prediction_results.append({
                "slide": stem,
                "oyster": "oyster_2",
                "dice": 0.0,
                "iou": 0.0,
                "j_and_f": 0.0,
                "status": "FAILED"
            })
            continue

        gt_mask_multichannel = rasterize_ground_truth(gt_geojson_path, pred_mask_1_raw.shape, wsi_dims, class_map, logger)
        if gt_mask_multichannel is None:
            continue

        # Convert all masks to boolean for metric calculations and matching.
        gt_mask_1 = gt_mask_multichannel[:, :, 0].astype(bool)
        gt_mask_2 = gt_mask_multichannel[:, :, 1].astype(bool)
        pred_mask_1 = pred_mask_1_raw.astype(bool)
        pred_mask_2 = pred_mask_2_raw.astype(bool)

        final_pred_1, final_gt_1, final_pred_2, final_gt_2 = intelligently_match_masks(
            pred_mask_1, pred_mask_2, gt_mask_1, gt_mask_2, logger, stem
        )

        # 2. Calculate the final metrics on the correctly paired masks.
        metrics1 = calculate_segmentation_metrics(final_pred_1, final_gt_1)
        metrics2 = calculate_segmentation_metrics(final_pred_2, final_gt_2)

        # 3. Append detailed results for the final report table.
        per_prediction_results.append({"slide": stem, "oyster": "oyster_1", "status": "OK", **metrics1})
        per_prediction_results.append({"slide": stem, "oyster": "oyster_2", "status": "OK", **metrics2})

        # 4. Generate and save the blended visualization for each correctly paired oyster.
        def create_blend_viz(pred_mask, gt_mask):
            pred_uint8 = (pred_mask * 255).astype(np.uint8)
            gt_uint8 = (gt_mask * 255).astype(np.uint8)

            # Create a 3-channel image for color overlays.
            gt_viz = cv2.cvtColor(gt_uint8, cv2.COLOR_GRAY2BGR)
            gt_viz[gt_mask, :] = [255, 0, 0]  # Blue for Ground Truth pixels

            pred_viz = cv2.cvtColor(pred_uint8, cv2.COLOR_GRAY2BGR)
            pred_viz[pred_mask, :] = [0, 255, 0]  # Green for Prediction pixels

            # The blend will show Cyan (Blue + Green) where they overlap (True Positives).
            blend = cv2.addWeighted(gt_viz, 0.5, pred_viz, 0.5, 0)
            return blend

        blend_1 = create_blend_viz(final_pred_1, final_gt_1)
        cv2.imwrite(str(eval_debug_dir / f"{stem}_oyster1_eval.png"), blend_1)

        blend_2 = create_blend_viz(final_pred_2, final_gt_2)
        cv2.imwrite(str(eval_debug_dir / f"{stem}_oyster2_eval.png"), blend_2)

    if not per_prediction_results:
        return {"iou": 0, "dice": 0, "j_and_f": 0}, pd.DataFrame()

    # Create the DataFrame from the detailed results.
    per_prediction_df = pd.DataFrame(per_prediction_results)

    # Calculate the summary statistics directly from this DataFrame.
    avg_iou = per_prediction_df['iou'].mean()
    avg_dice = per_prediction_df['dice'].mean()
    avg_j_and_f = per_prediction_df['j_and_f'].mean()

    summary = {"iou": avg_iou, "dice": avg_dice, "j_and_f": avg_j_and_f}

    return summary, per_prediction_df


def print_fold_comparison_table(all_fold_predictions: dict, fold_num: int, logger):
    """Prints a detailed, per-prediction comparison table for all methods in a fold.

    Args:
        all_fold_predictions: Dict mapping method names to their per-prediction DataFrames.
        fold_num: The current fold number being reported.
        logger: The logger instance for output.
    """
    if not all_fold_predictions:
        return

    first_method = list(all_fold_predictions.keys())[0]
    base_df = all_fold_predictions[first_method]

    if base_df.empty:
        logger.info("No predictions to display in table.")
        return

    comparison_rows = []

    for _, row in base_df.iterrows():
        slide = row['slide']
        oyster = row['oyster']

        comparison_row = {
            'slide': slide,
            'oyster': oyster
        }

        for method in all_fold_predictions.keys():
            method_df = all_fold_predictions[method]
            method_row = method_df[(method_df['slide'] == slide) & (method_df['oyster'] == oyster)]

            if not method_row.empty:
                comparison_row[f'{method}_dice'] = method_row.iloc[0]['dice']
                comparison_row[f'{method}_iou'] = method_row.iloc[0]['iou']
                comparison_row[f'{method}_j&f'] = method_row.iloc[0]['j_and_f']
                comparison_row[f'{method}_status'] = method_row.iloc[0]['status']
            else:
                comparison_row[f'{method}_dice'] = np.nan
                comparison_row[f'{method}_iou'] = np.nan
                comparison_row[f'{method}_j&f'] = np.nan
                comparison_row[f'{method}_status'] = 'N/A'

        comparison_rows.append(comparison_row)

    comparison_df = pd.DataFrame(comparison_rows)

    logger.info(f"\n{'=' * 80}")
    logger.info(f"PER-PREDICTION SCORES - FOLD {fold_num}")
    logger.info(f"{'=' * 80}")

    # Configure pandas display options for clean console output.
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', '{:.4f}'.format)

    logger.info(f"\n{comparison_df.to_string(index=False)}")
    logger.info(f"{'=' * 80}\n")

    logger.info(f"Per-Slide Averages (Fold {fold_num}):")
    slide_groups = comparison_df.groupby('slide')

    for method in all_fold_predictions.keys():
        dice_col = f'{method}_dice'
        if dice_col in comparison_df.columns:
            slide_avgs = slide_groups[dice_col].mean()
            logger.info(f"\n{method.upper()} - Average Dice per Slide:")
            for slide, avg in slide_avgs.items():
                logger.info(f"  {slide}: {avg:.4f}")

    logger.info(f"\n{'=' * 80}\n")


def main():
    """Main function to perform the K-Fold Cross-Validation process."""
    parser = argparse.ArgumentParser(description="Run K-Fold Cross-Validation for Segmentation Methods")
    parser.add_argument("-c", "--config", type=str, default="config.yaml", help="Path to the config file.")
    args = parser.parse_args()

    config = load_config(args.config)
    if not config:
        return

    logger = setup_logging(Path("logs"), "cross_validation")
    log_config(config, logger)

    cv_config = config["cross_validation"]
    logger.info(
        f"--- Starting {cv_config['n_splits']}-Fold Cross-Validation for methods: {cv_config['methods_to_run']} ---")

    # 1. Prepare Data and Folds
    gt_annot_dir = Path(config["paths"]["segmentation_annotations"])
    all_gt_paths = [p for p in gt_annot_dir.glob("*.geojson") if not p.name.startswith("._")]
    all_stems = sorted([p.stem for p in all_gt_paths])

    kf = KFold(n_splits=cv_config["n_splits"], shuffle=True, random_state=cv_config.get("data_split_seed"))

    # 2. Main Cross-Validation Loop
    all_results = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(all_stems)):
        fold_num = fold + 1
        logger.info(f"\n{'=' * 25} FOLD {fold_num}/{cv_config['n_splits']} {'=' * 25}")

        train_stems = [all_stems[i] for i in train_idx]
        test_stems = [all_stems[i] for i in test_idx]

        logger.info(f"Training on {len(train_stems)} slides, Testing on {len(test_stems)} slides.")

        # Directory to store all temporary outputs for this fold.
        fold_output_dir = Path(cv_config["temp_output_dir"]) / f"fold_{fold_num}"
        if fold_output_dir.exists():
            logger.warning(f"Removing existing temporary directory for fold {fold_num}: {fold_output_dir}")
            try:
                shutil.rmtree(fold_output_dir)
            except OSError as e:
                logger.error(f"Error removing directory {fold_output_dir}: {e}. This may be a temp file issue.")

        # Store per-prediction results for all methods to generate a comparison table at the end of the fold.
        fold_output_dir.mkdir(parents=True, exist_ok=True)

        # Store per-prediction results for all methods in this fold
        fold_prediction_tables = {}

        for method in cv_config["methods_to_run"]:
            logger.info(f"\n--- Running Method: {method.upper()} for Fold {fold_num} ---")

            method_pred_dir = fold_output_dir / method

            # 3. Run Each Method's Pipeline
            if method == "unet":
                model_path = fold_output_dir / "unet_model.pt"
                # Train the U-Net model on the current training split.
                run_unet_training(config, logger, train_stems, test_stems, model_path)
                # Run inference on the current test split using the just-trained model.
                run_unet_inference(config, logger, model_path, test_stems, method_pred_dir)

            elif method == "sam":
                run_sam_pipeline(config, logger, test_stems, output_override_dir=method_pred_dir)

            elif method == "watershed":
                run_watershed_pipeline(config, logger, test_stems, output_override_dir=method_pred_dir)

            # 4. Evaluate Results for the Current Method and Fold
            fold_metrics, per_prediction_df = evaluate_predictions_for_fold(
                method_pred_dir, gt_annot_dir, test_stems, config, logger, fold_num, method
            )

            fold_metrics["method"] = method
            fold_metrics["fold"] = fold_num
            all_results.append(fold_metrics)

            fold_prediction_tables[method] = per_prediction_df

            logger.info(f"Metrics for {method.upper()} on Fold {fold_num}: {fold_metrics}")

        print_fold_comparison_table(fold_prediction_tables, fold_num, logger)

    # 5. Final Report Generation
    logger.info(f"\n\n{'=' * 25} CROSS-VALIDATION SUMMARY {'=' * 25}")
    results_df = pd.DataFrame(all_results)

    summary = results_df.groupby("method").agg(
        avg_iou=('iou', 'mean'),
        std_iou=('iou', 'std'),
        avg_dice=('dice', 'mean'),
        std_dice=('dice', 'std'),
        avg_j_and_f=('j_and_f', 'mean'),
        std_j_and_f=('j_and_f', 'std')
    ).reset_index()

    logger.info(f"\nPer-Fold Metrics:\n{results_df.to_string(index=False)}")
    logger.info("\n" + "---" * 15)
    logger.info(f"\nFinal Aggregated Results:\n{summary.to_string(index=False)}")
    logger.info("---" * 15)


if __name__ == "__main__":
    main()
