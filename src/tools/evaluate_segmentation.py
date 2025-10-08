# src/utils/evaluation_utils.py
"""
Evaluates the final segmentation masks produced by a model against the
corresponding ground truth GeoJSON annotations.

This script is intended for final assessment after a model has been trained and
has generated its predictions. It iterates through each slide's output directory,
loads the predicted masks and ground truth, calculates performance metrics
(IoU, Dice, etc.), and prints a summary report to the console.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add src to path to allow for relative imports.
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.file_handling import load_config
from src.utils.logging_config import setup_logging, log_config
from src.utils.wsi_utils import find_matching_wsi_path, get_wsi_level0_dimensions
from src.utils.evaluation_utils import (
    calculate_segmentation_metrics,
    rasterize_ground_truth,
    intelligently_match_masks,
    find_polygon_coords
)


def main():
    """Runs the main evaluation pipeline."""
    parser = argparse.ArgumentParser(description="Evaluate Segmentation Performance")
    parser.add_argument("-c", "--config", type=str, default="config.yaml", help="Path to the config file.")
    args = parser.parse_args()

    config = load_config(args.config)
    if not config:
        return

    logger = setup_logging(Path("logs"), "evaluate_segmentation")
    log_config(config, logger)
    logger.info("--- Starting Segmentation Performance Evaluation ---")

    pred_mask_dir = Path(config["paths"]["oyster_masks"])
    gt_annot_dir = Path(config["paths"]["segmentation_annotations"])
    wsi_dir = Path(config["paths"]["raw_wsis"])

    if "ml_segmentation" not in config:
        logger.critical("Missing 'ml_segmentation' section in config.yaml for class mapping.");
        return
    class_map = config["ml_segmentation"]["classes"]

    results = []
    slide_dirs = [d for d in pred_mask_dir.iterdir() if d.is_dir()]
    logger.info(f"Found {len(slide_dirs)} prediction directories to evaluate.")

    for slide_dir in tqdm(slide_dirs, desc="Evaluating Slides"):
        slide_name = slide_dir.name
        gt_geojson_path = gt_annot_dir / f"{slide_name}.geojson"
        if not gt_geojson_path.exists(): continue

        wsi_path = find_matching_wsi_path(gt_geojson_path, wsi_dir)
        if not wsi_path: continue
        wsi_dims = get_wsi_level0_dimensions(wsi_path, logger)
        if not wsi_dims: continue

        pred_mask_1 = cv2.imread(str(slide_dir / "oyster_1_mask.png"), cv2.IMREAD_GRAYSCALE)
        pred_mask_2 = cv2.imread(str(slide_dir / "oyster_2_mask.png"), cv2.IMREAD_GRAYSCALE)
        if pred_mask_1 is None or pred_mask_2 is None: continue

        # Convert vector ground truth annotations to a raster mask for pixel-wise comparison.
        target_shape = pred_mask_1.shape
        gt_mask_multichannel = rasterize_ground_truth(gt_geojson_path, target_shape, wsi_dims, class_map, logger)
        if gt_mask_multichannel is None: continue

        gt_mask_1 = gt_mask_multichannel[:, :, 0]
        gt_mask_2 = gt_mask_multichannel[:, :, 1]

        # Determine the best pairing of predicted masks to ground truth masks to
        # avoid penalizing the model for arbitrary label ordering.
        final_pred_1, final_gt_1, final_pred_2, final_gt_2 = intelligently_match_masks(
            pred_mask_1, pred_mask_2, gt_mask_1, gt_mask_2, logger, slide_name
        )

        metrics1 = calculate_segmentation_metrics(final_pred_1, final_gt_1)
        metrics2 = calculate_segmentation_metrics(final_pred_2, final_gt_2)

        results.append({
            "slide_name": slide_name,
            "iou_oyster1": metrics1["iou"], "dice_oyster1": metrics1["dice"], "j_and_f_oyster1": metrics1["j_and_f"],
            "iou_oyster2": metrics2["iou"], "dice_oyster2": metrics2["dice"], "j_and_f_oyster2": metrics2["j_and_f"]
        })

    if not results:
        logger.error("No results were generated. Check paths and ground truth files.");
        return

    # --- Generate and Display Final Report ---
    df = pd.DataFrame(results)
    avg_iou = df[['iou_oyster1', 'iou_oyster2']].values.mean()
    avg_dice = df[['dice_oyster1', 'dice_oyster2']].values.mean()
    avg_j_and_f = df[['j_and_f_oyster1', 'j_and_f_oyster2']].values.mean()

    logger.info("\n\n--- Segmentation Performance Report ---")
    pd.set_option('display.precision', 4)
    report_df = df[["slide_name", "dice_oyster1", "iou_oyster1", "j_and_f_oyster1", "dice_oyster2", "iou_oyster2",
                    "j_and_f_oyster2"]]
    logger.info(f"\nPer-Slide Metrics:\n{report_df.to_string(index=False)}")
    logger.info("\n" + "---" * 10)
    logger.info(f"Overall Average IoU Score:   {avg_iou:.4f}")
    logger.info(f"Overall Average Dice Score:  {avg_dice:.4f}")
    logger.info(f"Overall Average J&F Score:   {avg_j_and_f:.4f}")
    logger.info("---" * 10)


if __name__ == "__main__":
    main()
