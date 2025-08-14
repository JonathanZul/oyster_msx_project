# src/tools/evaluate_segmentation.py

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from tqdm import tqdm

# Add src to path to allow for relative imports
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.file_handling import load_config
from src.utils.logging_config import setup_logging, log_config
from src.utils.wsi_utils import find_matching_wsi_path, get_wsi_level0_dimensions
# from archive.unet_00a_create_dataset import find_polygon_coords


def _calculate_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> tuple[float, float, float]:
    """
    Calculates Dice, IoU, and the combined J&F scores for a single pair of boolean masks.
    """
    if pred_mask.shape != gt_mask.shape:
        raise ValueError("Prediction and ground truth masks must have the same shape.")

    pred_mask = pred_mask > 0
    gt_mask = gt_mask > 0

    intersection = np.sum(np.logical_and(pred_mask, gt_mask))
    pred_sum = np.sum(pred_mask)
    gt_sum = np.sum(gt_mask)

    # Add a small epsilon to the denominator to avoid division by zero.
    epsilon = 1e-6
    dice = (2.0 * intersection) / (pred_sum + gt_sum + epsilon)
    iou = intersection / (pred_sum + gt_sum - intersection + epsilon)

    # NEW: Calculate J&F score as the average of IoU (Jaccard) and Dice (F-measure proxy).
    j_and_f = (iou + dice) / 2.0

    return dice, iou, j_and_f


def find_polygon_coords(coordinates_list):
    """
    Recursively searches a nested list to find the first list that contains
    the actual polygon coordinates (a list of [x, y] pairs).
    This handles inconsistencies in GeoJSON nesting from QuPath.
    """
    if all(isinstance(elem, list) and len(elem) == 2 and isinstance(elem[0], (int, float)) for elem in coordinates_list):
        return coordinates_list
    if isinstance(coordinates_list, list) and len(coordinates_list) > 0:
        return find_polygon_coords(coordinates_list[0])
    return None


def _save_evaluation_visualization(pred_mask: np.ndarray, gt_mask: np.ndarray, slide_name: str, oyster_id: int,
                                   output_dir: Path):
    """
    Creates and saves a color-coded visualization comparing a prediction and ground truth mask.
    - Red: False Negatives (Ground truth missed by prediction)
    - Green: False Positives (Prediction outside of ground truth)
    - Yellow: True Positives (Correctly identified pixels)
    """
    # Ensure masks are single-channel and in the 0-255 range for visualization
    pred_vis = (pred_mask > 0).astype(np.uint8) * 255
    gt_vis = (gt_mask > 0).astype(np.uint8) * 255

    # Create a 3-channel BGR image
    h, w = pred_mask.shape
    viz_image = np.zeros((h, w, 3), dtype=np.uint8)

    # Logic for color coding:
    # True Positives (overlap) = Green + Red = Yellow
    # False Negatives (in GT but not Pred) = Red
    # False Positives (in Pred but not GT) = Green
    true_positives = np.logical_and(pred_vis, gt_vis)
    false_negatives = np.logical_and(gt_vis, np.logical_not(pred_vis))
    false_positives = np.logical_and(pred_vis, np.logical_not(gt_vis))

    viz_image[false_negatives] = [0, 0, 255]  # Red (BGR)
    viz_image[false_positives] = [0, 255, 255]  # Green (BGR)
    viz_image[true_positives] = [0, 255, 0]  # Yellow (BGR)

    output_path = output_dir / f"eval_{slide_name}_oyster_{oyster_id}.png"
    cv2.imwrite(str(output_path), viz_image)


def rasterize_ground_truth(geojson_path: Path, target_shape: tuple, wsi_dims: tuple, class_map: dict,
                           logger) -> np.ndarray | None:
    """
    Loads a GeoJSON file and rasterizes annotations into a ground truth mask,
    using precise WSI dimensions for accurate scaling.
    """
    target_h, target_w = target_shape
    wsi_w, wsi_h = wsi_dims

    num_classes = len(class_map)
    gt_mask = np.zeros((target_h, target_w, num_classes), dtype=np.uint8)

    try:
        with open(geojson_path, "r") as f:
            annotations = json.load(f)["features"]

        # Use the true WSI dimensions for a precise scaling factor.
        scale_x = target_w / wsi_w
        scale_y = target_h / wsi_h

        for ann in annotations:
            class_name = ann.get("properties", {}).get("classification", {}).get("name")
            if class_name not in class_map: continue

            channel_idx = class_map[class_name] - 1
            raw_coords = find_polygon_coords(ann["geometry"]["coordinates"])
            if raw_coords is None: continue

            coords_scaled = np.array(raw_coords, dtype=np.float32)
            coords_scaled[:, 0] *= scale_x
            coords_scaled[:, 1] *= scale_y

            mask_channel_slice = gt_mask[:, :, channel_idx].copy()
            cv2.fillPoly(mask_channel_slice, [coords_scaled.astype(np.int32)], 1)
            gt_mask[:, :, channel_idx] = mask_channel_slice

        return gt_mask
    except Exception as e:
        logger.error(f"Failed to process ground truth GeoJSON {geojson_path.name}: {e}", exc_info=True)
        return None


def main():
    """Main function to evaluate segmentation performance against ground truth annotations."""
    parser = argparse.ArgumentParser(description="Evaluate Segmentation Performance")
    parser.add_argument("-c", "--config", type=str, default="config.yaml", help="Path to the config file.")
    args = parser.parse_args()

    config = load_config(args.config)
    if not config: return

    logger = setup_logging(Path("logs"), "evaluate_segmentation")
    log_config(config, logger)
    logger.info("--- Starting Segmentation Performance Evaluation ---")

    pred_mask_dir = Path(config["paths"]["oyster_masks"])
    gt_annot_dir = Path(config["paths"]["segmentation_annotations"])
    wsi_dir = Path(config["paths"]["raw_wsis"])

    # Setup directory for saving visualizations
    eval_config = config.get("evaluation", {})
    save_visuals = eval_config.get("save_debug_visuals", False)
    if save_visuals:
        viz_dir = Path("verification_outputs/evaluation")
        viz_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving evaluation visualizations to: {viz_dir.resolve()}")

    # We need the class map from the archived U-Net config to interpret the GeoJSONs
    if "segmentation" not in config:
        logger.critical("Missing 'segmentation' section in config.yaml for class mapping.");
        return
    class_map = config["segmentation"]["classes"]

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

        target_shape = pred_mask_1.shape
        gt_mask_multichannel = rasterize_ground_truth(gt_geojson_path, target_shape, wsi_dims, class_map, logger)
        if gt_mask_multichannel is None: continue

        gt_mask_1 = gt_mask_multichannel[:, :, 0].astype(bool)
        gt_mask_2 = gt_mask_multichannel[:, :, 1].astype(bool)
        pred_mask_1 = pred_mask_1.astype(bool)
        pred_mask_2 = pred_mask_2.astype(bool)

        # --- Intelligent Matching of Prediction to Ground Truth ---
        dice_A1, _, _ = _calculate_metrics(pred_mask_1, gt_mask_1)
        dice_A2, _, _ = _calculate_metrics(pred_mask_2, gt_mask_2)
        total_dice_A = dice_A1 + dice_A2

        dice_B1, _, _ = _calculate_metrics(pred_mask_1, gt_mask_2)
        dice_B2, _, _ = _calculate_metrics(pred_mask_2, gt_mask_1)
        total_dice_B = dice_B1 + dice_B2

        if total_dice_A >= total_dice_B:
            final_pred_1, final_gt_1 = pred_mask_1, gt_mask_1
            final_pred_2, final_gt_2 = pred_mask_2, gt_mask_2
        else:
            logger.info(f"Slide {slide_name}: Labels swapped. Matching P1->G2, P2->G1.")
            final_pred_1, final_gt_1 = pred_mask_1, gt_mask_2
            final_pred_2, final_gt_2 = pred_mask_2, gt_mask_1

        # NEW: Unpack all three metrics
        dice1, iou1, jf1 = _calculate_metrics(final_pred_1, final_gt_1)
        dice2, iou2, jf2 = _calculate_metrics(final_pred_2, final_gt_2)

        results.append({
            "slide_name": slide_name,
            "iou_oyster1": iou1, "dice_oyster1": dice1, "j_and_f_oyster1": jf1,
            "iou_oyster2": iou2, "dice_oyster2": dice2, "j_and_f_oyster2": jf2
        })

        if save_visuals:
            _save_evaluation_visualization(final_pred_1, final_gt_1, slide_name, 1, viz_dir)
            _save_evaluation_visualization(final_pred_2, final_gt_2, slide_name, 2, viz_dir)

    if not results:
        logger.error("No results were generated. Check paths and that ground truth GeoJSONs exist.");
        return

    # --- Display Final Report ---
    df = pd.DataFrame(results)
    avg_iou = df[['iou_oyster1', 'iou_oyster2']].values.mean()
    avg_dice = df[['dice_oyster1', 'dice_oyster2']].values.mean()
    # NEW: Calculate average J&F
    avg_j_and_f = df[['j_and_f_oyster1', 'j_and_f_oyster2']].values.mean()

    logger.info("\n\n--- Segmentation Performance Report ---")
    pd.set_option('display.precision', 4)
    # NEW: Update DataFrame columns for the report
    report_df = df[[
        "slide_name", "dice_oyster1", "iou_oyster1", "j_and_f_oyster1",
        "dice_oyster2", "iou_oyster2", "j_and_f_oyster2"
    ]]
    logger.info(f"\nPer-Slide Metrics:\n{report_df.to_string(index=False)}")
    logger.info("\n" + "---" * 10)
    logger.info(f"Overall Average IoU Score:   {avg_iou:.4f}")
    logger.info(f"Overall Average Dice Score:  {avg_dice:.4f}")
    logger.info(f"Overall Average J&F Score:   {avg_j_and_f:.4f}")
    logger.info("---" * 10)


if __name__ == "__main__":
    main()
