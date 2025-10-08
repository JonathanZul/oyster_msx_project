# src/utils/evaluation_utils.py

import json
from pathlib import Path

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch

# Add src to path to allow for relative imports from tools
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from archive.unet_00a_create_dataset import find_polygon_coords


def calculate_segmentation_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
    """
    Calculates Dice, IoU, and J&F scores for a single pair of boolean masks.
    This is the centralized, definitive metric calculation function for the project.
    """
    if pred_mask.shape != gt_mask.shape:
        # This can happen if a mask is empty; return zero scores.
        return {"dice": 0.0, "iou": 0.0, "j_and_f": 0.0}

    pred_mask_bool = pred_mask > 0
    gt_mask_bool = gt_mask > 0

    intersection = np.sum(np.logical_and(pred_mask_bool, gt_mask_bool))
    pred_sum = np.sum(pred_mask_bool)
    gt_sum = np.sum(gt_mask_bool)

    epsilon = 1e-6
    dice = (2.0 * intersection) / (pred_sum + gt_sum + epsilon)
    iou = intersection / (pred_sum + gt_sum - intersection + epsilon)
    j_and_f = (iou + dice) / 2.0

    return {"dice": dice, "iou": iou, "j_and_f": j_and_f}


def rasterize_ground_truth(geojson_path: Path, target_shape: tuple, wsi_dims: tuple, class_map: dict,
                           logger) -> np.ndarray | None:
    """
    Loads a GeoJSON file and rasterizes annotations into a multi-channel ground truth mask.
    """
    target_h, target_w = target_shape
    wsi_w, wsi_h = wsi_dims
    num_classes = len(class_map)
    gt_mask = np.zeros((target_h, target_w, num_classes), dtype=np.uint8)

    try:
        with open(geojson_path, "r") as f:
            annotations = json.load(f)["features"]

        scale_x = target_w / wsi_w
        scale_y = target_h / wsi_h

        for ann in annotations:
            class_name = ann.get("properties", {}).get("classification", {}).get("name")
            if class_name not in class_map: continue

            # YAML class map is 1-based, array is 0-based.
            channel_idx = class_map[class_name] - 1
            raw_coords = find_polygon_coords(ann["geometry"]["coordinates"])
            if raw_coords is None: continue

            coords_scaled = np.array(raw_coords, dtype=np.float32)
            coords_scaled[:, 0] *= scale_x
            coords_scaled[:, 1] *= scale_y

            cv2.fillPoly(gt_mask[:, :, channel_idx], [coords_scaled.astype(np.int32)], 1)

        return gt_mask
    except Exception as e:
        logger.error(f"Failed to process ground truth GeoJSON {geojson_path.name}: {e}", exc_info=True)
        return None


def intelligently_match_masks(pred_mask_1, pred_mask_2, gt_mask_1, gt_mask_2, logger, slide_name):
    """
    Matches predicted masks to ground truth masks to handle potential label swapping.
    Returns the correctly ordered prediction and ground truth pairs.
    """
    # Case A: pred_1 -> gt_1, pred_2 -> gt_2
    metrics_A1 = calculate_segmentation_metrics(pred_mask_1, gt_mask_1)
    metrics_A2 = calculate_segmentation_metrics(pred_mask_2, gt_mask_2)
    total_dice_A = metrics_A1["dice"] + metrics_A2["dice"]

    # Case B: pred_1 -> gt_2, pred_2 -> gt_1
    metrics_B1 = calculate_segmentation_metrics(pred_mask_1, gt_mask_2)
    metrics_B2 = calculate_segmentation_metrics(pred_mask_2, gt_mask_1)
    total_dice_B = metrics_B1["dice"] + metrics_B2["dice"]

    if total_dice_A >= total_dice_B:
        return pred_mask_1, gt_mask_1, pred_mask_2, gt_mask_2
    else:
        logger.info(f"Slide {slide_name}: Labels swapped during evaluation. Matching P1->G2, P2->G1.")
        return pred_mask_1, gt_mask_2, pred_mask_2, gt_mask_1
