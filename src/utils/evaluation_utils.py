# src/utils/evaluation_utils.py
"""
Provides core utility functions for evaluating segmentation model performance.

This module includes functions for calculating standard segmentation metrics,
rasterizing vector-based ground truth annotations into pixel masks, and
logically matching predicted masks to ground truth masks.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch

# Add src to path to allow for relative imports from other project directories.
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))


def find_polygon_coords(coordinates_list: list):
    """
    Recursively searches a nested list to find the first list that contains
    the actual polygon coordinates (a list of [x, y] pairs).
    This handles inconsistencies in GeoJSON nesting from QuPath.

    Args:
        coordinates_list (list): The potentially nested list of coordinates.

    Returns:
        list | None: The list of [x, y] pairs if found, else None.
    """
    # Base case: The list contains elements that are [x, y] pairs.
    if all(isinstance(elem, list) and len(elem) == 2 and isinstance(elem[0], (int, float)) for elem in
           coordinates_list):
        return coordinates_list
    # Recursive step: If the list is not the coordinate list, search its first element.
    if isinstance(coordinates_list, list) and len(coordinates_list) > 0:
        return find_polygon_coords(coordinates_list[0])
    return None


def calculate_segmentation_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
    """Calculates Dice, IoU, and J&F scores for a pair of boolean masks.

    This function serves as the project's standard for calculating segmentation
    performance metrics.

    Args:
        pred_mask: The predicted segmentation mask (boolean or binary).
        gt_mask: The ground truth segmentation mask (boolean or binary).

    Returns:
        A dictionary containing the 'dice', 'iou', and 'j_and_f' scores.
        Returns zero for all scores if mask shapes are mismatched.
    """
    if pred_mask.shape != gt_mask.shape:
        # This can happen if a method fails to produce a mask; return zero scores.
        return {"dice": 0.0, "iou": 0.0, "j_and_f": 0.0}

    pred_mask_bool = pred_mask > 0
    gt_mask_bool = gt_mask > 0

    intersection = np.sum(np.logical_and(pred_mask_bool, gt_mask_bool))
    pred_sum = np.sum(pred_mask_bool)
    gt_sum = np.sum(gt_mask_bool)

    # A small epsilon is added to avoid division by zero if both masks are empty.
    epsilon = 1e-6
    dice = (2.0 * intersection) / (pred_sum + gt_sum + epsilon)
    iou = intersection / (pred_sum + gt_sum - intersection + epsilon)
    # The J&F score is a simple average of the Dice and IoU coefficients.
    j_and_f = (iou + dice) / 2.0

    return {"dice": dice, "iou": iou, "j_and_f": j_and_f}


def rasterize_ground_truth(geojson_path: Path, target_shape: tuple, wsi_dims: tuple, class_map: dict,
                           logger) -> np.ndarray | None:
    """Loads a GeoJSON file and rasterizes its polygons into a multi-channel mask.

    Each channel in the output mask corresponds to a specific class defined in
    the class_map. The function scales the polygon coordinates from the
    whole-slide image's absolute dimensions to the target mask's dimensions.

    Args:
        geojson_path: Path to the GeoJSON annotation file.
        target_shape: The (height, width) of the desired output mask.
        wsi_dims: The (width, height) of the original whole-slide image.
        class_map: A dictionary mapping class names to integer channel indices.
        logger: A logger instance for reporting errors.

    Returns:
        A multi-channel numpy array (H, W, C) representing the ground truth
        mask, or None if an error occurs.
    """
    target_h, target_w = target_shape
    wsi_w, wsi_h = wsi_dims
    num_classes = len(class_map)
    gt_mask = np.zeros((target_h, target_w, num_classes), dtype=np.uint8)

    try:
        with open(geojson_path, "r") as f:
            annotations = json.load(f)["features"]

        # Calculate scaling factors to map WSI coordinates to the target mask shape.
        scale_x = target_w / wsi_w
        scale_y = target_h / wsi_h

        for ann in annotations:
            class_name = ann.get("properties", {}).get("classification", {}).get("name")
            if class_name not in class_map:
                continue

            # The class map in the config is 1-based, but array indices are 0-based.
            channel_idx = class_map[class_name] - 1
            raw_coords = find_polygon_coords(ann["geometry"]["coordinates"])
            if raw_coords is None:
                continue

            coords_scaled = np.array(raw_coords, dtype=np.float32)
            coords_scaled[:, 0] *= scale_x
            coords_scaled[:, 1] *= scale_y

            # OpenCV's fillPoly requires a C-contiguous array. Slicing can create
            # a non-contiguous view, so we make an explicit copy.
            mask_channel_slice = gt_mask[:, :, channel_idx].copy()
            cv2.fillPoly(mask_channel_slice, [coords_scaled.astype(np.int32)], 1)
            gt_mask[:, :, channel_idx] = mask_channel_slice

        return gt_mask
    except Exception as e:
        logger.error(f"Failed to process ground truth GeoJSON {geojson_path.name}: {e}", exc_info=True)
        return None


def intelligently_match_masks(pred_mask_1, pred_mask_2, gt_mask_1, gt_mask_2, logger, slide_name):
    """Matches predicted masks to ground truths to find the optimal pairing.

    A segmentation model outputs two masks but doesn't assign them a fixed label
    (e.g., "oyster_1" vs "oyster_2"). This function checks both possible
    assignments and chooses the one that maximizes the total Dice score, ensuring
    a fair evaluation.

    Args:
        pred_mask_1: The first predicted mask.
        pred_mask_2: The second predicted mask.
        gt_mask_1: The first ground truth mask.
        gt_mask_2: The second ground truth mask.
        logger: Logger for reporting when a swap occurs.
        slide_name: Name of the slide being evaluated, for logging.

    Returns:
        A tuple of the correctly ordered masks:
        (pred_mask_for_gt1, gt_mask_1, pred_mask_for_gt2, gt_mask_2).
    """
    # Calculate total score for the default pairing (P1->G1, P2->G2).
    metrics_A1 = calculate_segmentation_metrics(pred_mask_1, gt_mask_1)
    metrics_A2 = calculate_segmentation_metrics(pred_mask_2, gt_mask_2)
    total_dice_A = metrics_A1["dice"] + metrics_A2["dice"]

    # Calculate total score for the swapped pairing (P1->G2, P2->G1).
    metrics_B1 = calculate_segmentation_metrics(pred_mask_1, gt_mask_2)
    metrics_B2 = calculate_segmentation_metrics(pred_mask_2, gt_mask_1)
    total_dice_B = metrics_B1["dice"] + metrics_B2["dice"]

    if total_dice_A >= total_dice_B:
        # Default pairing is best.
        return pred_mask_1, gt_mask_1, pred_mask_2, gt_mask_2
    else:
        # Swapped pairing is best.
        logger.info(f"Slide {slide_name}: Labels swapped during evaluation. Matching P1->G2, P2->G1.")
        return pred_mask_1, gt_mask_2, pred_mask_2, gt_mask_1
