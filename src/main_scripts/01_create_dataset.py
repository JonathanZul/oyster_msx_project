# src/main_scripts/01_create_dataset.py

import os
import argparse
import glob
import json
import random
import shutil
import math
from pathlib import Path

import cv2
import numpy as np
import tifffile
import zarr
from shapely.geometry import shape
from tqdm import tqdm

# Import utility functions from our project
from src.utils.file_handling import load_config
from src.utils.logging_config import setup_logging


def _normalize_class_name(name: str | None) -> str:
    """Normalize annotation class names for robust matching."""
    if not name:
        return ""
    return " ".join(str(name).strip().lower().split())


def _normalize_slide_stem(stem: str | None) -> str:
    """Normalizes slide stems so '.ome' suffix is treated consistently."""
    if not stem:
        return ""
    s = str(stem).strip()
    return s[:-4] if s.endswith(".ome") else s


def _resolve_class_id(
    ann_class_name: str | None,
    class_map: dict,
    class_aliases: dict | None = None
):
    """
    Resolves an annotation class name to a configured class_id.
    Supports exact matches plus case-insensitive alias-based matching.
    """
    if ann_class_name in class_map:
        return class_map[ann_class_name]

    normalized_to_id = {_normalize_class_name(k): v for k, v in class_map.items()}
    alias_map = class_aliases or {}

    normalized_name = _normalize_class_name(ann_class_name)
    if normalized_name in normalized_to_id:
        return normalized_to_id[normalized_name]

    alias_target = alias_map.get(ann_class_name, alias_map.get(normalized_name))
    if alias_target is None:
        return None

    # Alias value can be a class name or direct class id.
    if isinstance(alias_target, int):
        return alias_target

    return class_map.get(alias_target, normalized_to_id.get(_normalize_class_name(alias_target)))


def _read_single_class_id_from_label(label_path: Path):
    """
    Reads a class ID from a YOLO label file.
    Returns None for empty/invalid/multi-class label files.
    """
    try:
        lines = [ln.strip() for ln in label_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except Exception:
        return None
    if not lines:
        return None

    class_ids = set()
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        try:
            class_ids.add(int(float(parts[0])))
        except ValueError:
            return None

    if len(class_ids) != 1:
        return None
    return next(iter(class_ids))


def _collect_subset_class_files(yolo_dataset_path: Path, subset: str):
    """
    Builds a mapping {class_id: [label_path,...]} for a dataset subset.
    """
    label_dir = yolo_dataset_path / "labels" / subset
    class_to_files = {}
    if not label_dir.exists():
        return class_to_files

    for label_path in label_dir.glob("*.txt"):
        class_id = _read_single_class_id_from_label(label_path)
        if class_id is None:
            continue
        class_to_files.setdefault(class_id, []).append(label_path)
    return class_to_files


def _log_subset_distribution(yolo_dataset_path: Path, subset: str, class_map: dict, logger, prefix: str = ""):
    class_to_files = _collect_subset_class_files(yolo_dataset_path, subset)
    class_names = {v: k for k, v in class_map.items()}
    if not class_to_files:
        logger.info(f"{prefix}{subset}: no labeled samples found.")
        return

    summary = ", ".join(
        f"{class_names.get(class_id, class_id)}={len(paths)}"
        for class_id, paths in sorted(class_to_files.items(), key=lambda x: x[0])
    )
    logger.info(f"{prefix}{subset} class distribution: {summary}")


def balance_training_subset(yolo_dataset_path: Path, config: dict, logger):
    """
    Rebalances only the training subset via minority oversampling.

    Expected config:
      dataset_creation:
        class_balance:
          enabled: true
          target_max_ratio: 2.0
          max_oversample_multiplier: 4.0
    """
    balance_cfg = config.get("dataset_creation", {}).get("class_balance", {})
    if not balance_cfg.get("enabled", False):
        logger.info("Class balancing disabled for training subset.")
        return

    target_max_ratio = float(balance_cfg.get("target_max_ratio", 2.0))
    max_oversample_multiplier = float(balance_cfg.get("max_oversample_multiplier", 4.0))
    if target_max_ratio < 1.0:
        logger.warning("Invalid class_balance.target_max_ratio (<1). Using 1.0.")
        target_max_ratio = 1.0
    if max_oversample_multiplier < 1.0:
        logger.warning("Invalid class_balance.max_oversample_multiplier (<1). Using 1.0.")
        max_oversample_multiplier = 1.0

    class_map = config["dataset_creation"]["classes"]
    class_names = {v: k for k, v in class_map.items()}
    class_to_files = _collect_subset_class_files(yolo_dataset_path, "train")
    if len(class_to_files) < 2:
        logger.info("Class balancing skipped: need at least 2 classes present in train subset.")
        return

    counts = {class_id: len(paths) for class_id, paths in class_to_files.items()}
    majority_class = max(counts, key=counts.get)
    majority_count = counts[majority_class]
    logger.info(
        f"Balancing train subset. Majority class '{class_names.get(majority_class, majority_class)}' has {majority_count} samples."
    )

    train_images_dir = yolo_dataset_path / "images" / "train"
    train_labels_dir = yolo_dataset_path / "labels" / "train"
    added_total = 0

    for class_id, files in class_to_files.items():
        current_count = len(files)
        desired_count = int(math.ceil(majority_count / target_max_ratio))
        target_count = min(
            max(current_count, desired_count),
            int(math.ceil(current_count * max_oversample_multiplier))
        )
        to_add = target_count - current_count
        if to_add <= 0:
            continue

        logger.info(
            f"Oversampling class '{class_names.get(class_id, class_id)}': "
            f"{current_count} -> {target_count} (adding {to_add})"
        )

        for dup_idx in range(to_add):
            src_label = files[dup_idx % current_count]
            src_image = train_images_dir / f"{src_label.stem}.png"
            if not src_image.exists():
                continue

            new_stem = f"{src_label.stem}_bal{dup_idx+1}"
            dst_label = train_labels_dir / f"{new_stem}.txt"
            dst_image = train_images_dir / f"{new_stem}.png"
            shutil.copy2(src_label, dst_label)
            shutil.copy2(src_image, dst_image)
            added_total += 1

    logger.info(f"Class balancing complete. Added {added_total} oversampled train patches.")


def setup_directories(base_path: Path, append_mode: bool, logger):
    """
    Cleans and creates the necessary directory structure for the YOLO dataset.
    If append_mode is True, it will NOT delete the existing directory.
    """
    if base_path.exists():
        if not append_mode:
            logger.info(f"Removing existing dataset directory: {base_path}")
            shutil.rmtree(base_path, ignore_errors=True)
        else:
            logger.info(f"Append mode enabled. Keeping existing dataset directory: {base_path}")

    logger.info(f"Creating new dataset directory structure at: {base_path}")
    (base_path / "images/train").mkdir(parents=True, exist_ok=True)
    (base_path / "images/val").mkdir(parents=True, exist_ok=True)
    (base_path / "images/test").mkdir(parents=True, exist_ok=True)
    (base_path / "images/all").mkdir(parents=True, exist_ok=True)
    (base_path / "labels/train").mkdir(parents=True, exist_ok=True)
    (base_path / "labels/val").mkdir(parents=True, exist_ok=True)
    (base_path / "labels/test").mkdir(parents=True, exist_ok=True)
    (base_path / "labels/all").mkdir(parents=True, exist_ok=True)


def _create_assignment_context(parent_masks: dict, boundary_tolerance_px: int):
    """
    Prepares mask and contour data for robust annotation-to-oyster assignment.
    """
    context = {}
    tol = max(0, int(boundary_tolerance_px))
    kernel = None
    if tol > 0:
        ksize = (2 * tol + 1, 2 * tol + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize)

    for oyster_id, mask in parent_masks.items():
        mask_binary = (mask > 0).astype(np.uint8)
        mask_u8 = (mask_binary * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dilated = cv2.dilate(mask_binary, kernel, iterations=1) if kernel is not None else mask_binary
        context[oyster_id] = {
            "mask": mask_binary,
            "dilated": dilated,
            "contours": contours,
        }
    return context


def _compute_best_signed_distance(contours, x: int, y: int) -> float:
    """
    Returns the best signed contour distance for a point.
    Positive means inside, negative means outside.
    """
    if not contours:
        return float("-inf")
    point = (float(x), float(y))
    return max(cv2.pointPolygonTest(cnt, point, True) for cnt in contours)


def _assign_from_scaled_point(
    x: int,
    y: int,
    assignment_context: dict,
    nearest_oyster_max_distance_px: float
):
    """
    Assigns an oyster ID using:
      1) direct mask hit
      2) boundary-tolerant dilated mask hit
      3) nearest contour fallback within threshold
    """
    if not assignment_context:
        return None, "no_context", float("-inf")

    any_entry = next(iter(assignment_context.values()))
    h, w = any_entry["mask"].shape
    if not (0 <= y < h and 0 <= x < w):
        return None, "out_of_bounds", float("-inf")

    direct_scores = {}
    tolerant_scores = {}
    nearest_scores = {}

    for oyster_id, entry in assignment_context.items():
        signed_distance = _compute_best_signed_distance(entry["contours"], x, y)
        nearest_scores[oyster_id] = signed_distance

        if entry["mask"][y, x] > 0:
            direct_scores[oyster_id] = signed_distance
        elif entry["dilated"][y, x] > 0:
            tolerant_scores[oyster_id] = signed_distance

    if direct_scores:
        oyster_id = max(direct_scores, key=direct_scores.get)
        return oyster_id, "direct", direct_scores[oyster_id]

    if tolerant_scores:
        oyster_id = max(tolerant_scores, key=tolerant_scores.get)
        return oyster_id, "boundary_tolerant", tolerant_scores[oyster_id]

    if nearest_scores and nearest_oyster_max_distance_px > 0:
        oyster_id = max(nearest_scores, key=nearest_scores.get)
        best_signed = nearest_scores[oyster_id]
        if best_signed >= -float(nearest_oyster_max_distance_px):
            return oyster_id, "nearest_fallback", best_signed
        return None, "outside_fallback_radius", best_signed

    best_signed = max(nearest_scores.values()) if nearest_scores else float("-inf")
    return None, "no_mask_match", best_signed


def _save_skipped_annotations_overlay(
    slide_stem: str,
    assignment_context: dict,
    skipped_points: list[dict],
    output_dir: Path,
    max_points: int,
    logger
):
    """
    Saves a debug overlay showing mask regions and skipped annotation centroids.
    """
    if not skipped_points or not assignment_context:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    any_entry = next(iter(assignment_context.values()))
    h, w = any_entry["mask"].shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    mask_colors = [(255, 80, 80), (80, 255, 80), (80, 80, 255), (255, 220, 80)]
    for idx, (oyster_id, entry) in enumerate(sorted(assignment_context.items(), key=lambda item: item[0])):
        color = mask_colors[idx % len(mask_colors)]
        overlay = np.zeros_like(canvas)
        overlay[entry["mask"] > 0] = color
        canvas = cv2.addWeighted(canvas, 1.0, overlay, 0.30, 0)
        if entry["contours"]:
            cv2.drawContours(canvas, entry["contours"], -1, color, 1)
        cv2.putText(canvas, f"Oyster {oyster_id}", (10, 28 + idx * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    shown = 0
    out_of_bounds = 0
    for info in skipped_points[:max(1, int(max_points))]:
        x, y = int(info["scaled_x"]), int(info["scaled_y"])
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(canvas, (x, y), 2, (0, 0, 255), -1)
            shown += 1
        else:
            out_of_bounds += 1

    cv2.putText(
        canvas,
        f"Skipped points shown: {shown} / {len(skipped_points)} (oob skipped: {out_of_bounds})",
        (10, h - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )
    out_path = output_dir / f"{slide_stem}_skipped_assignment_overlay.png"
    cv2.imwrite(str(out_path), canvas)
    logger.info(f"Saved skipped-annotation debug overlay: {out_path}")


def get_oyster_id_for_annotation(
    annotation,
    assignment_context: dict,
    scale_x: float,
    scale_y: float,
    nearest_oyster_max_distance_px: float
):
    """
    Determines which oyster an annotation belongs to using robust assignment rules.
    """
    geom = shape(annotation["geometry"])
    centroid = geom.centroid

    scaled_centroid_x = int(centroid.x / max(scale_x, 1e-9))
    scaled_centroid_y = int(centroid.y / max(scale_y, 1e-9))
    oyster_id, mode, signed_distance = _assign_from_scaled_point(
        scaled_centroid_x,
        scaled_centroid_y,
        assignment_context,
        nearest_oyster_max_distance_px
    )
    return oyster_id, {
        "centroid_x": float(centroid.x),
        "centroid_y": float(centroid.y),
        "scaled_x": int(scaled_centroid_x),
        "scaled_y": int(scaled_centroid_y),
        "assignment_mode": mode,
        "best_signed_distance": float(signed_distance),
    }


def split_slides(all_slides, config, logger):
    """
    Assigns slides to train, val, and test sets based on config.

    Args:
        all_slides (list): List of slide names (stems).
        config (dict): Configuration dictionary.
        logger: Logger instance.

    Returns:
        dict: Mapping of slide_name -> subset ('train', 'val', 'test')
    """
    train_slides_config = {_normalize_slide_stem(s) for s in config['dataset_creation'].get('train_slides', [])}
    val_slides_config = {_normalize_slide_stem(s) for s in config['dataset_creation'].get('validation_slides', [])}
    test_slides_config = {_normalize_slide_stem(s) for s in config['dataset_creation'].get('test_slides', [])}

    slide_assignments = {}
    remaining_slides = []

    # 1. Assign explicitly defined slides (train takes priority over val/test)
    for slide in all_slides:
        slide_key = _normalize_slide_stem(slide)
        if slide_key in train_slides_config:
            slide_assignments[slide] = 'train'
        elif slide_key in val_slides_config:
            slide_assignments[slide] = 'val'
        elif slide_key in test_slides_config:
            slide_assignments[slide] = 'test'
        else:
            remaining_slides.append(slide)

    # 2. Randomly split the rest if needed
    if remaining_slides:
        # Logic: If ALL explicit lists are empty, do full random split.
        # If at least one list is provided, assume the user is managing splits manually
        # and put everything else in train.

        if not train_slides_config and not val_slides_config and not test_slides_config:
            random.shuffle(remaining_slides)
            n_total = len(remaining_slides)
            ratios = config['dataset_creation'].get('train_val_test_split', [0.7, 0.15, 0.15])

            n_train = int(n_total * ratios[0])
            n_val = int(n_total * ratios[1])
            # n_test is the rest

            train_set = remaining_slides[:n_train]
            val_set = remaining_slides[n_train:n_train+n_val]
            test_set = remaining_slides[n_train+n_val:]

            for s in train_set: slide_assignments[s] = 'train'
            for s in val_set: slide_assignments[s] = 'val'
            for s in test_set: slide_assignments[s] = 'test'

            logger.info(f"Random Split: {len(train_set)} Train, {len(val_set)} Val, {len(test_set)} Test")
        else:
            # Explicit mode: everything else is train
            for s in remaining_slides:
                slide_assignments[s] = 'train'
            logger.info(f"Explicit Split: {len(remaining_slides)} remaining slides assigned to Train.")

    # Log explicit assignments
    all_slide_keys = {_normalize_slide_stem(s) for s in all_slides}
    n_explicit_train = len(train_slides_config & all_slide_keys)
    n_explicit_val = len(val_slides_config & all_slide_keys)
    n_explicit_test = len(test_slides_config & all_slide_keys)
    if n_explicit_train or n_explicit_val or n_explicit_test:
        logger.info(f"Explicit Assignments: {n_explicit_train} Train, {n_explicit_val} Val, {n_explicit_test} Test")

    return slide_assignments


def filter_geojson_files_by_allowed_slides(geojson_files, config, logger):
    """
    Optionally restrict dataset creation to an explicit whitelist of slide stems.
    """
    allowed_slides = config["dataset_creation"].get("allowed_slides", [])
    if not allowed_slides:
        return geojson_files

    allowed_set = {_normalize_slide_stem(s) for s in allowed_slides if str(s).strip()}
    if not allowed_set:
        logger.warning("dataset_creation.allowed_slides was provided but empty after normalization. Using all slides.")
        return geojson_files

    filtered = [p for p in geojson_files if _normalize_slide_stem(p.stem) in allowed_set]
    excluded = [p.stem for p in geojson_files if _normalize_slide_stem(p.stem) not in allowed_set]

    logger.info(
        f"Allowed-slides filter active: keeping {len(filtered)}/{len(geojson_files)} slides."
    )
    if excluded:
        logger.info(f"Excluded {len(excluded)} slides not in allowed_slides list.")

    missing_from_data = sorted(list(allowed_set - {_normalize_slide_stem(p.stem) for p in geojson_files}))
    if missing_from_data:
        logger.warning(
            f"{len(missing_from_data)} allowed slides were not found in qupath exports: {missing_from_data[:10]}"
        )

    return filtered


def process_single_wsi(
    wsi_path: Path, geojson_path: Path, parent_mask_paths: list, config: dict, logger
):
    """
    Processes a single WSI and its annotations to generate oyster-aware patches.
    """
    try:
        logger.info(f"Opening WSI: {wsi_path.name}")
        with tifffile.TiffFile(wsi_path) as tif:
            level0 = tif.series[0].levels[0]
            img_height, img_width = level0.shape[:2]
            logger.debug(f"WSI dimensions (H, W): ({img_height}, {img_width})")

            zarr_store = tif.series[0].aszarr()
            zarr_group = zarr.open(zarr_store, mode='r')
            zarr_slicer = zarr_group["0"]
            logger.info(f"Successfully created Zarr slicer for Level 0.")

            # Load parent masks and calculate downsample factor
            parent_masks = {}
            for mask_path in parent_mask_paths:
                # Extract oyster ID from filename like 'oyster_1_mask.png'
                oyster_id = int(mask_path.stem.split("_")[1])
                mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask_img is not None:
                    parent_masks[oyster_id] = mask_img

            if not parent_masks:
                logger.error(
                    f"No valid oyster masks found for {wsi_path.name}. Skipping."
                )
                return

            # Calculate coordinate-space scaling factors between Level-0 WSI and mask.
            mask_height, mask_width = next(iter(parent_masks.values())).shape
            scale_x = img_width / mask_width
            scale_y = img_height / mask_height
            logger.info(
                f"Coordinate scaling: scale_x={scale_x:.4f}, scale_y={scale_y:.4f} "
                f"(WSI={img_width}x{img_height}, mask={mask_width}x{mask_height})"
            )
            anisotropy = abs(scale_x - scale_y) / max(scale_x, scale_y, 1e-9)
            if anisotropy > 0.05:
                logger.warning(
                    f"Significant scale anisotropy detected ({anisotropy * 100:.2f}%). "
                    "This may indicate coordinate-space mismatch."
                )

            assignment_cfg = config["dataset_creation"].get("annotation_assignment", {})
            boundary_tolerance_px = int(assignment_cfg.get("boundary_tolerance_px", 3))
            nearest_oyster_max_distance_px = float(assignment_cfg.get("nearest_oyster_max_distance_px", 8.0))
            log_each_unassigned = bool(assignment_cfg.get("log_each_unassigned", False))
            save_skipped_overlay = bool(assignment_cfg.get("save_skipped_overlay", True))
            max_overlay_points = int(assignment_cfg.get("max_overlay_points", 3000))

            assignment_context = _create_assignment_context(parent_masks, boundary_tolerance_px)
            logger.info(
                f"Assignment settings: boundary_tolerance_px={boundary_tolerance_px}, "
                f"nearest_oyster_max_distance_px={nearest_oyster_max_distance_px}"
            )

            # Load GeoJSON data
            with open(geojson_path) as f:
                annotations = json.load(f)["features"]
            logger.info(f"Found {len(annotations)} annotations in {geojson_path.name}")

            patch_size = config["dataset_creation"]["patch_size"]
            class_map = config["dataset_creation"]["classes"]
            class_aliases = config["dataset_creation"].get("class_aliases", {})
            yolo_base_path = Path(config["paths"]["yolo_dataset"])
            patches_created = 0
            extracted_records = []
            skipped_points = []
            assignment_counts = {
                "direct": 0,
                "boundary_tolerant": 0,
                "nearest_fallback": 0,
                "unassigned": 0,
                "unmapped_class": 0,
            }

            for ann_idx, ann in enumerate(tqdm(annotations, desc=f"Processing {wsi_path.stem}")):
                try:
                    ann_class_name = (
                        ann["properties"].get("classification", {}).get("name")
                    )
                    class_id = _resolve_class_id(ann_class_name, class_map, class_aliases)
                    if class_id is None:
                        assignment_counts["unmapped_class"] += 1
                        if not hasattr(process_single_wsi, "warned_classes"):
                            process_single_wsi.warned_classes = set()
                        if ann_class_name not in process_single_wsi.warned_classes:
                            logger.warning(f"Skipping annotation with unmapped class: '{ann_class_name}'. "
                                           f"Ensure this name exists in your config.yaml `classes` map.")
                            process_single_wsi.warned_classes.add(ann_class_name)
                        continue

                    oyster_id, assignment_debug = get_oyster_id_for_annotation(
                        ann,
                        assignment_context,
                        scale_x=scale_x,
                        scale_y=scale_y,
                        nearest_oyster_max_distance_px=nearest_oyster_max_distance_px
                    )
                    if oyster_id is None:
                        assignment_counts["unassigned"] += 1
                        skipped_points.append(assignment_debug)
                        if log_each_unassigned:
                            logger.warning(
                                "Annotation at "
                                f"({assignment_debug['centroid_x']:.2f}, {assignment_debug['centroid_y']:.2f}) "
                                f"scaled=({assignment_debug['scaled_x']}, {assignment_debug['scaled_y']}) "
                                f"did not fall within any oyster mask "
                                f"(mode={assignment_debug['assignment_mode']}, "
                                f"best_signed_distance={assignment_debug['best_signed_distance']:.2f}). Skipping."
                            )
                        continue
                    assignment_mode = assignment_debug["assignment_mode"]
                    if assignment_mode in assignment_counts:
                        assignment_counts[assignment_mode] += 1

                    geom = shape(ann["geometry"])
                    g_x_min, g_y_min, g_x_max, g_y_max = geom.bounds
                    center_x, center_y = (
                        (g_x_min + g_x_max) / 2,
                        (g_y_min + g_y_max) / 2,
                    )
                    p_x_min, p_y_min = (
                        int(center_x - patch_size / 2),
                        int(center_y - patch_size / 2),
                    )
                    p_x_max, p_y_max = p_x_min + patch_size, p_y_min + patch_size

                    if (
                        p_x_min < 0
                        or p_y_min < 0
                        or p_x_max > img_width
                        or p_y_max > img_height
                    ):
                        logger.warning(
                            f"Skipping annotation at ({center_x}, {center_y}) - patch out of bounds."
                        )
                        continue

                    patches_created += 1

                    p_x_max, p_y_max = p_x_min + patch_size, p_y_min + patch_size
                    image_patch = zarr_slicer[p_y_min:p_y_max, p_x_min:p_x_max]

                    if image_patch.ndim == 3 and image_patch.shape[2] == 4:
                        image_patch = cv2.cvtColor(image_patch, cv2.COLOR_RGBA2BGR)
                    elif image_patch.ndim == 2:
                        image_patch = cv2.cvtColor(image_patch, cv2.COLOR_GRAY2BGR)

                    # Patch-first workflow: write all annotations into a staging pool,
                    # then perform a dataset split from extracted patches.
                    patch_filename = f"{wsi_path.stem}_oyster_{oyster_id}_patch_x{p_x_min}_y{p_y_min}_ann{ann_idx}"

                    cv2.imwrite(
                        str(yolo_base_path / f"images/all/{patch_filename}.png"),
                        image_patch,
                    )

                    l_x_min, l_y_min = g_x_min - p_x_min, g_y_min - p_y_min
                    l_x_max, l_y_max = g_x_max - p_x_min, g_y_max - p_y_min
                    yolo_center_x = ((l_x_min + l_x_max) / 2) / patch_size
                    yolo_center_y = ((l_y_min + l_y_max) / 2) / patch_size
                    yolo_width = (l_x_max - l_x_min) / patch_size
                    yolo_height = (l_y_max - l_y_min) / patch_size

                    label_path = (
                        yolo_base_path / f"labels/all/{patch_filename}.txt"
                    )
                    with open(label_path, "w") as f_label:
                        f_label.write(
                            f"{class_id} {yolo_center_x} {yolo_center_y} {yolo_width} {yolo_height}\n"
                        )
                    extracted_records.append({
                        "filename": patch_filename,
                        "class_id": int(class_id),
                    })
                except Exception as e:
                    logger.error(
                        f"Failed to process a single annotation. Error: {e}",
                        exc_info=True,
                    )
                    continue

            logger.info(
                "Assignment summary for "
                f"{wsi_path.stem}: direct={assignment_counts['direct']}, "
                f"boundary_tolerant={assignment_counts['boundary_tolerant']}, "
                f"nearest_fallback={assignment_counts['nearest_fallback']}, "
                f"unassigned={assignment_counts['unassigned']}, "
                f"unmapped_class={assignment_counts['unmapped_class']}, "
                f"patches_created={patches_created}"
            )
            if save_skipped_overlay and skipped_points:
                _save_skipped_annotations_overlay(
                    slide_stem=wsi_path.stem,
                    assignment_context=assignment_context,
                    skipped_points=skipped_points,
                    output_dir=Path(config["paths"]["logs"]) / "dataset_assignment_debug",
                    max_points=max_overlay_points,
                    logger=logger
                )
            return extracted_records
    except Exception as e:
        logger.error(
            f"An unexpected error occurred while processing {wsi_path.name}: {e}",
            exc_info=True,
        )
    return []


def split_extracted_patches(yolo_dataset_path: Path, extracted_records: list[dict], config: dict, logger):
    """
    Splits extracted annotation patches into train/val/test at patch level.
    Uses per-class stratified splitting so each subset receives samples of each class.
    """
    if not extracted_records:
        logger.warning("No extracted records available for splitting.")
        return

    ratios = config["dataset_creation"].get("train_val_test_split", [0.7, 0.15, 0.15])
    if len(ratios) != 3:
        logger.warning("Invalid train_val_test_split (expected 3 values). Falling back to [0.7, 0.15, 0.15].")
        ratios = [0.7, 0.15, 0.15]
    ratio_sum = sum(ratios)
    if ratio_sum <= 0:
        logger.warning("Invalid train_val_test_split sum <= 0. Falling back to [0.7, 0.15, 0.15].")
        ratios = [0.7, 0.15, 0.15]
        ratio_sum = 1.0
    ratios = [r / ratio_sum for r in ratios]

    random.seed(config.get("cross_validation", {}).get("data_split_seed", 42))

    by_class = {}
    for rec in extracted_records:
        by_class.setdefault(rec["class_id"], []).append(rec["filename"])

    assignments = {"train": [], "val": [], "test": []}
    for class_id, filenames in by_class.items():
        random.shuffle(filenames)
        n = len(filenames)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        n_test = n - n_train - n_val

        assignments["train"].extend(filenames[:n_train])
        assignments["val"].extend(filenames[n_train:n_train + n_val])
        assignments["test"].extend(filenames[n_train + n_val:n_train + n_val + n_test])
        logger.info(
            f"Class {class_id}: total={n}, train={n_train}, val={n_val}, test={n_test}"
        )

    src_img_dir = yolo_dataset_path / "images" / "all"
    src_lbl_dir = yolo_dataset_path / "labels" / "all"

    for subset, names in assignments.items():
        dst_img_dir = yolo_dataset_path / "images" / subset
        dst_lbl_dir = yolo_dataset_path / "labels" / subset
        for name in names:
            src_img = src_img_dir / f"{name}.png"
            src_lbl = src_lbl_dir / f"{name}.txt"
            dst_img = dst_img_dir / f"{name}.png"
            dst_lbl = dst_lbl_dir / f"{name}.txt"
            if src_img.exists() and src_lbl.exists():
                shutil.move(src_img, dst_img)
                shutil.move(src_lbl, dst_lbl)

    shutil.rmtree(src_img_dir, ignore_errors=True)
    shutil.rmtree(src_lbl_dir, ignore_errors=True)
    logger.info(
        f"Patch-level split complete: train={len(assignments['train'])}, "
        f"val={len(assignments['val'])}, test={len(assignments['test'])}"
    )


def main():
    """Main function to execute the dataset creation process."""
    parser = argparse.ArgumentParser(description="Stage 01: Dataset Creation")
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="config.yaml",
        help="Path to the master configuration file."
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if not config:
        return

    logger = setup_logging(Path(config["paths"]["logs"]), "01_create_dataset")
    logger.info("--- Starting Dataset Creation Script ---")

    yolo_dataset_path = Path(config["paths"]["yolo_dataset"])
    append_mode = config["dataset_creation"].get("append_mode", False)
    setup_directories(yolo_dataset_path, append_mode, logger)

    qupath_exports_dir = Path(config["paths"]["qupath_exports"])
    geojson_files = list(qupath_exports_dir.glob("*.geojson"))
    if not geojson_files:
        logger.error(f"No GeoJSON files found in '{qupath_exports_dir}'. Exiting.")
        return

    geojson_files = filter_geojson_files_by_allowed_slides(geojson_files, config, logger)
    if not geojson_files:
        logger.error("No GeoJSON files remain after applying allowed_slides filter. Exiting.")
        return

    logger.info(f"Found {len(geojson_files)} GeoJSON files to process.")
    
    logger.info("Dataset split mode: patch-level (stratified by class).")
    extracted_records = []

    for geojson_path in geojson_files:
        wsi_name_stem = geojson_path.stem
        wsi_name_key = _normalize_slide_stem(wsi_name_stem)

        # Find matching WSI file
        possible_wsi_paths = list(Path(config["paths"]["raw_wsis"]).glob(f"{wsi_name_stem}.*"))
        if not possible_wsi_paths:
            possible_wsi_paths = list(Path(config["paths"]["raw_wsis"]).glob(f"{wsi_name_key}.*"))
        wsi_path = next(
            (
                p
                for p in possible_wsi_paths
                if p.suffix.lower() in [".tif", ".tiff", ".vsi"]
            ),
            None,
        )
        if not wsi_path:
            logger.warning(
                f"Could not find a matching WSI for '{geojson_path.name}' (normalized stem: '{wsi_name_key}'). Skipping."
            )
            continue

        # Find matching oyster masks for this slide
        mask_dir = Path(config["paths"]["oyster_masks"]) / wsi_name_key
        if not mask_dir.exists():
            logger.warning(
                f"No mask directory found for '{wsi_name_stem}' at '{mask_dir}'. Skipping."
            )
            continue

        parent_mask_paths = list(mask_dir.glob("oyster_*_mask.png"))
        if not parent_mask_paths:
            logger.warning(f"No oyster masks found in '{mask_dir}'. Skipping.")
            continue

        logger.info(f"Found {len(parent_mask_paths)} oyster masks for this slide.")
        extracted_records.extend(
            process_single_wsi(wsi_path, geojson_path, parent_mask_paths, config, logger)
        )

    split_extracted_patches(yolo_dataset_path, extracted_records, config, logger)

    class_map = config["dataset_creation"]["classes"]
    _log_subset_distribution(yolo_dataset_path, "train", class_map, logger, prefix="Before balancing - ")
    _log_subset_distribution(yolo_dataset_path, "val", class_map, logger, prefix="Before balancing - ")
    _log_subset_distribution(yolo_dataset_path, "test", class_map, logger, prefix="Before balancing - ")

    balance_training_subset(yolo_dataset_path, config, logger)

    _log_subset_distribution(yolo_dataset_path, "train", class_map, logger, prefix="After balancing - ")
    logger.info("--- Dataset Creation Script Finished ---")


if __name__ == "__main__":
    main()
