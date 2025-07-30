# src/main_scripts/01_create_dataset.py

import os
import argparse
import glob
import json
import random
import shutil
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


def setup_directories(base_path: Path, logger):
    """Cleans and creates the necessary directory structure for the YOLO dataset."""
    if base_path.exists():
        logger.info(f"Removing existing dataset directory: {base_path}")
        shutil.rmtree(base_path)

    logger.info(f"Creating new dataset directory structure at: {base_path}")
    (base_path / "images/train").mkdir(parents=True, exist_ok=True)
    (base_path / "images/val").mkdir(parents=True, exist_ok=True)
    (base_path / "labels/train").mkdir(parents=True, exist_ok=True)
    (base_path / "labels/val").mkdir(parents=True, exist_ok=True)


def get_oyster_id_for_annotation(annotation, parent_masks, downsample_factor, logger):
    """
    Determines which oyster an annotation belongs to by checking against instance masks.

    Args:
        annotation (dict): A single GeoJSON feature.
        parent_masks (dict): A dictionary mapping oyster_id to its mask array.
        downsample_factor (float): The factor to scale high-res coordinates down.
        logger: The logger instance.

    Returns:
        int: The ID of the oyster (e.g., 1, 2) or None if not found.
    """
    geom = shape(annotation["geometry"])
    centroid = geom.centroid  # This gives the center of the annotation

    # Scale the high-res annotation centroid down to the low-res mask coordinate system
    scaled_centroid_x = int(centroid.x / downsample_factor)
    scaled_centroid_y = int(centroid.y / downsample_factor)

    for oyster_id, mask in parent_masks.items():
        # Check if the scaled coordinate is within the mask bounds
        if (
            0 <= scaled_centroid_y < mask.shape[0]
            and 0 <= scaled_centroid_x < mask.shape[1]
        ):
            # Check if the pixel at that coordinate in the mask is white (part of the oyster)
            if mask[scaled_centroid_y, scaled_centroid_x] > 0:
                return oyster_id

    logger.warning(
        f"Annotation at ({centroid.x}, {centroid.y}) did not fall within any oyster mask. Skipping."
    )
    return None


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
            zarr_slicer = zarr_group[0]
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

            # Calculate the downsample factor
            mask_height, mask_width = next(iter(parent_masks.values())).shape
            downsample_factor = img_width / mask_width
            logger.info(f"Calculated downsample factor: {downsample_factor:.2f}x")

            # Load GeoJSON data
            with open(geojson_path) as f:
                annotations = json.load(f)["features"]
            logger.info(f"Found {len(annotations)} annotations in {geojson_path.name}")

            patch_size = config["dataset_creation"]["patch_size"]
            class_map = config["dataset_creation"]["classes"]
            yolo_base_path = Path(config["paths"]["yolo_dataset"])
            patches_created = 0

            for ann in tqdm(annotations, desc=f"Processing {wsi_path.stem}"):
                try:
                    ann_class_name = (
                        ann["properties"].get("classification", {}).get("name")
                    )
                    if ann_class_name not in class_map:
                        if not hasattr(process_single_wsi, "warned_classes"):
                            process_single_wsi.warned_classes = set()
                        if ann_class_name not in process_single_wsi.warned_classes:
                            logger.warning(f"Skipping annotation with unmapped class: '{ann_class_name}'. "
                                           f"Ensure this name exists in your config.yaml `classes` map.")
                            process_single_wsi.warned_classes.add(ann_class_name)
                        continue
                    class_id = class_map[ann_class_name]

                    oyster_id = get_oyster_id_for_annotation(
                        ann, parent_masks, downsample_factor, logger
                    )
                    if oyster_id is None:
                        logger.debug(f"Annotation '{ann_class_name}' did not fall within any oyster mask. Skipping.")
                        continue

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

                    subset = (
                        "train"
                        if random.random()
                        < config["dataset_creation"]["train_val_split"]
                        else "val"
                    )

                    # --- NEW: Create a more descriptive filename ---
                    patch_filename = f"{wsi_path.stem}_oyster_{oyster_id}_patch_x{p_x_min}_y{p_y_min}"

                    cv2.imwrite(
                        str(yolo_base_path / f"images/{subset}/{patch_filename}.png"),
                        image_patch,
                    )

                    l_x_min, l_y_min = g_x_min - p_x_min, g_y_min - p_y_min
                    l_x_max, l_y_max = g_x_max - p_x_min, g_y_max - p_y_min
                    yolo_center_x = ((l_x_min + l_x_max) / 2) / patch_size
                    yolo_center_y = ((l_y_min + l_y_max) / 2) / patch_size
                    yolo_width = (l_x_max - l_x_min) / patch_size
                    yolo_height = (l_y_max - l_y_min) / patch_size

                    label_path = (
                        yolo_base_path / f"labels/{subset}/{patch_filename}.txt"
                    )
                    with open(label_path, "w") as f_label:
                        f_label.write(
                            f"{class_id} {yolo_center_x} {yolo_center_y} {yolo_width} {yolo_height}\n"
                        )
                except Exception as e:
                    logger.error(
                        f"Failed to process a single annotation. Error: {e}",
                        exc_info=True,
                    )
                    continue
    except Exception as e:
        logger.error(
            f"An unexpected error occurred while processing {wsi_path.name}: {e}",
            exc_info=True,
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
    setup_directories(yolo_dataset_path, logger)

    qupath_exports_dir = Path(config["paths"]["qupath_exports"])
    geojson_files = list(qupath_exports_dir.glob("*.geojson"))
    if not geojson_files:
        logger.error(f"No GeoJSON files found in '{qupath_exports_dir}'. Exiting.")
        return

    logger.info(f"Found {len(geojson_files)} GeoJSON files to process.")

    for geojson_path in geojson_files:
        wsi_name_stem = geojson_path.stem

        # Find matching WSI file
        possible_wsi_paths = list(
            Path(config["paths"]["raw_wsis"]).glob(f"{wsi_name_stem}.*")
        )
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
                f"Could not find a matching WSI for '{geojson_path.name}'. Skipping."
            )
            continue

        # Find matching oyster masks for this slide
        mask_dir = Path(config["paths"]["oyster_masks"]) / wsi_name_stem
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
        process_single_wsi(wsi_path, geojson_path, parent_mask_paths, config, logger)

    logger.info("--- Dataset Creation Script Finished ---")


if __name__ == "__main__":
    main()
