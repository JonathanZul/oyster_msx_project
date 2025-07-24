import os
import glob
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import tifffile
from shapely.geometry import shape
from tqdm import tqdm

# Import utility functions from our project
from utils.file_handling import load_config
from utils.logging_config import setup_logging


def setup_directories(base_path: Path, logger):
    """
    Cleans and creates the necessary directory structure for the YOLO dataset.

    Args:
        base_path (Path): The root path of the YOLO dataset.
        logger: The logger instance for logging messages.
    """
    if base_path.exists():
        logger.info(f"Removing existing dataset directory: {base_path}")
        shutil.rmtree(base_path)

    logger.info(f"Creating new dataset directory structure at: {base_path}")
    (base_path / "images/train").mkdir(parents=True, exist_ok=True)
    (base_path / "images/val").mkdir(parents=True, exist_ok=True)
    (base_path / "labels/train").mkdir(parents=True, exist_ok=True)
    (base_path / "labels/val").mkdir(parents=True, exist_ok=True)


def process_single_wsi(wsi_path: Path, geojson_path: Path, config: dict, logger):
    """
    Processes a single WSI and its corresponding GeoJSON annotations
    to generate patches and labels.

    Args:
        wsi_path (Path): Path to the Whole Slide Image.
        geojson_path (Path): Path to the GeoJSON annotation file.
        config (dict): The project configuration dictionary.
        logger: The logger instance for logging messages.
    """
    try:
        # Load the WSI using tifffile. Use aszarr() for efficient patch reading.
        logger.info(f"Opening WSI: {wsi_path.name}")
        with tifffile.TiffFile(wsi_path) as tif:
            wsi = tif.series[0].levels[0].aszarr()
            img_height, img_width = wsi.shape[:2]
            logger.debug(f"WSI dimensions (H, W): ({img_height}, {img_width})")

            # Load GeoJSON data
            with open(geojson_path) as f:
                annotations = json.load(f)["features"]
            logger.info(f"Found {len(annotations)} annotations in {geojson_path.name}")

            patch_size = config["dataset_creation"]["patch_size"]
            class_map = config["dataset_creation"]["classes"]
            yolo_base_path = Path(config["paths"]["yolo_dataset"])

            # Use tqdm for a progress bar over the annotations
            for ann in tqdm(annotations, desc=f"Processing {wsi_path.stem}"):
                try:
                    # Get the class name from the properties
                    ann_class_name = (
                        ann["properties"].get("classification", {}).get("name")
                    )

                    # Skip if the annotation class is not in our config
                    if ann_class_name not in class_map:
                        logger.debug(
                            f"Skipping annotation with unmapped class: {ann_class_name}"
                        )
                        continue

                    class_id = class_map[ann_class_name]

                    # Use shapely to handle the geometry and get the bounding box
                    geom = shape(ann["geometry"])
                    g_x_min, g_y_min, g_x_max, g_y_max = geom.bounds

                    # Calculate the center of the annotation's bounding box
                    center_x = (g_x_min + g_x_max) / 2
                    center_y = (g_y_min + g_y_max) / 2

                    # Define the patch boundaries, centered on the annotation
                    p_x_min = int(center_x - patch_size / 2)
                    p_y_min = int(center_y - patch_size / 2)
                    p_x_max = p_x_min + patch_size
                    p_y_max = p_y_min + patch_size

                    # Handle edge cases: ensure the patch is within the WSI bounds
                    if (
                        p_x_min < 0
                        or p_y_min < 0
                        or p_x_max > img_width
                        or p_y_max > img_height
                    ):
                        logger.warning(
                            f"Skipping annotation at ({center_x}, {center_y}) because its patch would be out of bounds."
                        )
                        continue

                    # Read the image patch from the WSI
                    image_patch = wsi[p_y_min:p_y_max, p_x_min:p_x_max]

                    # Convert to 3-channel BGR for saving with OpenCV
                    if image_patch.ndim == 3 and image_patch.shape[2] == 4:
                        image_patch = cv2.cvtColor(image_patch, cv2.COLOR_RGBA2BGR)
                    elif image_patch.ndim == 2:
                        image_patch = cv2.cvtColor(image_patch, cv2.COLOR_GRAY2BGR)

                    # Decide if this patch goes into train or validation set
                    subset = (
                        "train"
                        if random.random()
                        < config["dataset_creation"]["train_val_split"]
                        else "val"
                    )

                    # Create a unique filename for the patch
                    patch_filename = f"{wsi_path.stem}_patch_x{p_x_min}_y{p_y_min}"

                    # Save the image patch
                    cv2.imwrite(
                        str(yolo_base_path / f"images/{subset}/{patch_filename}.png"),
                        image_patch,
                    )

                    # --- Create the YOLO label file ---
                    # Calculate annotation coordinates *relative to the patch*
                    l_x_min = g_x_min - p_x_min
                    l_y_min = g_y_min - p_y_min
                    l_x_max = g_x_max - p_x_min
                    l_y_max = g_y_max - p_y_min

                    # Convert to YOLO format (normalized center x, y, width, height)
                    yolo_center_x = ((l_x_min + l_x_max) / 2) / patch_size
                    yolo_center_y = ((l_y_min + l_y_max) / 2) / patch_size
                    yolo_width = (l_x_max - l_x_min) / patch_size
                    yolo_height = (l_y_max - l_y_min) / patch_size

                    # Write the label file
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

    except FileNotFoundError:
        logger.error(
            f"WSI or GeoJSON file not found. Searched for: {wsi_path} and {geojson_path}"
        )
    except Exception as e:
        logger.error(
            f"An unexpected error occurred while processing {wsi_path.name}: {e}",
            exc_info=True,
        )


def main():
    """
    Main function to orchestrate the dataset creation process.
    """
    # Load configuration
    config = load_config()
    if not config:
        return

    # Setup logging
    log_dir = Path(config["paths"]["logs"])
    logger = setup_logging(log_dir, "01_create_dataset")
    logger.info("--- Starting Dataset Creation Script ---")

    # Clean and create output directories
    yolo_dataset_path = Path(config["paths"]["yolo_dataset"])
    setup_directories(yolo_dataset_path, logger)

    # Find all GeoJSON files to process
    qupath_exports_dir = Path(config["paths"]["qupath_exports"])
    geojson_files = list(qupath_exports_dir.glob("*.geojson"))

    if not geojson_files:
        logger.error(f"No GeoJSON files found in '{qupath_exports_dir}'. Exiting.")
        return

    logger.info(f"Found {len(geojson_files)} GeoJSON files to process.")

    # Process each WSI/GeoJSON pair
    for geojson_path in geojson_files:
        # Find the matching WSI file
        wsi_name_stem = geojson_path.stem

        # Search for .tif, .tiff, .vsi, etc.
        possible_wsi_paths = list(
            Path(config["paths"]["raw_wsis"]).glob(f"{wsi_name_stem}.*")
        )

        # Filter out non-image files if necessary, e.g. '.vsi.json'
        wsi_path = None
        for p in possible_wsi_paths:
            if p.suffix.lower() in [".tif", ".tiff", ".vsi"]:
                wsi_path = p
                break

        if not wsi_path or not wsi_path.exists():
            logger.warning(
                f"Could not find a matching WSI for '{geojson_path.name}'. Skipping."
            )
            continue

        process_single_wsi(wsi_path, geojson_path, config, logger)

    logger.info("--- Dataset Creation Script Finished ---")


if __name__ == "__main__":
    main()
