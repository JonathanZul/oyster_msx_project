import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.utils.file_handling import load_config
from src.utils.logging_config import setup_logging, log_config
from src.utils.wsi_utils import (
    extract_downsampled_overview,
    get_wsi_level0_dimensions,
    find_matching_wsi_path
)
from src.utils.evaluation_utils import find_polygon_coords


def rasterize_annotations_to_mask(blank_mask: np.ndarray, geojson_path: Path, scaling_factors: tuple, class_map: dict,
                                  logger):
    """
    Loads annotations from a GeoJSON file and draws ("rasterizes") them onto a 
    blank multi-channel mask array.

    Args:
        blank_mask (np.ndarray): A pre-initialized NumPy array to draw masks on.
        geojson_path (Path): The path to the QuPath GeoJSON annotation file.
        scaling_factors (tuple): A tuple (scale_x, scale_y) for resizing coordinates.
        class_map (dict): A dictionary mapping class names to channel indices.
        logger: The logger instance.

    Returns:
        np.ndarray | None: The mask with annotations drawn on it, or None on failure.
    """
    scale_x, scale_y = scaling_factors
    try:
        with open(geojson_path, "r") as f:
            annotations = json.load(f)["features"]

        # Loop through each annotation (polygon) in the GeoJSON file
        for ann in annotations:
            class_name = ann.get("properties", {}).get("classification", {}).get("name")
            if class_name not in class_map:
                continue  # Skip annotations with classes not defined in our config

            # Config is 1-based for human readability, so subtract 1 for 0-based array indexing
            channel_idx = class_map[class_name] - 1

            # Safely parse potentially messy coordinate structures from QuPath
            raw_coords = find_polygon_coords(ann["geometry"]["coordinates"])
            if raw_coords is None:
                logger.warning(f"Could not parse coordinates for an annotation in {geojson_path.name}. Skipping it.")
                continue

            # Scale coordinates from high-res WSI space to low-res overview space
            coords = np.array(raw_coords, dtype=np.float32)
            coords_scaled = coords.copy()
            coords_scaled[:, 0] *= scale_x
            coords_scaled[:, 1] *= scale_y

            # To avoid OpenCV errors with non-contiguous memory, we make a safe copy of the
            # specific mask channel we want to draw on.
            mask_channel_slice = blank_mask[:, :, channel_idx].copy()
            cv2.fillPoly(mask_channel_slice, [coords_scaled.astype(np.int32)], 255)

            # Assign the modified copy back to the main mask array.
            blank_mask[:, :, channel_idx] = mask_channel_slice

        return blank_mask

    except Exception as e:
        logger.error(f"Failed to process GeoJSON {geojson_path.name}: {e}", exc_info=True)
        return None


def process_and_save_sample(wsi_path: Path, geojson_path: Path, config: dict, logger):
    """
    Processes a single WSI and its corresponding GeoJSON annotations to create
    and save the image, label mask, and ROI mask for segmentation training.

    Args:
        wsi_path (Path): Path to the Whole Slide Image file.
        geojson_path (Path): Path to the corresponding GeoJSON annotation file.
        config (dict): The project configuration dictionary.
        logger: The logger instance.

    Returns:
        None
    """
    logger.info(f"--- Processing: {wsi_path.name} ---")
    seg_config = config["ml_segmentation"]
    target_size_wh = tuple(seg_config["image_size"][::-1])
    dataset_dir = Path(config["paths"]["segmentation_dataset"])

    # Step 1: Get high-resolution WSI dimensions for accurate coordinate scaling.
    wsi_dims_wh = get_wsi_level0_dimensions(wsi_path, logger)
    if not wsi_dims_wh: return

    # Step 2: Extract a low-resolution overview image to work with.
    overview_img = extract_downsampled_overview(wsi_path, seg_config["overview_downsample"], logger)
    if overview_img is None: return

    # Step 3: Standardize the image size for consistent model input.
    resized_img = cv2.resize(overview_img, target_size_wh, interpolation=cv2.INTER_AREA)

    # Step 4: Prepare a blank canvas for the masks and calculate scaling factors.
    num_classes = len(seg_config["classes"])
    mask_shape = (*seg_config["image_size"], num_classes)
    blank_mask = np.zeros(mask_shape, dtype=np.uint8)
    scaling_factors = (target_size_wh[0] / wsi_dims_wh[0], target_size_wh[1] / wsi_dims_wh[1])

    # Step 5: Create the label mask by rasterizing the annotations.
    label_mask = rasterize_annotations_to_mask(blank_mask, geojson_path, scaling_factors, seg_config["classes"], logger)
    if label_mask is None:
        logger.error(f"Mask creation failed for {wsi_path.name}. Skipping sample.")
        return

    # Step 6: Create the final masks needed for training.
    # The ROI mask combines all classes to define the area for loss calculation.
    roi_mask = np.logical_or(label_mask[:, :, 0], label_mask[:, :, 1]).astype(np.uint8)
    # The label mask is normalized to 0s and 1s for the loss function.
    label_mask_normalized = (label_mask > 0).astype(np.uint8)

    # Step 7: Save all three components to the dataset directory.
    try:
        slide_stem = wsi_path.stem.replace(".ome", "")
        cv2.imwrite(str(dataset_dir / "images" / f"{slide_stem}.png"), cv2.cvtColor(resized_img, cv2.COLOR_RGB2BGR))
        np.save(dataset_dir / "masks" / f"{slide_stem}.npy", label_mask_normalized)
        np.save(dataset_dir / "rois" / f"{slide_stem}.npy", roi_mask)
        logger.info(f"Successfully created image, label mask, and ROI mask for {slide_stem}")
    except Exception as e:
        logger.error(f"Failed to save files for {slide_stem}: {e}", exc_info=True)


def setup_directories(dataset_dir: Path, logger):
    """
    Cleans and creates the necessary directory structure for the segmentation dataset.

    Args:
        dataset_dir (Path): The root directory for the segmentation dataset.
        logger: The logger instance.

    Returns:
        None
    """
    if dataset_dir.exists():
        logger.warning(f"Removing existing segmentation dataset at: {dataset_dir}")
        try:
            shutil.rmtree(dataset_dir)
        except OSError as e:
            logger.error(f"Error removing directory {dataset_dir}: {e}. Attempting to continue.")

    logger.info("Creating fresh dataset directories...")
    (dataset_dir / "images").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "masks").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "rois").mkdir(parents=True, exist_ok=True)


def main():
    """
    Main function for the segmentation dataset creation process.

    Parses command-line arguments, loads configuration, sets up logging,
    processes each WSI and its annotations, and saves the resulting dataset.
    """
    # --- Parse command-line arguments ---
    parser = argparse.ArgumentParser(description="Stage 00a: Create Segmentation Dataset")
    parser.add_argument("-c", "--config", type=str, default="config.yaml", help="Path to the config file.")
    args = parser.parse_args()

    # --- Load Config ---
    config = load_config(args.config)
    if not config: return

    # --- Setup Logging ---
    logger = setup_logging(Path(config["paths"]["logs"]), "00a_create_seg_dataset")
    log_config(config, logger)
    logger.info("--- Starting Segmentation Dataset Creation (Script 00a) ---")

    # --- Dataset Preparation ---
    # Prepare the output directory structure.
    setup_directories(Path(config["paths"]["segmentation_dataset"]), logger)

    # Define input directories.
    annot_dir = Path(config["paths"]["segmentation_annotations"])
    wsi_dir = Path(config["paths"]["raw_wsis"])

    # Find all annotation files, filtering out hidden system files.
    all_files = list(annot_dir.glob("*.geojson"))
    geojson_files = [f for f in all_files if not f.name.startswith("._")]

    if not geojson_files:
        logger.error(f"No valid GeoJSON annotation files found in '{annot_dir}'.")
        return

    # --- Process Each Annotation and Corresponding WSI ---
    logger.info(f"Found {len(geojson_files)} valid annotation files to process.")
    for geojson_path in tqdm(geojson_files, desc="Creating Segmentation Dataset"):
        # Find the corresponding WSI file for each annotation.
        wsi_path = find_matching_wsi_path(geojson_path, wsi_dir)
        if not wsi_path:
            logger.warning(f"No matching WSI found for annotation '{geojson_path.name}'. Skipping.")
            continue

        # Process the matched pair.
        process_and_save_sample(wsi_path, geojson_path, config, logger)

    logger.info("--- Segmentation Dataset Creation Finished ---")


if __name__ == "__main__":
    main()
