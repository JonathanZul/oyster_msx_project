import os
import torch
import tifffile
import zarr
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

from src.utils.file_handling import load_config
from src.utils.logging_config import setup_logging


def process_single_wsi_inference(wsi_path: Path, model, config: dict, logger):
    """
    Runs inference on a single WSI by breaking it into patches.

    Args:
        wsi_path (Path): Path to the WSI file to process.
        model (YOLO): The trained YOLO model instance.
        config (dict): The project configuration dictionary.
        logger: The logger instance for logging messages.
    """
    logger.info(f"--- Starting inference on: {wsi_path.name} ---")

    params = config['inference']
    patch_size = config['dataset_creation']['patch_size']  # Use same patch size as training
    patch_overlap = params['patch_overlap']

    # Create a unique output directory for this slide's raw predictions
    slide_output_dir = Path(config['paths']['inference_results']) / wsi_path.stem.replace('.ome', '')
    slide_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Raw prediction files will be saved to: {slide_output_dir}")

    try:
        with tifffile.TiffFile(wsi_path) as tif:
            level0 = tif.series[0].levels[0]
            img_height, img_width = level0.shape[:2]
            zarr_slicer = zarr.open(tif.series[0].aszarr(), mode='r')[0]
            logger.info(f"WSI opened. Dimensions: {img_width}x{img_height}")

            # Define the grid of overlapping patches
            step_size = int(patch_size * (1 - patch_overlap))
            if step_size <= 0:
                logger.error("Patch overlap must be less than 1.0. Setting step size to patch size.")
                step_size = patch_size

            x_coords = range(0, img_width - patch_size + 1, step_size)
            y_coords = range(0, img_height - patch_size + 1, step_size)

            total_patches = len(x_coords) * len(y_coords)
            logger.info(f"Generating a grid of {len(x_coords)}x{len(y_coords)} = {total_patches} patches...")

            # Iterate through the grid and run prediction on each patch
            for y in tqdm(y_coords, desc=f"Processing rows for {wsi_path.stem}"):
                for x in x_coords:
                    try:
                        # Read the patch from the WSI
                        patch_image = zarr_slicer[y:y + patch_size, x:x + patch_size]

                        # The model expects a list of images, so we put our patch in a list
                        results = model.predict(source=[patch_image], conf=params['conf_threshold'], verbose=False)

                        # The result object contains the predictions for the first image
                        result = results[0]

                        # If the model found any objects in this patch...
                        if len(result.boxes) > 0:
                            # Define a unique filename for this patch's prediction file
                            pred_filename = slide_output_dir / f"patch_x{x}_y{y}.txt"

                            with open(pred_filename, 'w') as f:
                                for box in result.boxes:
                                    # Extract class, confidence, and normalized coordinates
                                    class_id = int(box.cls)
                                    confidence = float(box.conf)
                                    # Coordinates are already normalized (0-1) relative to the patch
                                    x_center, y_center, width, height = box.xywhn[0].tolist()

                                    # Write to file in YOLO format
                                    f.write(f"{class_id} {confidence} {x_center} {y_center} {width} {height}\n")

                    except Exception as e:
                        logger.error(f"Failed to process patch at (x={x}, y={y}). Error: {e}", exc_info=True)
                        continue

    except Exception as e:
        logger.error(f"Failed to process WSI {wsi_path.name}. Error: {e}", exc_info=True)

    logger.info(f"--- Finished inference on: {wsi_path.name} ---")


def main():
    """
    Main function to orchestrate the inference process.
    """
    config = load_config()
    if not config:
        return

    logger = setup_logging(Path(config['paths']['logs']), '03_run_inference')
    logger.info("--- Starting Inference Script ---")

    # 1. Load the trained YOLO model
    model_path = Path(config['inference']['model_checkpoint'])
    if not model_path.exists():
        # Let's try to find the 'best.pt' in the latest run directory
        model_dir = Path(config['paths']['model_output_dir'])
        run_dirs = sorted([d for d in model_dir.iterdir() if d.is_dir()])
        if run_dirs:
            latest_run = run_dirs[-1]
            potential_model_path = latest_run / 'weights' / 'best.pt'
            if potential_model_path.exists():
                model_path = potential_model_path
                logger.warning(f"Model checkpoint in config not found. Using latest found model: {model_path}")
            else:
                logger.critical(f"No trained model found at '{model_path}' or in the latest run directory. Aborting.")
                return
        else:
            logger.critical(f"No trained model found at '{model_path}' and no training runs found. Aborting.")
            return

    try:
        logger.info(f"Loading trained model from: {model_path}")
        model = YOLO(model_path)
    except Exception as e:
        logger.critical(f"Failed to load YOLO model. Error: {e}", exc_info=True)
        return

    # 2. Find all WSI files that have NOT been annotated yet
    raw_wsis_dir = Path(config['paths']['raw_wsis'])
    qupath_exports_dir = Path(config['paths']['qupath_exports'])

    annotated_stems = {f.stem.replace('.ome', '') for f in qupath_exports_dir.glob("*.geojson")}
    all_wsi_paths = list(raw_wsis_dir.glob("*.tif")) + list(raw_wsis_dir.glob("*.vsi"))
    all_usable_wsi_paths = [p for p in all_wsi_paths if not p.name.startswith('.')]

    unannotated_wsis = [
        p for p in all_usable_wsi_paths
        if p.stem.replace('.ome', '') not in annotated_stems
    ]

    if not unannotated_wsis:
        logger.info("No unannotated WSIs found to process. All slides appear to have a corresponding GeoJSON file.")
        return

    logger.info(f"Found {len(unannotated_wsis)} unannotated WSIs to process.")

    # 3. Run inference on each unannotated WSI
    for wsi_path in unannotated_wsis:
        process_single_wsi_inference(wsi_path, model, config, logger)

    logger.info("--- Inference Script Finished ---")


if __name__ == "__main__":
    main()
