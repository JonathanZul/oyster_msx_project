# src/main_scripts/04_format_predictions.py

import os
import glob
import json
import torch
import torchvision
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Import utility functions from our project
from src.utils.file_handling import load_config
from src.utils.logging_config import setup_logging


def process_single_slide_predictions(slide_pred_dir: Path, config: dict, logger):
    """
    Processes all raw prediction files for a single slide, performs NMS,
    and saves the result as a single GeoJSON file.

    Args:
        slide_pred_dir (Path): The directory containing raw .txt predictions for a slide.
        config (dict): The project configuration dictionary.
        logger: The logger instance.
    """
    logger.info(f"--- Processing predictions for slide: {slide_pred_dir.name} ---")

    # 1. Load all raw predictions into a single list
    all_boxes = []
    all_scores = []
    all_class_ids = []

    txt_files = list(slide_pred_dir.glob("*.txt"))
    if not txt_files:
        logger.warning("No prediction files found in this directory. Skipping.")
        return

    logger.info(f"Found {len(txt_files)} prediction files to process.")

    patch_size = config['dataset_creation']['patch_size']

    for txt_file in tqdm(txt_files, desc="Reading raw predictions"):
        if txt_file.name.startswith('.'):
            logger.debug(f"Skipping hidden file: {txt_file.name}")
            continue

        # Extract patch's top-left coordinates from the filename
        parts = txt_file.stem.split('_')
        patch_x = int(parts[1][1:])  # e.g., 'x1024' -> 1024
        patch_y = int(parts[2][1:])  # e.g., 'y2048' -> 2048

        with open(txt_file, 'r') as f:
            for line in f:
                class_id, conf, x_c, y_c, w, h = map(float, line.strip().split())

                # Convert normalized, patch-relative coords to absolute global WSI coords
                # a. Un-normalize to patch pixel coordinates
                abs_w = w * patch_size
                abs_h = h * patch_size
                abs_xc = x_c * patch_size
                abs_yc = y_c * patch_size

                # b. Convert from (center, w, h) to (x1, y1, x2, y2)
                x1 = abs_xc - abs_w / 2
                y1 = abs_yc - abs_h / 2
                x2 = abs_xc + abs_w / 2
                y2 = abs_yc + abs_h / 2

                # c. Add patch offset to get global WSI coordinates
                global_x1 = x1 + patch_x
                global_y1 = y1 + patch_y
                global_x2 = x2 + patch_x
                global_y2 = y2 + patch_y

                all_boxes.append([global_x1, global_y1, global_x2, global_y2])
                all_scores.append(conf)
                all_class_ids.append(int(class_id))

    if not all_boxes:
        logger.warning("No valid detections found across all patches. Skipping GeoJSON creation.")
        return

    logger.info(f"Loaded a total of {len(all_boxes)} raw detections.")

    # 2. Perform Non-Maximum Suppression (NMS) to merge overlapping boxes
    logger.info("Performing Non-Maximum Suppression (NMS) to merge overlapping detections...")
    boxes_tensor = torch.tensor(all_boxes, dtype=torch.float32)
    scores_tensor = torch.tensor(all_scores, dtype=torch.float32)

    # NMS is a crucial step to clean up results from overlapping inference patches
    # We use a low IoU threshold to aggressively merge boxes that likely represent the same object.
    nms_indices = torchvision.ops.nms(boxes_tensor, scores_tensor, iou_threshold=0.4)

    clean_boxes = boxes_tensor[nms_indices].tolist()
    clean_scores = scores_tensor[nms_indices].tolist()
    clean_class_ids = np.array(all_class_ids)[nms_indices].tolist()

    logger.info(f"NMS reduced the number of detections from {len(all_boxes)} to {len(clean_boxes)}.")

    # 1. Create a list of GeoJSON Features that match the QuPath format
    geojson_features = []
    class_map_rev = {v: k for k, v in config['dataset_creation']['classes'].items()}

    for box, score, class_id in zip(clean_boxes, clean_scores, clean_class_ids):
        x1, y1, x2, y2 = box
        class_name = class_map_rev.get(class_id, "Unknown")

        # GeoJSON requires a closed polygon: [p1, p2, p3, p4, p1]
        coordinates = [[
            [x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]
        ]]

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": coordinates
            },
            "properties": {
                # Key change: "objectType" instead of "object_type"
                "objectType": "annotation",
                "classification": {
                    "name": class_name,
                    # We can let QuPath assign the color, or define one here.
                    # This format with RGB values is also valid.
                    "color": [255, 0, 0]
                },
                # We will add our confidence score as a measurement QuPath can read.
                "measurements": [
                    {"name": "Confidence", "value": score}
                ]
            }
        }
        geojson_features.append(feature)

    # 2. Create the final FeatureCollection object
    qupath_geojson_output = {
        "type": "FeatureCollection",
        "features": geojson_features
    }

    # 3. Save the final GeoJSON file
    output_path = Path(config['paths']['qupath_imports']) / f"{slide_pred_dir.name}_predictions.geojson"
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(qupath_geojson_output, f, indent=2)

    logger.info(f"Successfully saved QuPath-compatible predictions to: {output_path}")


def main():
    """Main function to orchestrate the prediction formatting process."""
    config = load_config()
    if not config:
        return

    logger = setup_logging(Path(config['paths']['logs']), '04_format_predictions')
    logger.info("--- Starting Prediction Formatting Script ---")

    inference_dir = Path(config['paths']['inference_results'])

    if not inference_dir.exists() or not any(inference_dir.iterdir()):
        logger.warning(f"Inference results directory is empty or does not exist: '{inference_dir}'")
        return

    # Find all slide prediction subdirectories
    slide_dirs = [d for d in inference_dir.iterdir() if d.is_dir()]

    valid_slide_dirs = [d for d in slide_dirs if not d.name.startswith('.')]

    logger.info(f"Found {len(slide_dirs)} slide prediction directories to process.")

    for slide_dir in valid_slide_dirs:
        process_single_slide_predictions(slide_dir, config, logger)

    logger.info("--- Prediction Formatting Script Finished ---")


if __name__ == "__main__":
    main()
