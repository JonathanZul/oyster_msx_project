# src/main_scripts/04_format_predictions.py

import argparse
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
    detections_jsonl = slide_pred_dir / "detections.jsonl"

    if detections_jsonl.exists():
        logger.info(f"Using compact detection file: {detections_jsonl.name}")
        with open(detections_jsonl, "r", encoding="utf-8") as in_f:
            for line in in_f:
                if not line.strip():
                    continue
                row = json.loads(line)
                all_boxes.append([row["x1"], row["y1"], row["x2"], row["y2"]])
                all_scores.append(row["score"])
                all_class_ids.append(int(row["class_id"]))
    else:
        txt_files = list(slide_pred_dir.glob("*.txt"))
        if not txt_files:
            logger.warning("No prediction files found in this directory. Skipping.")
            return

        logger.info(f"Using legacy patch text files: {len(txt_files)} files.")
        patch_size = config['dataset_creation']['patch_size']

        for txt_file in tqdm(txt_files, desc="Reading raw predictions"):
            if txt_file.name.startswith('.'):
                logger.debug(f"Skipping hidden file: {txt_file.name}")
                continue

            parts = txt_file.stem.split('_')
            patch_x = int(parts[1][1:])
            patch_y = int(parts[2][1:])

            with open(txt_file, 'r', encoding="utf-8") as f:
                for line in f:
                    class_id, conf, x_c, y_c, w, h = map(float, line.strip().split())
                    abs_w = w * patch_size
                    abs_h = h * patch_size
                    abs_xc = x_c * patch_size
                    abs_yc = y_c * patch_size
                    x1 = abs_xc - abs_w / 2
                    y1 = abs_yc - abs_h / 2
                    x2 = abs_xc + abs_w / 2
                    y2 = abs_yc + abs_h / 2
                    all_boxes.append([x1 + patch_x, y1 + patch_y, x2 + patch_x, y2 + patch_y])
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

    # Run class-wise NMS to avoid suppressing detections across classes.
    class_ids_array = np.array(all_class_ids)
    kept_indices = []
    for class_id in np.unique(class_ids_array):
        class_mask = np.where(class_ids_array == class_id)[0]
        class_boxes = boxes_tensor[class_mask]
        class_scores = scores_tensor[class_mask]
        class_keep = torchvision.ops.nms(class_boxes, class_scores, iou_threshold=0.4)
        kept_indices.extend(class_mask[class_keep.cpu().numpy()].tolist())

    # Keep deterministic ordering by descending score.
    kept_indices = sorted(kept_indices, key=lambda i: all_scores[i], reverse=True)

    clean_boxes = boxes_tensor[kept_indices].tolist()
    clean_scores = scores_tensor[kept_indices].tolist()
    clean_class_ids = class_ids_array[kept_indices].tolist()

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


def is_slide_completed(slide_dir: Path) -> bool:
    """Check if a slide's inference was fully completed."""
    return (slide_dir / ".completed").exists()


def is_slide_already_formatted(slide_dir: Path, output_dir: Path) -> bool:
    """Check if a slide already has a formatted GeoJSON output."""
    output_path = output_dir / f"{slide_dir.name}_predictions.geojson"
    return output_path.exists()


def main():
    """Main function to orchestrate the prediction formatting process."""
    parser = argparse.ArgumentParser(description="Stage 04: Format YOLO Predictions for QuPath")
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="config.yaml",
        help="Path to the master configuration file."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process slides even if GeoJSON output already exists."
    )
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Process slides even if inference was not fully completed (no .completed marker)."
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if not config:
        return

    logger = setup_logging(Path(config['paths']['logs']), '04_format_predictions')
    logger.info("--- Starting Prediction Formatting Script ---")

    inference_dir = Path(config['paths']['inference_results'])
    output_dir = Path(config['paths']['qupath_imports'])

    if not inference_dir.exists() or not any(inference_dir.iterdir()):
        logger.warning(f"Inference results directory is empty or does not exist: '{inference_dir}'")
        return

    # Find all slide prediction subdirectories (exclude hidden dirs)
    slide_dirs = [d for d in inference_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]

    logger.info(f"Found {len(slide_dirs)} slide prediction directories.")

    # Filter to only completed slides unless --include-incomplete is set
    if not args.include_incomplete:
        completed_dirs = [d for d in slide_dirs if is_slide_completed(d)]
        incomplete_count = len(slide_dirs) - len(completed_dirs)
        if incomplete_count > 0:
            logger.info(f"Skipping {incomplete_count} incomplete slides (no .completed marker).")
        slide_dirs = completed_dirs

    # Skip already formatted slides unless --force is set
    if not args.force:
        to_process = [d for d in slide_dirs if not is_slide_already_formatted(d, output_dir)]
        skipped_count = len(slide_dirs) - len(to_process)
        if skipped_count > 0:
            logger.info(f"Skipping {skipped_count} slides (GeoJSON already exists). Use --force to re-process.")
        slide_dirs = to_process

    if not slide_dirs:
        logger.info("No slides to process.")
        return

    logger.info(f"Processing {len(slide_dirs)} slides.")

    for slide_dir in slide_dirs:
        process_single_slide_predictions(slide_dir, config, logger)

    logger.info("--- Prediction Formatting Script Finished ---")


if __name__ == "__main__":
    main()
