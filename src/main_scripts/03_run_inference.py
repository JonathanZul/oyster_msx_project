import os
import argparse
import json
import time
import torch
import tifffile
import zarr
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

from src.utils.file_handling import load_config
from src.utils.logging_config import setup_logging


def is_slide_completed(slide_output_dir: Path) -> bool:
    """
    Checks if a slide has already been fully processed by looking for a completion marker.

    Args:
        slide_output_dir (Path): The output directory for this slide.

    Returns:
        bool: True if the slide has been completed, False otherwise.
    """
    completion_marker = slide_output_dir / ".completed"
    return completion_marker.exists()


def mark_slide_completed(slide_output_dir: Path, logger):
    """
    Creates a completion marker file to indicate the slide has been fully processed.

    Args:
        slide_output_dir (Path): The output directory for this slide.
        logger: The logger instance.
    """
    completion_marker = slide_output_dir / ".completed"
    completion_marker.touch()
    logger.info(f"Marked slide as completed: {completion_marker}")


def mark_slide_incomplete(slide_output_dir: Path, failed_patches: int, logger):
    """
    Writes a marker that indicates the slide finished with patch-level failures.
    """
    failed_marker = slide_output_dir / ".failed"
    failed_marker.write_text(f"failed_patches={failed_patches}\n", encoding="utf-8")
    logger.warning(f"Marked slide as incomplete due to {failed_patches} failed patch reads: {failed_marker}")


def read_patch_with_retry(zarr_slicer, x: int, y: int, patch_size: int, max_retries: int, retry_sleep_s: float):
    """
    Reads a patch from a zarr-backed WSI with bounded retry attempts.
    """
    for attempt in range(max_retries + 1):
        try:
            return zarr_slicer[y:y + patch_size, x:x + patch_size]
        except OSError:
            if attempt >= max_retries:
                raise
            time.sleep(retry_sleep_s * (2 ** attempt))


def flush_detections_jsonl(detections_path: Path, detections_batch: list[dict]):
    """
    Appends batched detection rows to a JSONL file.
    """
    if not detections_batch:
        return

    with open(detections_path, "a", encoding="utf-8") as out_f:
        for row in detections_batch:
            out_f.write(json.dumps(row, separators=(",", ":")) + "\n")


def run_batched_prediction(
        model,
        patch_images: list[np.ndarray],
        patch_coords: list[tuple[int, int]],
        conf_threshold: float,
        patch_size: int
):
    """
    Runs YOLO prediction on a batch of patches and returns global-coordinate detections.
    """
    if not patch_images:
        return []

    batch_detections = []
    results = model.predict(source=patch_images, conf=conf_threshold, verbose=False)

    for result, (patch_x, patch_y) in zip(results, patch_coords):
        if len(result.boxes) == 0:
            continue

        for box in result.boxes:
            class_id = int(box.cls)
            score = float(box.conf)
            x_center, y_center, width, height = box.xywhn[0].tolist()

            x1 = (x_center - width / 2) * patch_size
            y1 = (y_center - height / 2) * patch_size
            x2 = (x_center + width / 2) * patch_size
            y2 = (y_center + height / 2) * patch_size

            batch_detections.append(
                {
                    "class_id": class_id,
                    "score": score,
                    "x1": x1 + patch_x,
                    "y1": y1 + patch_y,
                    "x2": x2 + patch_x,
                    "y2": y2 + patch_y,
                }
            )

    return batch_detections


def process_single_wsi_inference(wsi_path: Path, model, config: dict, logger, batch_size_override: int | None = None):
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
    batch_size = int(batch_size_override if batch_size_override is not None else params.get("inference_batch_size", 16))
    max_read_retries = int(params.get("max_patch_read_retries", 3))
    retry_sleep_s = float(params.get("patch_read_retry_sleep_seconds", 0.15))
    if batch_size < 1:
        logger.warning("Invalid inference_batch_size; defaulting to 1.")
        batch_size = 1

    # Create a unique output directory for this slide's raw predictions
    slide_output_dir = Path(config['paths']['inference_results']) / wsi_path.stem.replace('.ome', '')
    slide_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Raw prediction files will be saved to: {slide_output_dir}")
    detections_path = slide_output_dir / "detections.jsonl"

    # Check if this slide has already been completed
    if is_slide_completed(slide_output_dir):
        logger.info(f"Slide {wsi_path.name} already completed. Skipping.")
        return

    if detections_path.exists():
        detections_path.unlink()
    failed_marker = slide_output_dir / ".failed"
    if failed_marker.exists():
        failed_marker.unlink()

    try:
        with tifffile.TiffFile(wsi_path) as tif:
            level0 = tif.series[0].levels[0]
            img_height, img_width = level0.shape[:2]
            zarr_store = tif.series[0].aszarr()
            zarr_group = zarr.open(zarr_store, mode='r')
            zarr_slicer = zarr_group["0"]
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
            logger.info(f"Using inference batch size: {batch_size}")

            # Iterate through the grid and run prediction on each patch
            failed_patch_count = 0
            processed_patch_count = 0
            patch_images_batch = []
            patch_coords_batch = []
            detection_rows_buffer = []
            t0 = time.time()

            for y in tqdm(y_coords, desc=f"Processing rows for {wsi_path.stem}"):
                for x in x_coords:
                    try:
                        patch_image = read_patch_with_retry(
                            zarr_slicer=zarr_slicer,
                            x=x,
                            y=y,
                            patch_size=patch_size,
                            max_retries=max_read_retries,
                            retry_sleep_s=retry_sleep_s
                        )
                        patch_images_batch.append(patch_image)
                        patch_coords_batch.append((x, y))
                        processed_patch_count += 1

                        if len(patch_images_batch) >= batch_size:
                            detections = run_batched_prediction(
                                model=model,
                                patch_images=patch_images_batch,
                                patch_coords=patch_coords_batch,
                                conf_threshold=params['conf_threshold'],
                                patch_size=patch_size
                            )
                            detection_rows_buffer.extend(detections)
                            flush_detections_jsonl(detections_path, detection_rows_buffer)
                            detection_rows_buffer.clear()
                            patch_images_batch.clear()
                            patch_coords_batch.clear()

                    except Exception as e:
                        failed_patch_count += 1
                        logger.error(f"Failed to process patch at (x={x}, y={y}). Error: {e}", exc_info=True)
                        continue

            if patch_images_batch:
                detections = run_batched_prediction(
                    model=model,
                    patch_images=patch_images_batch,
                    patch_coords=patch_coords_batch,
                    conf_threshold=params['conf_threshold'],
                    patch_size=patch_size
                )
                detection_rows_buffer.extend(detections)
                flush_detections_jsonl(detections_path, detection_rows_buffer)

            elapsed_s = max(time.time() - t0, 1e-6)
            logger.info(
                f"Processed patches: {processed_patch_count}/{total_patches} "
                f"({processed_patch_count / elapsed_s:.2f} patches/s)"
            )
            logger.info(f"Failed patch reads: {failed_patch_count}")

    except Exception as e:
        logger.error(f"Failed to process WSI {wsi_path.name}. Error: {e}", exc_info=True)
        return

    # Mark slide as complete only if all patches were read and processed successfully.
    if failed_patch_count == 0:
        mark_slide_completed(slide_output_dir, logger)
    else:
        mark_slide_incomplete(slide_output_dir, failed_patch_count, logger)
    logger.info(f"--- Finished inference on: {wsi_path.name} ---")


def main():
    """
    Main function to execute the inference process.
    """
    parser = argparse.ArgumentParser(description="Stage 03: Run Inference on WSIs")
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="config.yaml",
        help="Path to the master configuration file."
    )
    parser.add_argument(
        "--slide-index",
        type=int,
        default=None,
        help="Process only the slide at this index (0-based). For batch job arrays."
    )
    parser.add_argument(
        "--slides-per-job",
        type=int,
        default=1,
        help="Number of slides to process per job (used with --slide-index)."
    )
    parser.add_argument(
        "--list-slides",
        action="store_true",
        help="List all unannotated slides and exit. Useful for planning batch jobs."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override inference batch size for patch-level prediction."
    )
    args = parser.parse_args()

    config = load_config(args.config)
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

    # Sort slides for consistent ordering across batch jobs
    unannotated_wsis = sorted(unannotated_wsis, key=lambda p: p.name)
    logger.info(f"Found {len(unannotated_wsis)} unannotated WSIs to process.")

    # Handle --list-slides flag
    if args.list_slides:
        logger.info("Slide listing mode. Printing slides and exiting.")
        for i, wsi_path in enumerate(unannotated_wsis):
            completed = is_slide_completed(
                Path(config['paths']['inference_results']) / wsi_path.stem.replace('.ome', '')
            )
            status = "[DONE]" if completed else "[PENDING]"
            print(f"{i}: {wsi_path.name} {status}")
        return

    # 3. Determine which slides to process based on arguments
    if args.slide_index is not None:
        # Batch mode: process only slides in the specified range
        start_idx = args.slide_index * args.slides_per_job
        end_idx = start_idx + args.slides_per_job
        slides_to_process = unannotated_wsis[start_idx:end_idx]
        logger.info(f"Batch mode: processing slides {start_idx} to {end_idx-1} ({len(slides_to_process)} slides)")
    else:
        # Normal mode: process all slides
        slides_to_process = unannotated_wsis

    # Run inference on each selected WSI
    for wsi_path in slides_to_process:
        process_single_wsi_inference(wsi_path, model, config, logger, batch_size_override=args.batch_size)

    logger.info("--- Inference Script Finished ---")


if __name__ == "__main__":
    main()
