import os
import argparse
import json
import time
import yaml
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import tifffile
import zarr
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.utils.file_handling import load_config
from src.utils.logging_config import setup_logging


def get_config_class_names(config: dict):
    classes = config['dataset_creation']['classes']
    return [name for name, idx in sorted(classes.items(), key=lambda item: item[1])]


def get_training_namespace(config: dict, class_names: list[str]) -> str:
    ns = str(config.get("training", {}).get("run_namespace", "")).strip()
    if ns:
        return ns
    slug = "_".join(c.lower().replace(" ", "_") for c in class_names)
    return f"classset_{slug}"


def get_inference_namespace(config: dict, class_names: list[str]) -> str:
    ns = str(config.get("inference", {}).get("model_run_namespace", "")).strip()
    if ns:
        return ns
    return get_training_namespace(config, class_names)


def read_candidate_class_names(best_model_path: Path):
    """
    Reads class names metadata from a training run directory when available.
    """
    run_dir = best_model_path.parent.parent
    manifest_path = run_dir / "class_names.json"
    if manifest_path.exists():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            names = payload.get("class_names")
            if isinstance(names, list) and all(isinstance(n, str) for n in names):
                return names
        except Exception:
            pass

    # Backward compatibility for older runs.
    args_yaml = run_dir / "args.yaml"
    if args_yaml.exists():
        try:
            args_data = yaml.safe_load(args_yaml.read_text(encoding="utf-8")) or {}
            data_path = args_data.get("data")
            if data_path:
                dataset_yaml_path = Path(str(data_path))
                if dataset_yaml_path.exists():
                    dataset_cfg = yaml.safe_load(dataset_yaml_path.read_text(encoding="utf-8")) or {}
                    names = dataset_cfg.get("names")
                    if isinstance(names, list) and all(isinstance(n, str) for n in names):
                        return names
        except Exception:
            pass
    return None


def find_latest_best_model(model_dir: Path, expected_class_names: list[str] | None = None):
    """
    Finds the most recently modified best.pt under the model output directory.
    If expected_class_names is provided, prefers runs whose class names match exactly.
    """
    if not model_dir.exists():
        return None
    candidates = sorted(model_dir.rglob("best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return None
    if not expected_class_names:
        return candidates[0]

    expected = [str(x).strip() for x in expected_class_names]
    for candidate in candidates:
        candidate_names = read_candidate_class_names(candidate)
        if candidate_names is not None and [str(x).strip() for x in candidate_names] == expected:
            return candidate
    return None


def collect_target_slides(config: dict, include_annotated: bool = False):
    """
    Collects raw WSIs and returns the subset targeted for inference.

    Returns:
        tuple[list[Path], set[str], list[Path], int]:
            - all usable raw WSI paths
            - annotated stems from GeoJSON files
            - target WSI paths for inference
            - count of raw WSIs that already have annotation matches
    """
    raw_wsis_dir = Path(config['paths']['raw_wsis'])
    qupath_exports_dir = Path(config['paths']['qupath_exports'])

    annotated_stems = {f.stem.replace('.ome', '') for f in qupath_exports_dir.glob("*.geojson")}
    all_wsi_paths = list(raw_wsis_dir.glob("*.tif")) + list(raw_wsis_dir.glob("*.vsi"))
    all_usable_wsi_paths = sorted(
        [p for p in all_wsi_paths if not p.name.startswith('.')],
        key=lambda p: p.name
    )

    matched_annotated_count = sum(
        1 for p in all_usable_wsi_paths if p.stem.replace('.ome', '') in annotated_stems
    )

    if include_annotated:
        target_wsis = all_usable_wsi_paths
    else:
        target_wsis = [
            p for p in all_usable_wsi_paths
            if p.stem.replace('.ome', '') not in annotated_stems
        ]

    return all_usable_wsi_paths, annotated_stems, target_wsis, matched_annotated_count


def print_slide_selection_debug(config: dict, target_wsis: list[Path], all_usable_wsi_paths: list[Path], annotated_stems: set[str], matched_annotated_count: int):
    """
    Prints a compact debug report for slide-selection logic.
    """
    inference_root = Path(config['paths']['inference_results'])
    done_count = 0
    for p in target_wsis:
        slide_output_dir = inference_root / p.stem.replace('.ome', '')
        if is_slide_completed(slide_output_dir):
            done_count += 1
    pending_count = len(target_wsis) - done_count

    print("=== Inference Slide Selection Debug ===")
    print(f"raw_wsis_dir: {config['paths']['raw_wsis']}")
    print(f"qupath_exports_dir: {config['paths']['qupath_exports']}")
    print(f"inference_results_dir: {config['paths']['inference_results']}")
    print(f"total_raw_wsis: {len(all_usable_wsi_paths)}")
    print(f"total_geojson_annotation_files: {len(list(Path(config['paths']['qupath_exports']).glob('*.geojson')))}")
    print(f"unique_annotated_stems: {len(annotated_stems)}")
    print(f"raw_wsis_with_annotation_match: {matched_annotated_count}")
    print(f"target_slides_for_inference: {len(target_wsis)}")
    print(f"target_slides_done_marker: {done_count}")
    print(f"target_slides_pending_marker: {pending_count}")

    if target_wsis:
        print("target_slide_sample:")
        for p in target_wsis[:15]:
            status = "[DONE]" if is_slide_completed(Path(config['paths']['inference_results']) / p.stem.replace('.ome', '')) else "[PENDING]"
            print(f"  - {p.name} {status}")
    else:
        print("target_slide_sample: none")


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
        patch_size: int,
        predict_kwargs: dict | None = None
):
    """
    Runs YOLO prediction on a batch of patches and returns global-coordinate detections.
    """
    if not patch_images:
        return []

    batch_detections = []
    results = model.predict(
        source=patch_images,
        conf=conf_threshold,
        verbose=False,
        **(predict_kwargs or {})
    )

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


def process_single_wsi_inference(
    wsi_path: Path,
    model,
    config: dict,
    logger,
    batch_size_override: int | None = None,
    force: bool = False,
    model_class_names: list[str] | None = None,
    model_source_path: str | None = None,
):
    """
    Runs inference on a single WSI by breaking it into patches.

    Args:
        wsi_path (Path): Path to the WSI file to process.
        model (YOLO): The trained YOLO model instance.
        config (dict): The project configuration dictionary.
        logger: The logger instance for logging messages.
    """
    logger.info(f"--- Starting inference on: {wsi_path.name} ---")
    import torch

    params = config['inference']
    patch_size = config['dataset_creation']['patch_size']  # Use same patch size as training
    patch_overlap = params['patch_overlap']
    batch_size = int(batch_size_override if batch_size_override is not None else params.get("inference_batch_size", 16))
    max_read_retries = int(params.get("max_patch_read_retries", 3))
    retry_sleep_s = float(params.get("patch_read_retry_sleep_seconds", 0.15))
    patch_read_workers = int(params.get("patch_read_workers", max(2, min(8, (os.cpu_count() or 4)))))
    flush_every_batches = int(params.get("detection_flush_every_batches", 8))
    predict_half = bool(params.get("predict_half", True)) and torch.cuda.is_available()
    predict_device = params.get("predict_device", None) or config.get("training", {}).get("device", None)
    predict_kwargs = {
        "imgsz": patch_size,
        "half": predict_half,
    }
    if predict_device:
        predict_kwargs["device"] = predict_device

    if batch_size < 1:
        logger.warning("Invalid inference_batch_size; defaulting to 1.")
        batch_size = 1
    if patch_read_workers < 1:
        logger.warning("Invalid patch_read_workers; defaulting to 1.")
        patch_read_workers = 1
    if flush_every_batches < 1:
        logger.warning("Invalid detection_flush_every_batches; defaulting to 1.")
        flush_every_batches = 1

    # Create a unique output directory for this slide's raw predictions
    slide_output_dir = Path(config['paths']['inference_results']) / wsi_path.stem.replace('.ome', '')
    slide_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Raw prediction files will be saved to: {slide_output_dir}")
    detections_path = slide_output_dir / "detections.jsonl"
    class_names_meta_path = slide_output_dir / "class_names.json"
    if model_class_names:
        meta = {
            "class_names": model_class_names,
            "model_source_path": model_source_path or "",
        }
        class_names_meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Check if this slide has already been completed
    completion_marker = slide_output_dir / ".completed"
    if completion_marker.exists() and not force:
        logger.info(f"Slide {wsi_path.name} already completed. Skipping.")
        return
    if completion_marker.exists() and force:
        completion_marker.unlink()
        logger.info(f"Force mode enabled. Clearing existing completion marker for {wsi_path.name}.")

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
            batch_count_since_flush = 0
            t0 = time.time()

            def _read_patch_task(coord: tuple[int, int]):
                x, y = coord
                try:
                    patch = read_patch_with_retry(
                        zarr_slicer=zarr_slicer,
                        x=x,
                        y=y,
                        patch_size=patch_size,
                        max_retries=max_read_retries,
                        retry_sleep_s=retry_sleep_s
                    )
                    return x, y, patch, None
                except Exception as err:
                    return x, y, None, err

            coord_iter = ((x, y) for y in y_coords for x in x_coords)
            max_inflight_reads = max(batch_size * 2, patch_read_workers)
            inflight: deque[tuple[tuple[int, int], object]] = deque()

            with ThreadPoolExecutor(max_workers=patch_read_workers) as executor:
                for _ in range(max_inflight_reads):
                    coord = next(coord_iter, None)
                    if coord is None:
                        break
                    inflight.append((coord, executor.submit(_read_patch_task, coord)))

                with tqdm(total=total_patches, desc=f"Patches for {wsi_path.stem}") as pbar:
                    while inflight:
                        (x, y), future = inflight.popleft()
                        next_coord = next(coord_iter, None)
                        if next_coord is not None:
                            inflight.append((next_coord, executor.submit(_read_patch_task, next_coord)))

                        _, _, patch_image, err = future.result()
                        pbar.update(1)

                        if err is not None:
                            failed_patch_count += 1
                            logger.error(f"Failed to process patch at (x={x}, y={y}). Error: {err}", exc_info=True)
                            continue

                        patch_images_batch.append(patch_image)
                        patch_coords_batch.append((x, y))
                        processed_patch_count += 1

                        if len(patch_images_batch) >= batch_size:
                            detections = run_batched_prediction(
                                model=model,
                                patch_images=patch_images_batch,
                                patch_coords=patch_coords_batch,
                                conf_threshold=params['conf_threshold'],
                                patch_size=patch_size,
                                predict_kwargs=predict_kwargs
                            )
                            detection_rows_buffer.extend(detections)
                            batch_count_since_flush += 1
                            if batch_count_since_flush >= flush_every_batches:
                                flush_detections_jsonl(detections_path, detection_rows_buffer)
                                detection_rows_buffer.clear()
                                batch_count_since_flush = 0
                            patch_images_batch.clear()
                            patch_coords_batch.clear()

            if patch_images_batch:
                detections = run_batched_prediction(
                    model=model,
                    patch_images=patch_images_batch,
                    patch_coords=patch_coords_batch,
                    conf_threshold=params['conf_threshold'],
                    patch_size=patch_size,
                    predict_kwargs=predict_kwargs
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
        help="List all target slides and exit. Useful for planning batch jobs."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override inference batch size for patch-level prediction."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run inference even for slides already marked completed."
    )
    parser.add_argument(
        "--include-annotated",
        action="store_true",
        help="Include slides even if they already have matching GeoJSON annotations."
    )
    parser.add_argument(
        "--debug-slide-selection",
        action="store_true",
        help="Print detailed counts used by slide-selection logic and exit."
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if not config:
        return

    logger = setup_logging(Path(config['paths']['logs']), '03_run_inference')
    logger.info("--- Starting Inference Script ---")

    all_usable_wsi_paths, annotated_stems, target_wsis, matched_annotated_count = collect_target_slides(
        config,
        include_annotated=args.include_annotated
    )

    if args.debug_slide_selection:
        print_slide_selection_debug(config, target_wsis, all_usable_wsi_paths, annotated_stems, matched_annotated_count)
        return

    # Handle --list-slides flag
    if args.list_slides:
        logger.info("Slide listing mode. Printing slides and exiting.")
        for i, wsi_path in enumerate(target_wsis):
            completed = is_slide_completed(
                Path(config['paths']['inference_results']) / wsi_path.stem.replace('.ome', '')
            )
            status = "[DONE]" if completed else "[PENDING]"
            print(f"{i}: {wsi_path.name} {status}")
        return

    if not target_wsis:
        if args.include_annotated:
            logger.info("No usable WSIs found to process.")
        else:
            logger.info("No unannotated WSIs found to process. All slides appear to have a corresponding GeoJSON file.")
        return

    # 1. Load the latest trained YOLO model from namespaced model_output_dir
    expected_class_names = get_config_class_names(config)
    model_namespace = get_inference_namespace(config, expected_class_names)
    model_dir = Path(config['paths']['model_output_dir']) / model_namespace
    if not model_dir.exists():
        logger.warning(f"Namespaced model directory not found: {model_dir}. Falling back to root model_output_dir.")
        model_dir = Path(config['paths']['model_output_dir'])

    latest_model_path = find_latest_best_model(model_dir, expected_class_names=expected_class_names)
    if latest_model_path is None:
        # Fall back to generic search if no class-matched run is found.
        latest_model_path = find_latest_best_model(model_dir, expected_class_names=None)

    if latest_model_path is None:
        configured_model_path = Path(config['inference']['model_checkpoint'])
        if configured_model_path.exists():
            latest_model_path = configured_model_path
            logger.warning(
                f"No best.pt found under model_output_dir. Falling back to configured checkpoint: {latest_model_path}"
            )
        else:
            logger.critical(
                f"No trained model found under '{model_dir}' and configured checkpoint does not exist: {configured_model_path}"
            )
            return

    try:
        from ultralytics import YOLO
        logger.info(f"Loading latest trained model from: {latest_model_path}")
        model = YOLO(latest_model_path)
        model_names_map = getattr(model, "names", {})
        if isinstance(model_names_map, dict):
            model_class_names = [model_names_map[i] for i in sorted(model_names_map.keys())]
        elif isinstance(model_names_map, list):
            model_class_names = [str(x) for x in model_names_map]
        else:
            model_class_names = []
        if model_class_names and model_class_names != expected_class_names:
            logger.critical(
                "Class-name mismatch between loaded model and current config. "
                f"Model names={model_class_names}, config names={expected_class_names}. "
                "Aborting inference to avoid swapped labels."
            )
            return
    except Exception as e:
        logger.critical(f"Failed to load YOLO model. Error: {e}", exc_info=True)
        return

    logger.info(f"Found {len(target_wsis)} target WSIs to process.")

    # 2. Determine which slides to process based on arguments
    if args.slide_index is not None:
        # Batch mode: process only slides in the specified range
        start_idx = args.slide_index * args.slides_per_job
        end_idx = start_idx + args.slides_per_job
        slides_to_process = target_wsis[start_idx:end_idx]
        logger.info(f"Batch mode: processing slides {start_idx} to {end_idx-1} ({len(slides_to_process)} slides)")
    else:
        # Normal mode: process all slides
        slides_to_process = target_wsis

    # Run inference on each selected WSI
    for wsi_path in slides_to_process:
        process_single_wsi_inference(
            wsi_path,
            model,
            config,
            logger,
            batch_size_override=args.batch_size,
            force=args.force,
            model_class_names=model_class_names,
            model_source_path=str(latest_model_path)
        )

    logger.info("--- Inference Script Finished ---")


if __name__ == "__main__":
    main()
