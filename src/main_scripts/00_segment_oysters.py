import os
import cv2
import glob
import time
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import our standardized utilities
from utils.file_handling import load_config
from utils.logging_config import setup_logging


# --- Helper Functions (Adapted from your original script) ---


def save_debug_image(image_data, filename_suffix, output_dir, config, cmap=None):
    """Saves a downsampled debug image if enabled in the config."""
    if not config["oyster_segmentation"]["save_debug_images"]:
        return

    debug_img_dir = output_dir / "debug_images"
    debug_img_dir.mkdir(exist_ok=True)
    save_path = str(debug_img_dir / f"{filename_suffix}.png")

    max_dim = config["oyster_segmentation"]["debug_img_max_dim"]
    h, w = image_data.shape[:2]

    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized_image_data = cv2.resize(
            image_data, (new_w, new_h), interpolation=cv2.INTER_AREA
        )
    else:
        resized_image_data = image_data

    try:
        if resized_image_data.ndim == 3:  # RGB
            cv2.imwrite(save_path, cv2.cvtColor(resized_image_data, cv2.COLOR_RGB2BGR))
        else:  # Grayscale or binary
            cv2.imwrite(save_path, resized_image_data)
    except Exception as e:
        print(f"Error saving debug image {save_path}: {e}")


def extract_downsampled_overview(wsi_path, target_downsample, logger):
    """Extracts the best low-resolution overview image from a WSI."""
    logger.info(f"Extracting overview from: {wsi_path.name}")
    try:
        with tifffile.TiffFile(wsi_path) as tif:
            main_series = tif.series[0]
            base_width = main_series.levels[0].shape[1]

            available_ds = [base_width / level.shape[1] for level in main_series.levels]
            logger.info(
                f"Available downsamples in TIFF: {[round(ds, 1) for ds in available_ds]}"
            )

            best_level_idx = np.where(np.array(available_ds) >= target_downsample)[0]
            if len(best_level_idx) == 0:
                best_level_idx = len(available_ds) - 1
                logger.warning(
                    f"Target downsample is high. Using lowest available resolution."
                )
            else:
                best_level_idx = best_level_idx[0]

            actual_ds = available_ds[best_level_idx]
            logger.info(
                f"Selected Level {best_level_idx} with actual downsample ~{actual_ds:.1f}x"
            )

            overview_raw = main_series.levels[best_level_idx].asarray()

            if overview_raw.ndim == 3 and overview_raw.shape[2] == 4:
                return cv2.cvtColor(overview_raw, cv2.COLOR_RGBA2RGB)
            elif overview_raw.ndim == 2:
                return cv2.cvtColor(overview_raw, cv2.COLOR_GRAY2RGB)
            return overview_raw
    except Exception as e:
        logger.error(
            f"Error extracting overview from {wsi_path.name}: {e}", exc_info=True
        )
        return None


def process_single_wsi(wsi_path, config, logger):
    """
    Processes a single WSI to find and save oyster instance masks using contours.
    """
    logger.info(f"--- Processing Image: {wsi_path.name} ---")

    slide_output_dir = Path(config["paths"]["oyster_masks"]) / wsi_path.stem
    slide_output_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Masks will be saved to: {slide_output_dir}")

    params = config["oyster_segmentation"]

    # 1. Load low-resolution image
    low_res_img = extract_downsampled_overview(
        wsi_path, params["processing_downsample"], logger
    )
    if low_res_img is None:
        return
    save_debug_image(low_res_img, "00_overview", slide_output_dir, config)

    # 2. Pre-process for tissue detection
    gray_img = cv2.cvtColor(low_res_img, cv2.COLOR_RGB2GRAY)
    blurred_img = cv2.GaussianBlur(gray_img, tuple(params["gaussian_blur_kernel"]), 0)
    binary_img = cv2.adaptiveThreshold(
        blurred_img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        params["adaptive_thresh_block_size"],
        params["adaptive_thresh_c"],
    )
    save_debug_image(binary_img, "01_binary", slide_output_dir, config)

    # 3. Morphological operations
    kernel_close = np.ones(tuple(params["morph_close_kernel"]), np.uint8)
    closed_img = cv2.morphologyEx(
        binary_img, cv2.MORPH_CLOSE, kernel_close, iterations=params["morph_close_iter"]
    )
    kernel_open = np.ones(tuple(params["morph_open_kernel"]), np.uint8)
    opened_img = cv2.morphologyEx(
        closed_img, cv2.MORPH_OPEN, kernel_open, iterations=params["morph_open_iter"]
    )
    save_debug_image(opened_img, "02_morph_opened", slide_output_dir, config)

    # 4. Find largest contours
    contours, _ = cv2.findContours(
        opened_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        logger.warning("No contours found. Skipping slide.")
        return

    min_area = (
        params["min_contour_area_percent"] * low_res_img.shape[0] * low_res_img.shape[1]
    )
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    if not valid_contours:
        logger.warning("No valid contours found. Skipping slide.")
        return

    selected_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)[
        : params["num_oysters_to_detect"]
    ]
    logger.info(
        f"Found and selected {len(selected_contours)} largest contours as oyster candidates."
    )

    final_masks = []
    for i, contour in enumerate(selected_contours):
        # Create a blank, black image with the same dimensions as the low-res image
        mask = np.zeros(low_res_img.shape[:2], dtype=np.uint8)

        # Draw the current contour filled in white on the blank mask
        cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)

        # Save the final binary mask
        mask_filename = (
            slide_output_dir / f"oyster_{i + 1}_mask.png"
        )  # Simplified filename
        cv2.imwrite(str(mask_filename), mask)
        logger.info(f"Saved contour-based mask to {mask_filename}")
        final_masks.append(mask.astype(bool))  # Store as boolean for visualization

    # 6. Save a final debug visualization with all contour masks
    if params["save_debug_images"]:
        viz_img = low_res_img.copy()
        for i, mask in enumerate(final_masks):
            color = np.random.randint(50, 255, size=3)
            viz_img[mask, :] = viz_img[mask, :] * 0.5 + color * 0.5
        save_debug_image(viz_img, "99_final_segmentation", slide_output_dir, config)


def main():
    """Main function to orchestrate the oyster segmentation process."""
    config = load_config()
    if not config:
        return

    logger = setup_logging(Path(config["paths"]["logs"]), "00_segment_oysters")
    logger.info("--- Starting Oyster Segmentation Script (Script 00) ---")

    # Find all WSI files to process
    wsi_dir = Path(config["paths"]["raw_wsis"])
    wsi_paths = (
        list(wsi_dir.glob("*.tif"))
        + list(wsi_dir.glob("*.tiff"))
        + list(wsi_dir.glob("*.vsi"))
    )

    if not wsi_paths:
        logger.error(f"No WSI files found in '{wsi_dir}'. Exiting.")
        return

    logger.info(f"Found {len(wsi_paths)} WSI files to process.")

    # Process each WSI
    for wsi_path in wsi_paths:
        process_single_wsi(wsi_path, config, logger)

    logger.info("--- Oyster Segmentation Script Finished ---")


if __name__ == "__main__":
    main()
