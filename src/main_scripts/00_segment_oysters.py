import os
import cv2
import glob
import time
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import our standardized utilities
from src.utils.file_handling import load_config
from src.utils.logging_config import setup_logging, log_config


# --- Helper Functions ---


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


def separate_touching_objects(binary_mask, config, logger, output_dir):
    """
    Uses the Watershed algorithm to separate touching objects in a binary mask.

    This function implements your idea: it "degrades" (erodes) the mask to
    find reliable starting points, then "floods" outwards to find the true
    boundaries.

    Args:
        binary_mask (np.ndarray): The binary mask after morphological operations.
        config (dict): The project configuration dictionary.
        logger: The logger instance for logging messages.
        output_dir (Path): Directory to save debug images.

    Returns:
        np.ndarray: A labeled mask where each separated object has a unique integer value.
    """
    logger.info("Applying Watershed algorithm to separate touching objects...")
    params = config['oyster_segmentation']

    # 1. Find the "sure foreground" by eroding the mask. This is the "degraded"
    # mask you described. It breaks the weak connections between oysters.
    erosion_iter = params['watershed_erosion_iterations']
    erosion_kernel = np.ones((3, 3), np.uint8)
    sure_fg = cv2.erode(binary_mask, erosion_kernel, iterations=erosion_iter)
    save_debug_image(sure_fg, "03_watershed_markers", output_dir, config)

    # 2. Find the "sure background" by dilating the mask.
    sure_bg = cv2.dilate(binary_mask, erosion_kernel, iterations=3)

    # 3. Identify the "unknown" region where the objects might be touching.
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 4. Create markers (the starting points for flooding)
    _, markers = cv2.connectedComponents(sure_fg)
    # Add 1 to all labels so that sure background is 0, not -1
    markers = markers + 1
    # Mark the region of unknown with zero
    markers[unknown == 255] = 0

    # 5. Apply the watershed algorithm
    watershed_img = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(watershed_img, markers)

    # The watershed algorithm marks boundaries with -1. Let's create a visual.
    watershed_viz = watershed_img.copy()
    watershed_viz[markers == -1] = [255, 0, 0]  # Draw dams in red
    save_debug_image(watershed_viz, "04_watershed_boundaries", output_dir, config)

    logger.info(f"Watershed found {markers.max()} distinct objects.")
    return markers


def process_single_wsi(wsi_path, config, logger):
    """
    Processes a single WSI to find and save oyster instance masks using a robust
    watershed and fragment merging pipeline.
    """
    logger.info(f"--- Processing Image: {wsi_path.name} ---")
    clean_stem = wsi_path.stem.replace('.ome', '')
    slide_output_dir = Path(config['paths']['oyster_masks']) / clean_stem
    slide_output_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Masks will be saved to: {slide_output_dir}")

    params = config["oyster_segmentation"]
    low_res_img = extract_downsampled_overview(wsi_path, params["processing_downsample"], logger)
    if low_res_img is None: return

    save_debug_image(low_res_img, "00_overview", slide_output_dir, config)

    # 1. Standard image pre-processing to get a clean binary mask
    gray_img = cv2.cvtColor(low_res_img, cv2.COLOR_RGB2GRAY)
    blurred_img = cv2.GaussianBlur(gray_img, tuple(params["gaussian_blur_kernel"]), 0)
    binary_img = cv2.adaptiveThreshold(
        blurred_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        params["adaptive_thresh_block_size"], params["adaptive_thresh_c"])
    kernel_close = np.ones(tuple(params["morph_close_kernel"]), np.uint8)
    kernel_open = np.ones(tuple(params["morph_open_kernel"]), np.uint8)
    closed_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel_close, iterations=params["morph_close_iter"])
    opened_img = cv2.morphologyEx(closed_img, cv2.MORPH_OPEN, kernel_open, iterations=params["morph_open_iter"])
    save_debug_image(opened_img, "02_morph_opened", slide_output_dir, config)

    # 2. Use Watershed to get a labeled map of all distinct objects
    labeled_mask = separate_touching_objects(opened_img, config, logger, slide_output_dir)
    num_labels = labeled_mask.max()
    if num_labels <= 1:
        logger.warning("Watershed did not find enough objects to proceed. Skipping slide.")
        return

    # 3. Analyze all found objects (parents and children)
    logger.info("Analyzing all detected objects to identify parents and fragments...")
    object_properties = []
    for label_idx in range(2, num_labels + 1):  # Start from 2 (1 is background)
        # Create a mask for just this object
        obj_mask = np.zeros(labeled_mask.shape, dtype=np.uint8)
        obj_mask[labeled_mask == label_idx] = 255

        # Calculate properties
        area = np.sum(obj_mask)
        moments = cv2.moments(obj_mask)
        if moments["m00"] > 0:
            centroid_x = int(moments["m10"] / moments["m00"])
            centroid_y = int(moments["m01"] / moments["m00"])
            object_properties.append({
                'label': label_idx,
                'area': area,
                'centroid': (centroid_x, centroid_y),
                'mask': obj_mask
            })

    if len(object_properties) < params['num_oysters_to_detect']:
        logger.warning("Not enough objects found after analysis. Skipping slide.")
        return

    # 4. Identify the N largest objects as the "Parent Oysters"
    object_properties.sort(key=lambda p: p['area'], reverse=True)
    parent_oysters = object_properties[:params['num_oysters_to_detect']]
    child_fragments = object_properties[params['num_oysters_to_detect']:]

    logger.info(f"Identified {len(parent_oysters)} parent oysters and {len(child_fragments)} child fragments.")

    # 5. Intelligent Fragment Merging (as you proposed)
    if params.get('enable_fragment_merging', False):
        logger.info("Fragment merging is enabled. Assigning valid fragments to nearest parent...")
        min_area_threshold = params.get('min_fragment_area_percent', 0.001) * low_res_img.shape[0] * low_res_img.shape[
            1]

        # Create final masks for each parent
        final_masks = [p['mask'].copy() for p in parent_oysters]

        for fragment in child_fragments:
            if fragment['area'] < min_area_threshold:
                continue  # Ignore noise

            # Find the closest parent based on centroid distance
            distances = [np.linalg.norm(np.array(fragment['centroid']) - np.array(p['centroid'])) for p in
                         parent_oysters]
            closest_parent_idx = np.argmin(distances)

            # Merge the fragment's mask into the closest parent's mask
            final_masks[closest_parent_idx] = cv2.bitwise_or(final_masks[closest_parent_idx], fragment['mask'])
            logger.debug(
                f"Merged fragment (label {fragment['label']}, area {fragment['area']}) into parent {closest_parent_idx + 1}.")
    else:
        logger.info("Fragment merging is disabled. Using only the largest objects.")
        final_masks = [p['mask'] for p in parent_oysters]

    # 6. Save the final, merged masks
    for i, mask in enumerate(final_masks):
        mask_filename = slide_output_dir / f"oyster_{i + 1}_mask.png"
        cv2.imwrite(str(mask_filename), mask)
        logger.info(f"Saved final merged mask to {mask_filename}")

    # 7. Save a final debug visualization
    if params["save_debug_images"]:
        viz_img = low_res_img.copy()
        for i, mask in enumerate(final_masks):
            color = np.random.randint(50, 255, size=3)
            viz_img[mask > 0, :] = viz_img[mask > 0, :] * 0.5 + color * 0.5
        save_debug_image(viz_img, "99_final_segmentation", slide_output_dir, config)

def main():
    """Main function to orchestrate the oyster segmentation process."""
    config = load_config()
    if not config:
        return

    logger = setup_logging(Path(config["paths"]["logs"]), "00_segment_oysters")

    log_config(config, logger)

    logger.info("--- Starting Oyster Segmentation Script (Script 00) ---")

    # Find all WSI files to process
    wsi_dir = Path(config["paths"]["raw_wsis"])
    all_paths = (
            list(wsi_dir.glob("*.tif")) +
            list(wsi_dir.glob("*.tiff")) +
            list(wsi_dir.glob("*.vsi"))
    )
    # This list comprehension filters out any path whose name starts with '._'
    wsi_paths = [p for p in all_paths if not p.name.startswith('._')]

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
