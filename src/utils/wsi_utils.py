import cv2
import tifffile
import numpy as np
from pathlib import Path


def extract_downsampled_overview(wsi_path, target_downsample, logger):
    """
    Extracts the best low-resolution overview image from a WSI.

    This function finds the level in the WSI pyramid that is closest to
    the desired downsample factor and returns it as an RGB image.

    Args:
        wsi_path (Path): The path to the Whole Slide Image file.
        target_downsample (float): The desired downsample factor (e.g., 32.0).
        logger: The logger instance for logging messages.

    Returns:
        np.ndarray: The downsampled overview image as an RGB NumPy array,
                    or None if an error occurs.
    """
    logger.info(f"Extracting overview from: {wsi_path.name}")
    try:
        with tifffile.TiffFile(wsi_path) as tif:
            main_series = tif.series[0]
            base_width = main_series.levels[0].shape[1]

            # Find the pyramid level closest to our target downsample
            available_ds = [base_width / level.shape[1] for level in main_series.levels]
            logger.info(f"Available downsamples in TIFF: {[round(ds, 1) for ds in available_ds]}")

            # Find the index of the first level where downsample is >= target
            best_level_idx_arr = np.where(np.array(available_ds) >= target_downsample)[0]
            if len(best_level_idx_arr) == 0:
                best_level_idx = len(available_ds) - 1
                logger.warning(
                    f"Target downsample ({target_downsample}x) is high. "
                    f"Using lowest available resolution ({available_ds[best_level_idx]:.1f}x)."
                )
            else:
                best_level_idx = best_level_idx_arr[0]

            actual_ds = available_ds[best_level_idx]
            logger.info(
                f"Selected Level {best_level_idx} with actual downsample ~{actual_ds:.1f}x"
            )

            # Read the image data from the selected level
            overview_raw = main_series.levels[best_level_idx].asarray()

            # Ensure image is in a standard 3-channel RGB format
            if overview_raw.ndim == 3 and overview_raw.shape[2] == 4: # RGBA
                return cv2.cvtColor(overview_raw, cv2.COLOR_RGBA2RGB)
            elif overview_raw.ndim == 2: # Grayscale
                return cv2.cvtColor(overview_raw, cv2.COLOR_GRAY2RGB)
            return overview_raw
    except Exception as e:
        logger.error(f"Error extracting overview from {wsi_path.name}: {e}", exc_info=True)
        return None

def get_wsi_level0_dimensions(wsi_path, logger):
    """
    Gets the dimensions (width, height) of the highest resolution level (Level 0) of a WSI.

    Args:
        wsi_path (Path): The path to the Whole Slide Image file.
        logger: The logger instance.

    Returns:
        tuple[int, int]: A tuple containing (width, height), or None if an error occurs.
    """
    try:
        with tifffile.TiffFile(wsi_path) as tif:
            level0_shape = tif.series[0].levels[0].shape
            # Shape is typically (height, width, channels) or (height, width)
            height, width = level0_shape[0], level0_shape[1]
            return (width, height)
    except Exception as e:
        logger.error(f"Could not read Level 0 dimensions from {wsi_path.name}: {e}", exc_info=True)
        return None


def find_matching_wsi_path(annotation_path: Path, wsi_dir: Path):
    """
    Finds a corresponding WSI file for a given annotation file path.

    Args:
        annotation_path (Path): Path to the annotation file (e.g., .geojson).
        wsi_dir (Path): The directory containing the WSI files.

    Returns:
        Path | None: The path to the matching WSI file or None if not found.
    """
    slide_stem = annotation_path.stem
    possible_wsi_paths = list(wsi_dir.glob(f"{slide_stem}.*"))

    # Filter for common WSI file extensions
    wsi_path = next(
        (
            p for p in possible_wsi_paths
            if p.suffix.lower() in [".tif", ".tiff", ".vsi"] and not p.name.startswith('._')
        ),
        None,
    )
    return wsi_path
