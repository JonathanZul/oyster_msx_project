import sys
from pathlib import Path

import cv2
import numpy as np

# Add src to path to allow for relative imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.file_handling import load_config
from src.utils.logging_config import setup_logging
from src.utils.verification_utils import setup_verification_directory, get_random_image_paths


def visualize_single_roi(image_path: Path, dataset_dir: Path, verification_dir: Path, logger):
    """
    Loads an image and its ROI mask, creates a visualization, and saves it.

    Args:
        image_path (Path): Path to the image file.
        dataset_dir (Path): Path to the root of the segmentation dataset.
        verification_dir (Path): Path to the directory where verification images will be saved.
        logger: Logger instance for logging messages.

    Returns:
        None
    """
    logger.info(f"Verifying ROI for: {image_path.name}")

    roi_path = dataset_dir / "rois" / f"{image_path.stem}.npy"
    if not roi_path.exists():
        logger.warning(f"No matching ROI mask found for {image_path.name}. Skipping.")
        return

    try:
        image = cv2.imread(str(image_path))
        roi_mask = np.load(roi_path, allow_pickle=True)

        # Create an overlay to visualize the ROI
        overlay = image.copy()
        roi_color = [0, 255, 255]  # Yellow in BGR

        # Add color to the ROI areas
        colored_overlay = np.zeros_like(overlay, dtype=np.uint8)
        colored_overlay[roi_mask > 0] = roi_color

        # Create a blended image to visualize the overlay
        blended_image = cv2.addWeighted(overlay, 1.0, colored_overlay, 0.4, 0)

        # Save the blended image
        output_path = verification_dir / f"verify_roi_{image_path.name}"
        cv2.imwrite(str(output_path), blended_image)

    except Exception as e:
        logger.error(f"Failed to process or save verification for {image_path.name}: {e}", exc_info=True)


def main():
    """
    Main function for the ROI mask verification process.

    This tool randomly selects a few images from the segmentation dataset,
    overlays their ROI masks, and saves the results for manual inspection.
    """
    logger = setup_logging(Path("logs"), "verify_roi_masks")
    logger.info("--- Starting ROI Mask Verification Tool ---")

    # --- Load Config ---
    config = load_config()
    if not config:
        logger.critical("Failed to load config.yaml. Make sure you are running from the project root.")
        return

    # --- Setup Directories ---
    dataset_dir = Path(config["paths"]["segmentation_dataset"])
    verification_dir = setup_verification_directory()
    logger.info(f"Saving verification images to: {verification_dir.resolve()}")

    # --- Select Random Images for Verification ---
    image_paths_to_check = get_random_image_paths(dataset_dir, num_samples=5, logger=logger)

    # --- Visualize and Save ROI Overlays ---
    for image_path in image_paths_to_check:
        visualize_single_roi(image_path, dataset_dir, verification_dir, logger)

    logger.info(f"\nVerification complete. Please check the images in the '{verification_dir}' folder.")
    logger.info("The yellow overlay should perfectly cover all oyster tissue and nothing else.")


if __name__ == "__main__":
    main()
