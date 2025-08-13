import sys
from pathlib import Path

import cv2
import numpy as np

# Add src to path to allow for relative imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.file_handling import load_config
from src.utils.logging_config import setup_logging
from src.utils.verification_utils import setup_verification_directory, get_random_image_paths


def visualize_single_seg(image_path: Path, dataset_dir: Path, verification_dir: Path, logger):
    """
    Loads an image and its label mask, creates a visualization, and saves it.

    Args:
        image_path (Path): Path to the image file.
        dataset_dir (Path): Path to the root of the segmentation dataset.
        verification_dir (Path): Path to the directory where verification images will be saved.
        logger: Logger instance for logging messages.

    Returns:
        None
    """
    logger.info(f"Verifying segmentation for: {image_path.name}")

    # Check for corresponding mask
    mask_path = dataset_dir / "masks" / f"{image_path.stem}.npy"
    if not mask_path.exists():
        logger.warning(f"No matching label mask found for {image_path.name}. Skipping.")
        return

    try:
        # Load image and mask
        image = cv2.imread(str(image_path))
        mask = np.load(mask_path, allow_pickle=True)

        # Create an overlay to visualize the segmentation
        overlay = image.copy()
        colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0)]  # Blue, Red, Green

        # Overlay each channel with a different color
        for c in range(mask.shape[2]):
            channel_mask = mask[:, :, c]
            if np.sum(channel_mask) > 0:
                color = colors[c % len(colors)]
                colored_overlay = np.zeros_like(overlay)
                colored_overlay[channel_mask > 0] = color
                overlay = cv2.addWeighted(overlay, 1.0, colored_overlay, 0.5, 0)

        # Save the overlay image
        output_path = verification_dir / f"verify_seg_{image_path.name}"
        cv2.imwrite(str(output_path), overlay)

    except Exception as e:
        logger.error(f"Failed to process or save verification for {image_path.name}: {e}", exc_info=True)


def main():
    """
    Main function for the label mask verification process.

    This tool randomly selects a few images from the segmentation dataset,
    overlays their label masks, and saves the results for manual inspection.
    """
    logger = setup_logging(Path("logs"), "verify_seg_dataset")
    logger.info("--- Starting Segmentation Dataset Verification Tool ---")

    # --- Load Config ---
    config = load_config()
    if not config:
        logger.critical("Failed to load config.yaml. Make sure you are running from the project root.")
        return

    # --- Setup Directories ---
    dataset_dir = Path(config["paths"]["segmentation_dataset"])
    verification_dir = setup_verification_directory()
    logger.info(f"Saving verification images to: {verification_dir.resolve()}")

    # --- Select Random Samples for Verification ---
    image_paths_to_check = get_random_image_paths(dataset_dir, num_samples=5, logger=logger)

    # --- Visualize and Save Segmentation Overlays ---
    for image_path in image_paths_to_check:
        visualize_single_seg(image_path, dataset_dir, verification_dir, logger)

    logger.info(f"\nVerification complete. Please check the images in the '{verification_dir}' folder.")
    logger.info("The colored overlays should perfectly match your QuPath annotations.")


if __name__ == "__main__":
    main()
