import argparse
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from src.utils.file_handling import load_config
from src.utils.logging_config import setup_logging, log_config
from src.utils.wsi_utils import extract_downsampled_overview, find_matching_wsi_path


def clean_mask(mask: np.ndarray, min_size: int) -> np.ndarray:
    """
    Cleans a binary mask by removing all but the largest connected component.
    Any component smaller than the specified min_size is considered noise and removed.

    Args:
        mask (np.ndarray): A binary mask with pixel values of 0 or 255.
        min_size (int): The minimum number of pixels for an object to be kept.

    Returns:
        np.ndarray: The cleaned binary mask containing only the largest object.
    """
    # Find all connected components (i.e., separate "blobs") in the mask.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels <= 1:  # No foreground objects found
        return mask

    # Find the label of the largest component, ignoring the background (label 0).
    largest_component_label = -1
    max_area = -1
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > max_area and area > min_size:
            max_area = area
            largest_component_label = i

    # Create a new, blank mask and draw only the largest component onto it.
    cleaned_mask = np.zeros_like(mask)
    if largest_component_label != -1:
        cleaned_mask[labels == largest_component_label] = 255

    return cleaned_mask


def postprocess_and_save_masks(raw_predictions: torch.Tensor, original_dims: tuple, output_dir: Path, config: dict,
                               overview_img: np.ndarray, logger):
    """
    Takes raw model logits and performs all post-processing steps: thresholding,
    cleaning, resizing, and saving the final masks and visualization.

    Args:
        raw_predictions (torch.Tensor): The raw output logits from the segmentation model.
        original_dims (tuple): The original dimensions of the overview image (width, height).
        output_dir (Path): Directory where the final masks and visualizations will be saved.
        config (dict): The project configuration dictionary.
        overview_img (np.ndarray): The overview image used for inference.
        logger: Logger instance for logging messages.

    Returns:
        None
    """
    seg_config = config["ml_segmentation"]
    original_w, original_h = original_dims

    # Convert raw logits to binary masks
    probabilities = torch.sigmoid(raw_predictions)
    binary_masks_tensor = (probabilities > seg_config["confidence_threshold"]).cpu()

    # Prepare for visualization if enabled
    save_viz = seg_config.get("save_visualization", False)
    if save_viz:
        viz_image = overview_img.copy()
        colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0)]  # Blue, Red, Green

    # Process each class channel from the model's output
    class_map = seg_config["classes"]
    for class_name, class_idx in class_map.items():
        # Config is 1-based, tensor is 0-based
        channel = class_idx - 1

        # Convert the small tensor mask to a NumPy array for OpenCV processing
        mask_small = binary_masks_tensor[0, channel, :, :].numpy().astype(np.uint8) * 255

        # Clean the mask to keep only the largest, most confident prediction
        logger.info(f"Cleaning mask for '{class_name}'...")
        mask_cleaned = clean_mask(mask_small, seg_config["min_object_size"])

        if np.sum(mask_cleaned) == 0:
            logger.warning(f"No objects of sufficient size found for '{class_name}'. Mask will be empty.")

        # Resize the clean mask back to the original overview's dimensions for saving
        mask_final = cv2.resize(mask_cleaned, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

        # Save the final mask, matching the format required by downstream scripts
        oyster_id = class_name.split(" ")[-1]
        mask_filename = output_dir / f"oyster_{oyster_id}_mask.png"
        cv2.imwrite(str(mask_filename), mask_final)
        logger.info(f"Saved final mask to {mask_filename}")

        # If enabled, draw this mask onto the visualization image
        if save_viz:
            color = colors[channel % len(colors)]
            colored_overlay = np.zeros_like(viz_image)
            colored_overlay[mask_final > 0] = color
            viz_image = cv2.addWeighted(viz_image, 1.0, colored_overlay, 0.5, 0)
            contours, _ = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(viz_image, contours, -1, color, 2)

    # Save the final composite visualization image
    if save_viz:
        slide_stem = output_dir.name
        viz_filename = output_dir / f"visualization_{slide_stem}.png"
        cv2.imwrite(str(viz_filename), viz_image)
        logger.info(f"Saved visualization to {viz_filename}")


def load_segmentation_model(config: dict, device, logger):
    """
    Initializes the U-Net model and loads the trained weights from a checkpoint.

    Args:
        config (dict): The project configuration dictionary.
        device: The device to load the model onto (CPU or GPU).
        logger: Logger instance for logging messages.

    Returns:
        smp.Unet | None: The loaded segmentation model or None if loading failed.
    """
    seg_config = config["ml_segmentation"]
    model_path = Path(seg_config["model_checkpoint"])

    if not model_path.exists():
        logger.critical(f"Model checkpoint not found at: {model_path}")
        return None

    logger.info(f"Loading trained model from {model_path}")
    try:
        model = smp.Unet(
            encoder_name=seg_config["encoder"],
            in_channels=3,
            classes=len(seg_config["classes"]),
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()  # Set model to evaluation mode
        return model
    except Exception as e:
        logger.critical(f"Failed to load model from {model_path}: {e}", exc_info=True)
        return None


def main():
    """
    Main function to orchestrate the ML-based oyster segmentation inference process.

    This script loads the configuration, sets up logging, initializes the model,
    processes each WSI, and saves the segmentation masks.
    """
    # --- Parse command-line arguments ---
    parser = argparse.ArgumentParser(description="Stage 00: Run ML-Based Oyster Segmentation")
    parser.add_argument("-c", "--config", type=str, default="config.yaml", help="Path to the config file.")
    args = parser.parse_args()

    # --- Load Config ---
    config = load_config(args.config)
    if not config: return

    # --- Setup Logging ---
    logger = setup_logging(Path(config["paths"]["logs"]), "00_run_segmentation")
    log_config(config, logger)
    logger.info("--- Starting ML-Based Segmentation Inference (Script 00) ---")

    seg_config = config["ml_segmentation"]

    # --- Device and Model Setup ---
    device_str = seg_config.get("device", "cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available. Falling back to CPU.");
        device_str = "cpu"
    elif device_str == "mps" and not torch.backends.mps.is_available():
        logger.warning("MPS not available. Falling back to CPU.");
        device_str = "cpu"
    device = torch.device(device_str)

    model = load_segmentation_model(config, device, logger)
    if model is None: return

    # --- Find WSIs and Run Inference ---
    wsi_dir = Path(config["paths"]["raw_wsis"])
    wsi_paths = [p for p in wsi_dir.iterdir() if
                 p.suffix.lower() in [".tif", ".tiff", ".vsi"] and not p.name.startswith('.')]

    if not wsi_paths:
        logger.error(f"No WSI files found in '{wsi_dir}'. Exiting.")
        return

    logger.info(f"Found {len(wsi_paths)} WSIs to process.")
    for wsi_path in tqdm(wsi_paths, desc="Segmenting WSIs"):
        logger.info(f"--- Processing slide: {wsi_path.name} ---")

        # Step 1: Prepare the input image
        overview_img = extract_downsampled_overview(wsi_path, seg_config["overview_downsample"], logger)
        if overview_img is None: continue

        target_h, target_w = seg_config["image_size"]
        transform = A.Compose([
            A.Resize(height=target_h, width=target_w, interpolation=cv2.INTER_AREA),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
        input_tensor = transform(image=overview_img)['image'].unsqueeze(0).to(device)

        # Step 2: Run model inference
        with torch.no_grad():
            raw_predictions = model(input_tensor)

        # Step 3: Post-process and save results
        output_dir = Path(config["paths"]["oyster_masks"]) / wsi_path.stem.replace(".ome", "")
        output_dir.mkdir(parents=True, exist_ok=True)
        original_dims = (overview_img.shape[1], overview_img.shape[0])  # (W, H)

        postprocess_and_save_masks(raw_predictions, original_dims, output_dir, config, overview_img, logger)

    logger.info("--- ML-Based Segmentation Finished ---")


if __name__ == "__main__":
    main()
