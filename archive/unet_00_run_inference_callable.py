# archive/unet_00_run_inference_callable.py

import argparse
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# Add src to path to allow for relative imports
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.file_handling import load_config
from src.utils.logging_config import setup_logging, log_config
from src.utils.wsi_utils import extract_downsampled_overview, find_matching_wsi_path

# We can reuse the clean_mask function from the SAM script
from src.main_scripts.s00_segment_with_sam import clean_mask, _save_debug_image


def run_unet_inference(
        config: dict,
        logger,
        model_path: Path,
        file_stems: list[str],
        output_dir: Path
):
    """
    Runs inference using a trained U-Net model on a specific list of slides.

    Args:
        config (dict): The full project configuration.
        logger: The logger instance.
        model_path (Path): Path to the trained U-Net model checkpoint (.pt file).
        file_stems (list[str]): A list of slide stems to process.
        output_dir (Path): The directory to save the resulting masks.
    """
    logger.info(f"--- Running U-Net Inference for {len(file_stems)} slides ---")
    seg_config = config["ml_segmentation"]  # Use the archived U-Net config

    # --- Device and Model Setup ---
    device = torch.device(seg_config.get("device", "cpu"))
    logger.info(f"Using device: {device}")

    if not model_path.exists():
        logger.critical(f"Model checkpoint not found at: {model_path}. Aborting inference.")
        return

    logger.info(f"Loading trained U-Net model from {model_path}")
    try:
        model = smp.Unet(
            encoder_name=seg_config["encoder"],
            in_channels=3,
            classes=len(seg_config["classes"]),
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    except Exception as e:
        logger.critical(f"Failed to load model: {e}", exc_info=True)
        return

    # --- Find WSIs and Run Inference ---
    wsi_dir = Path(config["paths"]["raw_wsis"])
    wsi_paths = []
    for stem in file_stems:
        annot_placeholder = wsi_dir / f"{stem}.geojson"
        wsi_path = find_matching_wsi_path(annot_placeholder, wsi_dir)
        if wsi_path:
            wsi_paths.append(wsi_path)

    # Define the same transformation as validation (resize and normalize)
    target_h, target_w = seg_config["image_size"]
    transform = A.Compose([
        A.Resize(height=target_h, width=target_w, interpolation=cv2.INTER_AREA),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    for wsi_path in tqdm(wsi_paths, desc="U-Net Inference"):
        logger.info(f"--- Processing slide: {wsi_path.name} ---")
        slide_output_dir = output_dir / wsi_path.stem.replace(".ome", "")
        slide_output_dir.mkdir(parents=True, exist_ok=True)

        overview_img = extract_downsampled_overview(wsi_path, seg_config["overview_downsample"], logger)
        if overview_img is None:
            continue

        original_h, original_w = overview_img.shape[:2]

        input_tensor = transform(image=overview_img)['image'].unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(input_tensor)

        # Post-process the output
        probabilities = torch.sigmoid(logits)
        binary_masks_tensor = (probabilities > seg_config["confidence_threshold"]).cpu()

        final_cleaned_masks = []

        # Process each class channel from the output
        for class_name, class_idx in seg_config["classes"].items():
            channel = class_idx - 1

            mask_small = binary_masks_tensor[0, channel, :, :].numpy().astype(np.uint8) * 255
            mask_resized = cv2.resize(mask_small, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

            # Use the same cleaning logic as the SAM pipeline for a fair comparison
            cleaned_mask = clean_mask(mask_resized, seg_config.get("min_mask_area", 5000))

            # Store cleaned mask for visualization
            final_cleaned_masks.append(cleaned_mask)

            oyster_id = class_name.split(" ")[-1]
            mask_filename = slide_output_dir / f"oyster_{oyster_id}_mask.png"
            cv2.imwrite(str(mask_filename), cleaned_mask)
            logger.info(f"Saved U-Net mask to {mask_filename}")

        if seg_config.get("save_visualization", False):
            logger.info("Creating composite visualization for U-Net output...")
            viz_image = overview_img.copy()
            colors = [(255, 0, 0), (0, 0, 255)]  # Blue, Red

            for i, cleaned_mask in enumerate(final_cleaned_masks):
                if np.sum(cleaned_mask) > 0:
                    color = colors[i % len(colors)]
                    colored_overlay = np.zeros_like(viz_image)
                    colored_overlay[cleaned_mask > 0] = color
                    viz_image = cv2.addWeighted(viz_image, 1.0, colored_overlay, 0.5, 0)
                    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(viz_image, contours, -1, color, 2)

            # Save the visualization in the main slide output directory
            viz_filename = slide_output_dir / f"visualization_unet_{wsi_path.stem}.png"
            # We use cv2.imwrite directly instead of _save_debug_image to save at full resolution
            cv2.imwrite(str(viz_filename), cv2.cvtColor(viz_image, cv2.COLOR_RGB2BGR))
            logger.info(f"Saved U-Net visualization to {viz_filename}")

    logger.info("--- U-Net Inference Finished ---")

# This script is intended to be called from another script, so it has no standalone execution block.
