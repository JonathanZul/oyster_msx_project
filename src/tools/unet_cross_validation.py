# src/tools/run_cross_validation.py
"""
Performs a K-fold cross-validation experiment specifically for the U-Net
segmentation model.

This script manages the end-to-end process of:
1.  Splitting the pre-generated patch dataset into K folds.
2.  For each fold:
    a. Initializing a new U-Net model.
    b. Training the model on the training splits of the current fold.
    c. Evaluating the trained model's performance on the hold-out test split.
3.  Aggregating the performance metrics from all folds to calculate the model's
    average performance and standard deviation, providing a robust measure of
    its effectiveness.
"""

import argparse
import random
import time
from glob import glob
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Add src to path to allow for relative imports from the project root.
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.file_handling import load_config
from src.utils.logging_config import setup_logging, log_config
from src.utils.evaluation_utils import calculate_segmentation_metrics
# We reuse the Dataset class and core training functions from the trainer
from archive.unet_00b_train_model import (
    OysterSegDataset,
    train_one_epoch,
    validate  # Note: we need to adapt this for testing
)


def evaluate_on_test_set(loader, model, device, logger):
    """
    Evaluates a trained model on a hold-out test set using the centralized
    NumPy-based metrics for a fair comparison.
    """
    logger.info("Evaluating model on the hold-out test fold...")
    model.eval()

    # Store all predictions and ground truths to calculate metrics at the end
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, targets, _ in tqdm(loader, desc="Testing"):
            data = data.to(device)
            # Get raw model output (logits)
            predictions_logits = model(data)
            # Convert logits to binary prediction masks (0 or 1)
            predictions_binary = (torch.sigmoid(predictions_logits) > 0.5).cpu().numpy()

            all_preds.append(predictions_binary)
            all_targets.append(targets.cpu().numpy())

    # Concatenate all batches into single large arrays
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # --- Use the EXACT same evaluation logic as the other script ---
    fold_metrics_oyster1 = []
    fold_metrics_oyster2 = []

    # Iterate through each sample in the test set
    for i in range(len(all_preds)):
        pred_mask_1 = all_preds[i, 0, :, :]  # Get mask for Oyster 1
        pred_mask_2 = all_preds[i, 1, :, :]  # Get mask for Oyster 2

        gt_mask_1 = all_targets[i, 0, :, :]
        gt_mask_2 = all_targets[i, 1, :, :]

        # NOTE: For cross-validation, we assume labels are not swapped.
        # Intelligent matching is for the final pipeline where order is unknown.
        metrics1 = calculate_segmentation_metrics(pred_mask_1, gt_mask_1)
        metrics2 = calculate_segmentation_metrics(pred_mask_2, gt_mask_2)

        fold_metrics_oyster1.append(metrics1)
        fold_metrics_oyster2.append(metrics2)

    # Average the metrics across all samples in the fold
    avg_iou = np.mean([m["iou"] for m in fold_metrics_oyster1] + [m["iou"] for m in fold_metrics_oyster2])
    avg_dice = np.mean([m["dice"] for m in fold_metrics_oyster1] + [m["dice"] for m in fold_metrics_oyster2])
    avg_j_and_f = np.mean([m["j_and_f"] for m in fold_metrics_oyster1] + [m["j_and_f"] for m in fold_metrics_oyster2])

    final_metrics = {"iou": avg_iou, "dice": avg_dice, "j_and_f": avg_j_and_f}
    logger.info(
        f"Test Set Performance ==> IoU: {final_metrics['iou']:.4f}, Dice: {final_metrics['dice']:.4f}, J&F: {final_metrics['j_and_f']:.4f}")

    return final_metrics


def main():
    """Main function to orchestrate the K-Fold Cross-Validation process."""
    parser = argparse.ArgumentParser(description="Run K-Fold Cross-Validation for Segmentation")
    parser.add_argument("-c", "--config", type=str, default="config.yaml", help="Path to the config file.")
    parser.add_argument("-k", "--folds", type=int, default=5, help="Number of folds for cross-validation.")
    args = parser.parse_args()

    config = load_config(args.config)
    if not config: return

    logger = setup_logging(Path("logs"), "cross_validation")
    log_config(config, logger)
    logger.info(f"--- Starting {args.folds}-Fold Cross-Validation ---")

    seg_config = config["ml_segmentation"]
    paths = config["paths"]

    # --- Data Preparation ---
    dataset_dir = Path(paths["segmentation_dataset"])
    all_images = sorted(glob(str(dataset_dir / "images" / "*.png")))
    all_masks = sorted(glob(str(dataset_dir / "masks" / "*.npy")))
    all_rois = sorted(glob(str(dataset_dir / "rois" / "*.npy")))

    # Create the full dataset instance once. Transforms will be assigned per-fold.
    full_dataset = OysterSegDataset(all_images, all_masks, all_rois, transform=None)

    # Random seed for K-Fold splits
    seed = seg_config.get("data_split_seed")
    if seed is not None:
        logger.info(f"Using fixed random seed for K-Fold splits: {seed}")
        kf = KFold(n_splits=args.folds, shuffle=True, random_state=seed)
    else:
        logger.info("Using a random K-Fold split (no seed provided).")
        kf = KFold(n_splits=args.folds, shuffle=True)

    # Define transforms (these are the same for all folds)
    target_h, target_w = seg_config["image_size"]
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
        A.Resize(height=target_h, width=target_w, interpolation=cv2.INTER_AREA),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    test_transform = A.Compose([
        A.Resize(height=target_h, width=target_w, interpolation=cv2.INTER_AREA),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(full_dataset)):
        logger.info(f"\n{'=' * 20} FOLD {fold + 1}/{args.folds} {'=' * 20}")

        train_subset = Subset(full_dataset, train_idx)
        test_subset = Subset(full_dataset, test_idx)

        # We need to access the underlying dataset to set the transform
        train_subset.dataset.transform = train_transform
        test_subset.dataset.transform = test_transform

        train_loader = DataLoader(train_subset, batch_size=seg_config["batch_size"], shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=seg_config["batch_size"], shuffle=False)

        # --- Per-Fold Model Training ---
        device = torch.device(seg_config.get("device", "cpu"))
        model = smp.Unet(
            encoder_name=seg_config["encoder"], encoder_weights=seg_config["encoder_weights"],
            in_channels=3, classes=len(seg_config["classes"]),
        ).to(device)

        dice_loss_fn = smp.losses.DiceLoss(mode='multilabel', from_logits=True)
        bce_loss_fn = smp.losses.SoftBCEWithLogitsLoss(reduction='none')

        def masked_loss(p, t, r):
            return 0.5 * dice_loss_fn(p, t) + 0.5 * (bce_loss_fn(p, t) * r).sum() / (r.sum() + 1e-6)

        optimizer = torch.optim.AdamW(model.parameters(), lr=seg_config["learning_rate"])
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

        epochs_to_train = seg_config.get("epochs_per_fold", 20)  # Use a specific key for this
        logger.info(f"Training model for {epochs_to_train} epochs...")
        for epoch in range(epochs_to_train):
            train_one_epoch(train_loader, model, optimizer, masked_loss, device, scaler)

        # --- Per-Fold Model Evaluation ---
        fold_metrics = evaluate_on_test_set(test_loader, model, device, logger)
        fold_metrics["fold"] = fold + 1
        fold_results.append(fold_metrics)

    # --- Final Report ---
    logger.info(f"\n\n{'=' * 20} CROSS-VALIDATION SUMMARY {'=' * 20}")
    results_df = pd.DataFrame(fold_results)

    avg_iou = results_df['iou'].mean();
    std_iou = results_df['iou'].std()
    avg_dice = results_df['dice'].mean();
    std_dice = results_df['dice'].std()
    avg_j_and_f = results_df['j_and_f'].mean();
    std_j_and_f = results_df['j_and_f'].std()

    logger.info(f"\nPer-Fold Metrics:\n{results_df.to_string(index=False)}")
    logger.info("\n" + "---" * 10)
    logger.info(f"Overall Average IoU Score:   {avg_iou:.4f} (+/- {std_iou:.4f})")
    logger.info(f"Overall Average Dice Score:  {avg_dice:.4f} (+/- {std_dice:.4f})")
    logger.info(f"Overall Average J&F Score:   {avg_j_and_f:.4f} (+/- {std_j_and_f:.4f})")
    logger.info("---" * 10)


if __name__ == "__main__":
    main()
