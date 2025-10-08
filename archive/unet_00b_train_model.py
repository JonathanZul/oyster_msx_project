import argparse
import random
import time
from glob import glob
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torchvision
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.utils.file_handling import load_config
from src.utils.logging_config import setup_logging, log_config


# --- PyTorch Dataset for Segmentation ---

class OysterSegDataset(Dataset):
    """
    Custom PyTorch Dataset for loading oyster segmentation data.
    It loads an image, its corresponding multi-class label mask, and a 
    region-of-interest (ROI) mask. It handles data augmentation and tensor conversion.

    Args:
        image_paths (list of str): List of file paths to the input images.
        mask_paths (list of str): List of file paths to the label masks (as NumPy arrays).
        roi_paths (list of str): List of file paths to the ROI masks (as NumPy arrays).
        transform (albumentations.Compose, optional): Albumentations transformations to apply.
    """

    def __init__(self, image_paths, mask_paths, roi_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.roi_paths = roi_paths
        self.transform = transform

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Loads an image and its corresponding masks, applies transformations if specified,

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing:
                - image (torch.Tensor): The input image as a PyTorch Tensor of shape (3, H, W).
                - label_mask_tensor (torch.Tensor): The label mask as a Tensor of shape (2, H, W).
                - roi_mask_tensor (torch.Tensor): The ROI mask as a Tensor of shape (1, H, W).
        """
        # Load all data as NumPy arrays first.
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label_mask = np.load(self.mask_paths[idx], allow_pickle=True)  # Shape: (H, W, 2)
        roi_mask = np.load(self.roi_paths[idx], allow_pickle=True)  # Shape: (H, W)

        # Deconstruct the label mask into a list of 2D masks. This is necessary
        # because albumentations requires all masks in a list to have the same shape.
        masks_to_transform = [label_mask[:, :, 0], label_mask[:, :, 1], roi_mask]

        if self.transform:
            # Albumentations takes NumPy arrays and returns a dictionary of PyTorch Tensors.
            # 'image' will be a Tensor of shape (3, H, W).
            # 'masks' will be a list of Tensors, each of shape (H, W).
            augmented = self.transform(image=image, masks=masks_to_transform)
            image = augmented['image']
            augmented_masks = augmented['masks']

            # Reconstruct the label mask Tensor by stacking the first two augmented masks.
            # The result will be a Tensor of shape (2, H, W).
            label_mask_tensor = torch.stack([augmented_masks[0], augmented_masks[1]], dim=0)

            # The ROI mask is the third one. It needs a channel dimension for the loss function.
            # Shape becomes (1, H, W).
            roi_mask_tensor = augmented_masks[2].unsqueeze(0)
        else:
            # If no transforms are applied, we must do the tensor conversion manually.
            label_mask_tensor = torch.from_numpy(label_mask).permute(2, 0, 1).float()
            roi_mask_tensor = torch.from_numpy(roi_mask).unsqueeze(0).float()

        # Ensure final tensors are float type for the loss function.
        return image.float(), label_mask_tensor.float(), roi_mask_tensor.float()


# --- Training and Validation Core Functions ---

def train_one_epoch(loader, model, optimizer, loss_fn, device, scaler):
    """
    Runs a single epoch of training.

    Args:
        loader (DataLoader): The DataLoader for the training dataset.
        model (torch.nn.Module): The segmentation model to train.
        optimizer (torch.optim.Optimizer): The optimizer for updating model weights.
        loss_fn (callable): The loss function to compute the training loss.
        device (torch.device): The device to run the training on (CPU or GPU).
        scaler (torch.cuda.amp.GradScaler): Scaler for Automatic Mixed Precision (AMP).

    Returns:
        float: The average training loss for the epoch.
    """
    model.train()
    loop = tqdm(loader, desc="Training")
    total_loss = 0
    for data, targets, rois in loop:
        data, targets, rois = data.to(device), targets.to(device), rois.to(device)

        # Forward pass with Automatic Mixed Precision (AMP) for performance on CUDA.
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            predictions = model(data)
            loss = loss_fn(predictions, targets, rois)

        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    return total_loss / len(loader)


def validate(loader, model, loss_fn, device):
    """
    Runs validation, calculating the primary loss and additional human-interpretable metrics.

    Args:
        loader (DataLoader): The DataLoader for the validation dataset.
        model (torch.nn.Module): The segmentation model to validate.
        loss_fn (callable): The loss function to compute the validation loss.
        device (torch.device): The device to run the validation on (CPU or GPU).

    Returns:
        dict: A dictionary containing the average loss, IoU, and Dice scores for the validation set.
    """
    model.eval()
    total_loss = 0
    total_tp = torch.tensor(0, dtype=torch.long, device=device)
    total_fp = torch.tensor(0, dtype=torch.long, device=device)
    total_fn = torch.tensor(0, dtype=torch.long, device=device)
    total_tn = torch.tensor(0, dtype=torch.long, device=device)

    with torch.no_grad():
        for data, targets, rois in loader:
            data, targets, rois = data.to(device), targets.to(device), rois.to(device)

            predictions = model(data)  # These are raw logits from the model
            loss = loss_fn(predictions, targets, rois)
            total_loss += loss.item()

            # Step 1: Get the statistics (true positives, false positives, etc.).
            # This function handles the sigmoid activation and thresholding internally.
            tp, fp, fn, tn = smp.metrics.get_stats(
                output=predictions, target=targets.long(), mode='multilabel', threshold=0.5
            )

            total_tp += tp.sum()
            total_fp += fp.sum()
            total_fn += fn.sum()
            total_tn += tn.sum()

    # Step 2: Calculate metrics from the accumulated stats for the entire validation set.
    # `reduction="micro"` correctly aggregates the stats before calculating the final score.
    epoch_iou = smp.metrics.iou_score(total_tp, total_fp, total_fn, total_tn, reduction="micro")
    epoch_dice = smp.metrics.fbeta_score(total_tp, total_fp, total_fn, total_tn, beta=1, reduction="micro")

    metrics = {"loss": total_loss / len(loader), "iou": epoch_iou.item(), "dice": epoch_dice.item()}
    print(f"Validation ==> Loss: {metrics['loss']:.4f}, IoU: {metrics['iou']:.4f}, Dice: {metrics['dice']:.4f}")
    return metrics


# --- Helper Functions for main() ---
def setup_paths_and_logging(config):
    """
    Sets up directories, logging, and TensorBoard.

    Args:
        config (dict): The project configuration dictionary.

    Returns:
        tuple: A tuple containing:
            - logger: The configured logger instance.
            - writer: The TensorBoard SummaryWriter instance.
            - model_dir: Path to the directory where models will be saved.
    """
    paths = config["paths"]
    model_dir = Path(paths["segmentation_model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(Path(paths["logs"]), "00b_train_seg_model")
    log_config(config, logger)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    experiment_name = f"run_{timestamp}"
    log_dir = Path(paths["tensorboard_log_dir"]) / experiment_name
    writer = SummaryWriter(log_dir=str(log_dir))
    logger.info(f"TensorBoard logs will be saved to: {log_dir}")

    return logger, writer, model_dir


def prepare_dataloaders(config, logger):
    """
    Prepares and splits the data, and creates train/validation DataLoaders.

    Args:
        config (dict): The project configuration dictionary.
        logger: The logger instance for logging messages.
    Returns:
        tuple: A tuple containing:
            - train_loader: DataLoader for the training dataset.
            - val_loader: DataLoader for the validation dataset.
    """
    # --- Load Configurations and Prepare Dataset ---
    seg_config = config["ml_segmentation"]
    paths = config["paths"]
    dataset_dir = Path(paths["segmentation_dataset"])

    all_images = sorted(glob(str(dataset_dir / "images" / "*.png")))
    all_masks = sorted(glob(str(dataset_dir / "masks" / "*.npy")))
    all_rois = sorted(glob(str(dataset_dir / "rois" / "*.npy")))

    combined = list(zip(all_images, all_masks, all_rois))

    # Use a seeded random split for reproducibility.
    seed = seg_config.get("data_split_seed")
    if seed is not None:
        logger.info(f"Using fixed random seed for data split: {seed}")
        random.Random(seed).shuffle(combined)
    else:
        logger.info("Using a random data split (no seed provided).")
        random.shuffle(combined)

    split_idx = int(len(combined) * seg_config["train_val_split"])
    train_files, val_files = combined[:split_idx], combined[split_idx:]
    if not train_files or not val_files:
        logger.critical("Dataset is too small to create a train/val split.")
        return None, None

    train_images, train_masks, train_rois = zip(*train_files)
    val_images, val_masks, val_rois = zip(*val_files)
    logger.info(f"Dataset split: {len(train_images)} training, {len(val_images)} validation samples.")

    # Print out images used for validation for debugging purposes.
    logger.info("Validation images:")
    for img in val_images:
        logger.info(f" - {img}")

    # Define data augmentations.
    target_h, target_w = seg_config["image_size"]
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # ColorJitter could be added to handle stain variations.
        # A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.Resize(height=target_h, width=target_w, interpolation=cv2.INTER_AREA),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    val_transform = A.Compose([
        A.Resize(height=target_h, width=target_w, interpolation=cv2.INTER_AREA),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    train_ds = OysterSegDataset(list(train_images), list(train_masks), list(train_rois), transform=train_transform)
    val_ds = OysterSegDataset(list(val_images), list(val_masks), list(val_rois), transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=seg_config["batch_size"], shuffle=True, num_workers=2,
                              pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=seg_config["batch_size"], shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader


def log_image_predictions(loader, model, writer, epoch, device):
    """
    Logs a grid of images to TensorBoard for visual inspection.

    Args:
        loader (DataLoader): The DataLoader for the validation dataset.
        model (torch.nn.Module): The segmentation model to use for predictions.
        writer (SummaryWriter): The TensorBoard SummaryWriter instance.
        epoch (int): The current epoch number for logging.
        device (torch.device): The device to run the model on (CPU or GPU).

    Returns:
        None
    """
    model.eval()
    data, targets, _ = next(iter(loader))  # Get a single batch of validation data
    data = data.to(device=device)

    with torch.no_grad():
        preds = torch.sigmoid(model(data))  # Get model predictions
        preds = (preds > 0.5).float()  # Threshold probabilities to get binary masks

    # To concatenate images for the grid, they must all have the same number of channels (3 for RGB).
    # We expand the single-channel masks by repeating them 3 times.
    target_mask_ch1 = targets[:, 0:1, :, :].cpu().repeat(1, 3, 1, 1)
    pred_mask_ch1 = preds[:, 0:1, :, :].cpu().repeat(1, 3, 1, 1)

    # Create a grid of images: original image, true mask, predicted mask
    grid = torchvision.utils.make_grid(torch.cat([data.cpu(), target_mask_ch1, pred_mask_ch1]), nrow=data.shape[0])

    writer.add_image(f"Predictions/Validation (Oyster 1)", grid, global_step=epoch)
    model.train()  # Set model back to training mode


# --- Main Training ---

def run_unet_training(
        config: dict,
        logger,
        train_stems: list[str],
        val_stems: list[str],
        model_output_path: Path
) -> Path:
    """
    Process the training of a U-Net model for a single fold of cross-validation.

    Args:
        config (dict): The full project configuration.
        logger: The logger instance.
        train_stems (list[str]): A list of slide stems for the training set.
        val_stems (list[str]): A list of slide stems for the validation set.
        model_output_path (Path): The path where the final best model should be saved.

    Returns:
        Path: The path to the saved best model.
    """
    logger.info(f"--- Starting U-Net Training for a single fold ---")
    logger.info(f"Training on {len(train_stems)} samples, validating on {len(val_stems)} samples.")

    seg_config = config["ml_segmentation"]  # Use the archived segmentation config
    paths = config["paths"]

    # --- Device and TensorBoard Setup ---
    device = torch.device(seg_config.get("device", "cpu"))
    writer = SummaryWriter(log_dir=str(model_output_path.parent / f"tensorboard_{model_output_path.stem}"))
    logger.info(f"Using device: {device}")
    logger.info(f"TensorBoard logs for this fold at: {writer.log_dir}")

    # --- Data Preparation ---
    dataset_dir = Path(paths["segmentation_dataset"])

    def get_and_verify_paths_from_stems(stems: list[str]) -> tuple[list, list, list]:
        """
        Converts slide stems to full file paths and verifies that each file exists.
        Returns only the paths for which all three files (image, mask, roi) are found.
        """
        verified_img_paths, verified_mask_paths, verified_roi_paths = [], [], []

        for s in stems:
            img_path = dataset_dir / "images" / f"{s}.png"
            mask_path = dataset_dir / "masks" / f"{s}.npy"
            roi_path = dataset_dir / "rois" / f"{s}.npy"

            if not img_path.exists():
                logger.warning(f"Image file not found, skipping sample: {img_path}")
                continue
            if not mask_path.exists():
                logger.warning(f"Mask file not found, skipping sample: {mask_path}")
                continue
            if not roi_path.exists():
                logger.warning(f"ROI file not found, skipping sample: {roi_path}")
                continue

            verified_img_paths.append(str(img_path))
            verified_mask_paths.append(str(mask_path))
            verified_roi_paths.append(str(roi_path))

        return verified_img_paths, verified_mask_paths, verified_roi_paths

    logger.info("Verifying training data paths...")
    train_images, train_masks, train_rois = get_and_verify_paths_from_stems(train_stems)
    logger.info("Verifying validation data paths...")
    val_images, val_masks, val_rois = get_and_verify_paths_from_stems(val_stems)

    if not train_images or not val_images:
        logger.critical(
            "Could not find any valid data for training or validation after path verification. Aborting fold.")
        # Return a sentinel value to indicate failure
        return None

    # Define transforms
    target_h, target_w = seg_config["image_size"]
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
        A.ColorJitter(p=0.8),
        A.Resize(height=target_h, width=target_w, interpolation=cv2.INTER_AREA),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    val_transform = A.Compose([
        A.Resize(height=target_h, width=target_w, interpolation=cv2.INTER_AREA),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    train_ds = OysterSegDataset(train_images, train_masks, train_rois, transform=train_transform)
    val_ds = OysterSegDataset(val_images, val_masks, val_rois, transform=val_transform)
    train_loader = DataLoader(train_ds, batch_size=seg_config["batch_size"], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=seg_config["batch_size"], shuffle=False, num_workers=2)

    # --- Model, Loss, and Optimizer Setup ---
    model = smp.Unet(
        encoder_name=seg_config["encoder"], encoder_weights=seg_config["encoder_weights"],
        in_channels=3, classes=len(seg_config["classes"]),
    ).to(device)

    dice_loss_fn = smp.losses.DiceLoss(mode='multilabel', from_logits=True)
    bce_loss_fn = smp.losses.SoftBCEWithLogitsLoss(reduction='none')

    def masked_loss(p, t, r):
        return 0.5 * dice_loss_fn(p, t) + 0.5 * (bce_loss_fn(p, t) * r).sum() / (r.sum() + 1e-6)

    loss_fn = masked_loss

    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == 'cuda'))
    # --- Training Loop ---
    best_val_loss = float("inf")
    epochs = seg_config.get("epochs_per_fold", 20)  # Use a specific key for CV

    # For simplicity in this callable function, we'll do a single-phase training.
    # The two-phase logic can be re-added if needed, but this is cleaner for a CV script.
    optimizer = torch.optim.AdamW(model.parameters(), lr=seg_config["learning_rate"])

    for epoch in range(epochs):
        logger.info(f"\n--- Fold Training Epoch {epoch + 1}/{epochs} ---")
        train_loss = train_one_epoch(train_loader, model, optimizer, loss_fn, device, scaler)
        val_metrics = validate(val_loader, model, loss_fn, device)
        val_loss = val_metrics["loss"]

        writer.add_scalar("Loss/train_fold", train_loss, epoch)
        writer.add_scalar("Loss/val_fold", val_loss, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_output_path)
            logger.info(f"Validation loss improved. Saved best model for this fold to {model_output_path}")

    writer.close()
    logger.info(f"--- Finished U-Net Training for fold. Best validation loss: {best_val_loss:.4f} ---")
    return model_output_path

def main():
    """
    Main function for the model training process.

    This function handles command-line arguments, loads the configuration,
    sets up logging, prepares the data loaders, initializes the model and loss function,
    and runs the two-phase training loop with early stopping.
    """
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Stage 00b: Train Segmentation Model")
    parser.add_argument("-c", "--config", type=str, default="config.yaml", help="Path to the config file.")
    args = parser.parse_args()

    # --- Load Configuration ---
    config = load_config(args.config)
    if not config: return

    # --- Setup Logging and TensorBoard ---
    logger, writer, model_dir = setup_paths_and_logging(config)
    seg_config = config["ml_segmentation"]

    # --- Device Selection ---
    device_str = seg_config.get("device", "cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available. Falling back to CPU.")
        device_str = "cpu"
    elif device_str == "mps" and not torch.backends.mps.is_available():
        device_str = "cpu"
    device = torch.device(device_str)
    logger.info(f"Using device: {device}")

    # --- Data Preparation ---
    train_loader, val_loader = prepare_dataloaders(config, logger)
    if not train_loader: return

    # --- Model and Loss Initialization ---
    model = smp.Unet(
        encoder_name=seg_config["encoder"],
        encoder_weights=seg_config["encoder_weights"],
        in_channels=3,
        classes=len(seg_config["classes"]),
    ).to(device)

    # This is the custom hybrid loss function that uses the ROI mask.
    dice_loss_fn = smp.losses.DiceLoss(mode='multilabel', from_logits=True)
    bce_loss_fn = smp.losses.SoftBCEWithLogitsLoss(reduction='none')  # Get per-pixel loss

    def masked_loss(predictions, targets, roi_mask):
        """
        Custom loss function that combines Dice loss and masked BCE loss.

        Args:
            predictions (torch.Tensor): Model predictions (logits).
            targets (torch.Tensor): Ground truth masks.
            roi_mask (torch.Tensor): Region of interest mask to focus loss calculation.

        Returns:
            torch.Tensor: Combined loss value.
        """
        loss_dice = dice_loss_fn(predictions, targets)
        pixel_bce_loss = bce_loss_fn(predictions, targets)
        masked_bce = pixel_bce_loss * roi_mask  # Zero out loss for background pixels
        bce_mean_loss = masked_bce.sum() / (roi_mask.sum() + 1e-6)  # Average over ROI only
        return 0.5 * loss_dice + 0.5 * bce_mean_loss

    loss_fn = masked_loss
    logger.info("Using a combined Dice + MASKED BCE loss function.")

    # --- Two-Phase Training Loop ---
    best_val_loss = float("inf")
    total_epochs = seg_config["epochs"]
    freeze_epochs = seg_config.get("freeze_epochs", 0)
    patience = seg_config.get("early_stopping_patience", 10)
    epochs_without_improvement = 0
    best_model_path = model_dir / "best_model.pt"
    latest_model_path = model_dir / "latest_model.pt"
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    # PHASE 1: Train Decoder with Frozen Encoder
    if freeze_epochs > 0:
        logger.info(f"--- Starting PHASE 1: Training Decoder ({freeze_epochs} epochs) ---")
        for param in model.encoder.parameters(): param.requires_grad = False
        optimizer = torch.optim.AdamW(model.parameters(), lr=seg_config["learning_rate"])

        for epoch in range(freeze_epochs):
            logger.info(f"\n--- Freeze Epoch {epoch + 1}/{freeze_epochs} ---")
            train_loss = train_one_epoch(train_loader, model, optimizer, loss_fn, device, scaler)
            val_metrics = validate(val_loader, model, loss_fn, device)
            val_loss = val_metrics["loss"]

            # Log all metrics to TensorBoard
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/validation", val_loss, epoch)
            writer.add_scalar("Metrics/IoU", val_metrics["iou"], epoch)
            writer.add_scalar("Metrics/Dice", val_metrics["dice"], epoch)
            writer.add_scalar("Hyperparameters/learning_rate", seg_config["learning_rate"], epoch)

            torch.save(model.state_dict(), latest_model_path)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                torch.save(model.state_dict(), best_model_path)
                logger.info("Validation loss improved. Saved best model.")
            else:
                epochs_without_improvement += 1
                logger.info(f"Validation loss did not improve. Patience: {epochs_without_improvement}/{patience}")

            if epochs_without_improvement >= patience:
                logger.warning("Early stopping triggered in Phase 1.")
                break

    # PHASE 2: Fine-tune the Full Model
    finetune_epochs = total_epochs - freeze_epochs
    if finetune_epochs > 0:
        logger.info(f"--- Starting PHASE 2: Fine-tuning Full Model ({finetune_epochs} epochs) ---")
        for param in model.encoder.parameters(): param.requires_grad = True
        finetune_lr = seg_config["learning_rate"] * seg_config.get("finetune_lr_factor", 0.1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=finetune_lr)
        logger.info(f"Fine-tuning with new learning rate: {finetune_lr}")
        epochs_without_improvement = 0

        for epoch_idx in range(finetune_epochs):
            global_epoch = freeze_epochs + epoch_idx

            logger.info(f"\n--- Finetune Epoch {epoch_idx + 1}/{finetune_epochs} ---")
            train_loss = train_one_epoch(train_loader, model, optimizer, loss_fn, device, scaler)
            val_metrics = validate(val_loader, model, loss_fn, device)
            val_loss = val_metrics["loss"]

            # Log all metrics to TensorBoard
            writer.add_scalar("Loss/train", train_loss, global_epoch)
            writer.add_scalar("Loss/validation", val_loss, global_epoch)
            writer.add_scalar("Metrics/IoU", val_metrics["iou"], global_epoch)
            writer.add_scalar("Metrics/Dice", val_metrics["dice"], global_epoch)
            writer.add_scalar("Hyperparameters/learning_rate", finetune_lr, global_epoch)

            torch.save(model.state_dict(), latest_model_path)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                torch.save(model.state_dict(), best_model_path)
                logger.info("Validation loss improved. Saved best model.")
            else:
                epochs_without_improvement += 1
                logger.info(f"Validation loss did not improve. Patience: {epochs_without_improvement}/{patience}")

            if epochs_without_improvement >= patience:
                logger.warning("Early stopping triggered in Phase 2.")
                break

            if global_epoch % 5 == 0:
                log_image_predictions(val_loader, model, writer, global_epoch, device)

    logger.info("--- Segmentation Model Training Finished ---")
    logger.info(f"Best validation loss achieved: {best_val_loss:.4f}")
    logger.info(f"Best model saved to: {best_model_path}")
    logger.info(f"Latest model saved to: {latest_model_path}")
    writer.close()


if __name__ == "__main__":
    """This block allows the script to be run standalone for testing, but it's not used in the CV pipeline."""
    parser = argparse.ArgumentParser(description="Standalone U-Net Trainer")
    parser.add_argument("-c", "--config", type=str, default="config.yaml", help="Path to config file.")
    args = parser.parse_args()

    config = load_config(args.config)
    if config:
        logger = setup_logging(Path(config["paths"]["logs"]), "unet_standalone_train")
        log_config(config, logger)

        # This is a dummy run for testing purposes. It splits the data and runs one training.
        dataset_dir = Path(config["paths"]["segmentation_dataset"])
        all_stems = [p.stem for p in (dataset_dir / "images").glob("*.png")]
        random.shuffle(all_stems)
        split_idx = int(len(all_stems) * 0.8)
        train_stems = all_stems[:split_idx]
        val_stems = all_stems[split_idx:]

        output_path = Path("models/unet_standalone_test.pt")
        output_path.parent.mkdir(exist_ok=True)

        run_unet_training(config, logger, train_stems, val_stems, output_path)
