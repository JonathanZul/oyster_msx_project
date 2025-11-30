import os
import argparse
import yaml
from pathlib import Path
from ultralytics import YOLO

from src.utils.file_handling import load_config
from src.utils.logging_config import setup_logging


def create_yolo_dataset_config(config: dict, logger):
    """
    Creates the dataset.yaml file required by YOLOv8 for training.

    This file tells the YOLO trainer where the training and validation images are,
    and what the class names are. It's generated programmatically from our main
    config.yaml to ensure consistency.

    Args:
        config (dict): The project configuration dictionary.
        logger: The logger instance.

    Returns:
        Path: The path to the newly created dataset.yaml file.
    """
    yolo_dataset_path = Path(config['paths']['yolo_dataset'])
    dataset_yaml_path = yolo_dataset_path / 'dataset.yaml'

    logger.info(f"Creating YOLO dataset config file at: {dataset_yaml_path}")

    # The paths need to be absolute or relative to where the training is run from.
    # We'll use absolute paths for robustness.
    project_root = Path(__file__).resolve().parents[2]
    train_path = project_root / yolo_dataset_path / 'images' / 'train'
    val_path = project_root / yolo_dataset_path / 'images' / 'val'

    # Extract class names from the config, ensuring they are in the correct order
    classes = config['dataset_creation']['classes']
    class_names = [name for name, idx in sorted(classes.items(), key=lambda item: item[1])]

    dataset_config = {
        'train': str(train_path),
        'val': str(val_path),
        'nc': len(class_names),
        'names': class_names
    }

    try:
        with open(dataset_yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        logger.info("Successfully created dataset.yaml.")
        return dataset_yaml_path
    except Exception as e:
        logger.error(f"Failed to create dataset.yaml. Error: {e}", exc_info=True)
        return None


def find_latest_best_model(model_dir: Path, logger):
    """
    Finds the most recently modified 'best.pt' file in the model directory.
    
    Args:
        model_dir (Path): The directory to search in.
        logger: The logger instance.
        
    Returns:
        Path: The path to the latest best.pt file, or None if not found.
    """
    if not model_dir.exists():
        return None
        
    # Search for all best.pt files recursively
    candidates = list(model_dir.rglob("best.pt"))
    if not candidates:
        return None
    
    # Sort by modification time, newest first
    latest_model = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest_model


def main():
    """
    Main function to execute the YOLO model training process.
    """
    parser = argparse.ArgumentParser(description="Stage 02: Train YOLO Model")
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="config.yaml",
        help="Path to the master configuration file."
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if not config:
        return

    # Setup logging
    log_dir = Path(config['paths']['logs'])
    logger = setup_logging(log_dir, '02_train_yolo')
    logger.info("--- Starting YOLO Model Training Script ---")

    # 1. Create the dataset.yaml file needed for YOLO
    dataset_yaml_path = create_yolo_dataset_config(config, logger)
    if not dataset_yaml_path:
        logger.critical("Could not create dataset config file. Aborting training.")
        return

    # 2. Initialize the YOLO model
    # This will download the pre-trained weights if they don't exist
    train_params = config['training']
    
    model_to_load = train_params['yolo_model']
    
    # Check if we should auto-resume from the latest model
    if train_params.get('use_latest_model', False):
        model_output_dir = Path(config['paths']['model_output_dir'])
        # Ensure we are looking relative to project root if path is relative
        if not model_output_dir.is_absolute():
            project_root = Path(__file__).resolve().parents[2]
            model_output_dir = project_root / model_output_dir
            
        logger.info(f"Searching for latest model in: {model_output_dir}")
        latest_model = find_latest_best_model(model_output_dir, logger)
        
        if latest_model:
            logger.info(f"Found latest model: {latest_model}")
            model_to_load = str(latest_model)
        else:
            logger.warning("use_latest_model is True, but no existing models found. Starting from base model.")

    try:
        logger.info(f"Initializing YOLO model with weights: {model_to_load}")
        model = YOLO(model_to_load)
    except Exception as e:
        logger.critical(f"Failed to initialize YOLO model. Error: {e}", exc_info=True)
        return

    # 3. Start the training process
    logger.info("Starting model training...")
    logger.info(
        f"Parameters: Epochs={train_params['epochs']}, Batch Size={train_params['batch_size']}, Device='{train_params['device']}'")

    model_name = train_params['yolo_model'].split('/')[-1].replace('.pt', '') + '_msx_oyster_run'

    try:
        model.train(
            data=str(dataset_yaml_path),
            epochs=train_params['epochs'],
            batch=train_params['batch_size'],
            device=train_params['device'],
            imgsz=train_params['img_size'],
            project=config['paths']['model_output_dir'],
            name=model_name
        )
        logger.info("--- Model Training Script Finished Successfully ---")

    except Exception as e:
        logger.critical(f"An error occurred during model training. Error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
