# src/main_scripts/02_train_yolo.py

import os
import yaml
from pathlib import Path
from ultralytics import YOLO

# Import utility functions from our project
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


def main():
    """
    Main function to orchestrate the YOLO model training process.
    """
    # Load configuration
    config = load_config()
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
    try:
        logger.info(f"Initializing YOLO model with pre-trained weights: {train_params['yolo_model']}")
        model = YOLO(train_params['yolo_model'])
    except Exception as e:
        logger.critical(f"Failed to initialize YOLO model. Error: {e}", exc_info=True)
        return

    # 3. Start the training process
    logger.info("Starting model training...")
    logger.info(
        f"Parameters: Epochs={train_params['epochs']}, Batch Size={train_params['batch_size']}, Device='{train_params['device']}'")

    try:
        model.train(
            data=str(dataset_yaml_path),
            epochs=train_params['epochs'],
            batch=train_params['batch_size'],
            device=train_params['device'],
            imgsz=train_params['img_size'],
            project=config['paths']['model_output_dir'],
            name='yolov8n_msx_oyster_run'  # A name for the training run folder
        )
        logger.info("--- Model Training Script Finished Successfully ---")

    except Exception as e:
        logger.critical(f"An error occurred during model training. Error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
