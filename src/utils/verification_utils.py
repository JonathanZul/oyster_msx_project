import random
from pathlib import Path


def setup_verification_directory(output_dir_name: str = "verification_outputs") -> Path:
    """
    Creates and returns the path to the verification output directory.

    Args:
        output_dir_name (str): Name of the verification output directory.
    Returns:
        Path: Path to the verification output directory.
    """
    verification_dir = Path(output_dir_name)
    verification_dir.mkdir(exist_ok=True)
    return verification_dir


def get_random_image_paths(dataset_dir: Path, num_samples: int, logger) -> list[Path]:
    """
    Finds all valid image paths in a dataset, filters hidden files,
    and returns a random subset of paths.

    Args:
        dataset_dir (Path): Path to the dataset directory containing 'images' subdirectory.
        num_samples (int): Number of random image paths to return.
        logger: Logger instance for logging messages.
    Returns:
        list[Path]: List of randomly selected image paths.
    """
    logger.info(f"Searching for images in: {dataset_dir / 'images'}")

    # Gather all image paths
    all_image_paths = sorted(list((dataset_dir / "images").glob("*.png")))

    # Standardized filtering for hidden system files
    valid_image_paths = [p for p in all_image_paths if not p.name.startswith('._')]

    # Ensure there are valid images to sample from
    if not valid_image_paths:
        logger.error(f"No valid images found in {dataset_dir / 'images'}")
        return []

    logger.info(f"Found {len(valid_image_paths)} valid images.")

    # Select random samples
    num_to_check = min(num_samples, len(valid_image_paths))
    logger.info(f"Selecting {num_to_check} random samples for verification.")

    # Use random.sample to avoid duplicates
    random_indices = random.sample(range(len(valid_image_paths)), num_to_check)
    selected_paths = [valid_image_paths[i] for i in random_indices]

    return selected_paths
