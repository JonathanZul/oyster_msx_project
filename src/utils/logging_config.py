import logging
import os
from logging.handlers import RotatingFileHandler
import time
import yaml


def setup_logging(log_dir="logs", script_name="script"):
    """
    Sets up a standardized logger for the project.

    This function creates a logger that writes to both the console and a
    rotating file. It ensures the log directory exists and creates a unique
    log file for each run based on the script name and timestamp.

    Args:
        log_dir (str): The directory to save log files in.
        script_name (str): The name of the script, used for the log filename.

    Returns:
        logging.Logger: A configured logger instance.
    """
    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Create a logger
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.DEBUG)  # Set the lowest level to capture all messages

    # Prevent duplicate handlers if called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create console handler with a specific format and level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Only show INFO and above on console
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Create file handler with a unique filename
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_filename = os.path.join(log_dir, f"{script_name}_{timestamp}.log")
    file_handler = RotatingFileHandler(
        log_filename, maxBytes=10 * 1024 * 1024, backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)  # Log everything to the file
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger

def log_config(config: dict, logger: logging.Logger):
    """
    Logs the entire project configuration for reproducibility.

    This function takes the configuration dictionary, formats it into a
    readable multi-line string using YAML syntax, and logs it. This
    ensures that every log file contains the exact parameters used for that run.

    Args:
        config (dict): The project's configuration dictionary.
        logger (logging.Logger): The logger instance to use.
    """
    # Use yaml.dump to create a nicely formatted string of the config
    config_str = yaml.dump(config, indent=4, default_flow_style=False)

    # Create a formatted message
    log_message = (
        "\n----------------- PROJECT CONFIGURATION -----------------\n"
        f"{config_str}"
        "-----------------------------------------------------------\n"
    )

    logger.info(log_message)
