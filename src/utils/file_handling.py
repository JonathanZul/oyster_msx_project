import yaml


def load_config(config_path="config.yaml"):
    """
    Loads the project configuration from a YAML file.

    Args:
        config_path (str): The path to the config.yaml file.

    Returns:
        dict: A dictionary containing the configuration.
    """
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at '{config_path}'")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return None
