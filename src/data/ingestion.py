
from pathlib import Path
import pandas as pd
import yaml


def load_config(config_path: str = "config.yaml") -> dict:

    path = Path(config_path)

    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Invalid configuration: expected a YAML mapping.")

    return config


def load_dataset(config_path: str = "config.yaml") -> pd.DataFrame:

    config = load_config(config_path)

    project_root = Path(config_path).resolve().parent
    data_config = config["data"]

    dataset_path = (
        project_root
        / data_config["raw_path"]
        / data_config["dataset_name"]
    )

    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    return pd.read_csv(
        dataset_path,
        engine="c"
    )
