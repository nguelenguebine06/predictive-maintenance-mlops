
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




def load_dataset(
    config_path: str = "config.yaml",
    directory_key: str = "raw_path",
    filename_key: str = "dataset_name",
) -> pd.DataFrame:

    config = load_config(config_path)

    project_root = Path(config_path).resolve().parent
    data_config = config["data"]

    dataset_path = (
        project_root
        / data_config[directory_key]
        / data_config[filename_key]
    )

    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    return pd.read_csv(
        dataset_path,
        engine="c"
    )
