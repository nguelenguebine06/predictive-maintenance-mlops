
import pandas as pd
import pytest
from src.data.ingestion import load_config, load_dataset


def test_load_dataset_returns_dataframe():
    df = load_dataset()

    assert isinstance(df, pd.DataFrame)


def test_load_dataset_has_expected_shape():
    df = load_dataset()

    assert df.shape == (1_516_948, 17)


def test_load_config_raises_error_for_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config("config_inexistant.yaml")
