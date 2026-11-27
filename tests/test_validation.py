import pytest
import pandas as pd

from src.data.validation import validate_dataset


@pytest.fixture
def valid_config():
    return {
        "data": {
            "timestamp_column": "timestamp"
        },
        "features": {
            "numeric": ["TP2"],
            "categorical": ["COMP"]
        }
    }


@pytest.fixture
def valid_dataframe():
    return pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2020-02-01 00:00:00",
            "2020-02-01 00:00:10",
            "2020-02-01 00:00:20"
        ]),
        "TP2": [1.2, 1.3, 1.4],
        "COMP": [0, 1, 0]
    })


def test_valid_dataset(valid_dataframe, valid_config):
    report = validate_dataset(valid_dataframe, valid_config)

    assert report["valid"] is True
    assert report["errors"] == []


def test_missing_column(valid_dataframe, valid_config):
    df = valid_dataframe.drop(columns=["TP2"])

    report = validate_dataset(df, valid_config)

    assert report["valid"] is False


def test_missing_values(valid_dataframe, valid_config):
    df = valid_dataframe.copy()
    df.loc[0, "TP2"] = None

    report = validate_dataset(df, valid_config)

    assert report["valid"] is False


def test_infinite_values(valid_dataframe, valid_config):
    df = valid_dataframe.copy()
    df.loc[0, "TP2"] = float("inf")

    report = validate_dataset(df, valid_config)

    assert report["valid"] is False


def test_duplicate_rows_warning(valid_dataframe, valid_config):
    df = pd.concat(
        [valid_dataframe, valid_dataframe.iloc[[0]]],
        ignore_index=True
    )

    report = validate_dataset(df, valid_config)

    print("RAPPORT COMPLET :", report)
    print("WARNINGS :", report["warnings"])
    print("DOUBLONS :", df.duplicated().sum())
    assert report["valid"] is True
    assert any(
        warning["check"] == "Lignes_dupliquées"
        for warning in report["warnings"]
    )

    print(report)
