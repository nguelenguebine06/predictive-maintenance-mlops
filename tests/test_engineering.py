
import numpy as np
import pandas as pd
import pytest

from src.features.engineering import engineer_features


SAMPLE_FEATURES = [
    "TP2",
    "Oil_temperature",
    "Motor_current",
    "TP3",
    "Reservoirs",
]

@pytest.fixture
def sample_data():
    """Dataset synthétique avec une interruption temporelle."""
    return pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2020-01-01 00:00:00",
            "2020-01-01 00:00:10",
            "2020-01-01 00:00:20",
            "2020-01-01 00:00:40",
            "2020-01-01 00:02:00",
            "2020-01-01 00:02:10",
            "2020-01-01 00:02:20",
        ]),
        "TP2": [1, 2, 3, 4, 10, 11, 12],
        "Oil_temperature": [50, 51, 52, 53, 60, 61, 62],
        "Motor_current": [4, 5, 6, 7, 8, 9, 10],
        "TP3": [8, 9, 10, 11, 12, 13, 14],
        "Reservoirs": [7, 8, 9, 10, 11, 12, 13],
    })


def test_differences_are_calculated_correctly(sample_data):
    result = engineer_features(
    sample_data,
    feature_columns=SAMPLE_FEATURES,)

    assert pd.isna(result.loc[0, "TP2_diff"])
    assert result.loc[1, "TP2_diff"] == 1
    assert result.loc[2, "TP2_diff"] == 1


def test_differences_reset_after_temporal_gap(sample_data):
    result = engineer_features(
    sample_data,
    feature_columns=SAMPLE_FEATURES,)

    # 00:02:00 follows a gap greater than 30 seconds.
    assert pd.isna(result.loc[4, "TP2_diff"])

    # The next measurement is consecutive.
    assert result.loc[5, "TP2_diff"] == 1


def test_rolling_mean_1min(sample_data):
    result = engineer_features(
    sample_data,
    feature_columns=SAMPLE_FEATURES, )

    # At 00:00:20: mean of 1, 2 and 3.
    assert result.loc[2, "TP2_rolling_mean_1min"] == pytest.approx(2.0)

    # At 00:00:40: mean of 1, 2, 3 and 4.
    assert result.loc[3, "TP2_rolling_mean_1min"] == pytest.approx(2.5)

    # Rolling window restarts after the temporal gap.
    assert pd.isna(result.loc[4, "TP2_rolling_mean_1min"])


def test_rolling_mean_5min_requires_five_observations(sample_data):
    result = engineer_features(
    sample_data,
    feature_columns=SAMPLE_FEATURES,)

    # First segment contains only four observations.
    assert result["TP2_rolling_mean_5min"].iloc[:4].isna().all()

    # Second segment contains only three observations.
    assert result["TP2_rolling_mean_5min"].iloc[4:].isna().all()


def test_interaction_features_are_correct(sample_data):
    result = engineer_features(
    sample_data,
    feature_columns=SAMPLE_FEATURES, )

    expected_gap_1 = sample_data["TP3"] - sample_data["Reservoirs"]
    expected_gap_2 = sample_data["TP2"] - sample_data["TP3"]

    pd.testing.assert_series_equal(
        result["pressure_gap_TP3_Reservoirs"],
        expected_gap_1,
        check_names=False,
    )

    pd.testing.assert_series_equal(
        result["pressure_gap_TP2_TP3"],
        expected_gap_2,
        check_names=False,
    )


def test_original_dataframe_is_not_modified(sample_data):
    original = sample_data.copy(deep=True)

    engineer_features(
        sample_data,
        feature_columns=SAMPLE_FEATURES,
    )

    pd.testing.assert_frame_equal(sample_data, original)


def test_engineered_features_contain_no_infinite_values(sample_data):
    result = engineer_features(
    sample_data,
    feature_columns=SAMPLE_FEATURES,)

    numeric_data = result.select_dtypes(include="number")

    assert not np.isinf(numeric_data.to_numpy()).any()


def test_invalid_timestamp_is_rejected(sample_data):
    sample_data["timestamp"] = sample_data["timestamp"].astype(object)
    sample_data.loc[0, "timestamp"] = "invalid-date"

    with pytest.raises(ValueError):
        engineer_features(
            sample_data,
            feature_columns=SAMPLE_FEATURES,)


def test_duplicate_timestamp_is_rejected(sample_data):
    sample_data.loc[1, "timestamp"] = sample_data.loc[0, "timestamp"]

    with pytest.raises(ValueError):
        engineer_features(
            sample_data,
            feature_columns=SAMPLE_FEATURES,
        )
