import pandas as pd

from src.features.engineering import engineer_features


def test_engineer_features_creates_expected_columns():
    df = pd.DataFrame({
        "timestamp": pd.date_range(
            "2020-01-01",
            periods=3,
            freq="10s"
        ),
        "TP2": [1.0, 2.0, 3.0],
        "Oil_temperature": [50.0, 51.0, 52.0],
        "Motor_current": [4.0, 5.0, 6.0],
        "TP3": [8.0, 8.1, 8.2],
        "Reservoirs": [7.0, 7.1, 7.2],
    })

    FEATURE_COLUMNS = [
        "TP2",
        "Oil_temperature",
        "Motor_current",
        "TP3",
        "Reservoirs",
    ]

    result = engineer_features(
        df,
        feature_columns=FEATURE_COLUMNS
    )

    expected_features = [
        "TP2_diff",
        "Oil_temperature_diff",
        "Motor_current_diff",
        "TP2_rolling_mean_1min",
        "TP2_rolling_mean_5min",
        "Oil_temperature_rolling_mean_1min",
        "Oil_temperature_rolling_mean_5min",
        "Motor_current_rolling_mean_1min",
        "Motor_current_rolling_mean_5min",
        "pressure_gap_TP3_Reservoirs",
        "pressure_gap_TP2_TP3",
    ]

    for feature in expected_features:
        assert feature in result.columns