
import numpy as np

from src.data.ingestion import load_config, load_dataset
from src.features.engineering import engineer_features


def test_feature_engineering_on_real_dataset():
    config = load_config()
    df = load_dataset()

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

    result = engineer_features(
        df,
        feature_columns=config["features"]["baseline"],
        timestamp_column=config["data"]["timestamp_column"],
    )

    # Dimensions
    assert len(result) == len(df)
    assert len(result.columns) == 27

    # Présence des 11 features
    assert all(col in result.columns for col in expected_features)

    # Exclusion des labels et identifiants
    assert "target_event" not in result.columns
    assert "event_id" not in result.columns
    assert "Unnamed: 0" not in result.columns

    # Intégrité temporelle
    assert result["timestamp"].is_monotonic_increasing
    assert not result["timestamp"].duplicated().any()

    # Absence de valeurs infinies
    numeric = result.select_dtypes(include="number")
    assert not np.isinf(numeric.to_numpy()).any()

    # Rapport des valeurs manquantes
    print("\nDimensions :", result.shape)
    print("\nValeurs manquantes (%) :")
    print((result[expected_features].isna().mean() * 100).round(4))

    print("\nValidation du Feature Engineering réussie.")
