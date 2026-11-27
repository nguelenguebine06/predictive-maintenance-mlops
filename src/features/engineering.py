
from collections.abc import Sequence
import pandas as pd


MAX_GAP_SECONDS = 30

ROLLING_WINDOWS = {
    "1min": (pd.Timedelta("1min"), 3),
    "5min": (pd.Timedelta("5min"), 5),
}

DIFFERENCE_FEATURES = [
    "TP2",
    "Oil_temperature",
    "Motor_current",
]

ROLLING_FEATURES = [
    "TP2",
    "Oil_temperature",
    "Motor_current",
]

REQUIRED_ENGINEERING_COLUMNS = [
    "TP2",
    "Oil_temperature",
    "Motor_current",
    "TP3",
    "Reservoirs",
]


def engineer_features(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    timestamp_column: str = "timestamp",
    max_gap_seconds: int = MAX_GAP_SECONDS,
) -> pd.DataFrame:
    """
    Generate causal features from sensor measurements only.

    Labels, event identifiers and unrelated columns are excluded
    from the returned DataFrame.
    """

    if max_gap_seconds <= 0:
        raise ValueError("max_gap_seconds doit être supérieur à zéro.")

    if not feature_columns:
        raise ValueError("La liste feature_columns ne peut pas être vide.")

    feature_columns = list(feature_columns)

    if timestamp_column in feature_columns:
        raise ValueError(
            "La colonne timestamp ne doit pas être une variable explicative."
        )

    if len(feature_columns) != len(set(feature_columns)):
        raise ValueError("La liste feature_columns contient des doublons.")

    required_columns = {timestamp_column, *feature_columns}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Colonnes manquantes : {sorted(missing_columns)}"
        )

    missing_engineering_columns = (
        set(REQUIRED_ENGINEERING_COLUMNS) - set(feature_columns)
    )

    if missing_engineering_columns:
        raise ValueError(
            "Variables nécessaires au Feature Engineering absentes : "
            f"{sorted(missing_engineering_columns)}"
        )

    # Sélection explicite : aucun label, event_id ou index exporté.
    result = df.loc[:, [timestamp_column, *feature_columns]].copy()

    timestamps = pd.to_datetime(
        result[timestamp_column],
        errors="coerce",
    )

    if timestamps.isna().any():
        raise ValueError("Des timestamps invalides ont été détectés.")

    if timestamps.duplicated().any():
        raise ValueError("Des timestamps dupliqués ont été détectés.")

    result[timestamp_column] = timestamps

    result = (
        result.sort_values(timestamp_column, kind="stable")
        .reset_index(drop=True)
    )

    time_index = pd.DatetimeIndex(result[timestamp_column])

    time_delta_seconds = (
        time_index.to_series()
        .diff()
        .dt.total_seconds()
        .reset_index(drop=True)
    )

    new_segment = (
        time_delta_seconds.isna()
        | (time_delta_seconds > max_gap_seconds)
    )

    segment_id = new_segment.cumsum()

    valid_consecutive = (
        time_delta_seconds.gt(0)
        & time_delta_seconds.le(max_gap_seconds)
    )

    # Différences entre observations consécutives.
    for column in DIFFERENCE_FEATURES:
        result[f"{column}_diff"] = (
            result[column]
            .diff()
            .where(valid_consecutive)
        )

    # Moyennes glissantes causales, réinitialisées après chaque rupture.
    indexed_data = result.set_index(timestamp_column)

    for column in ROLLING_FEATURES:
        for window_name, (window_size, min_periods) in ROLLING_WINDOWS.items():

            feature_name = f"{column}_rolling_mean_{window_name}"

            rolling_result = (
                indexed_data[column]
                .groupby(segment_id.to_numpy(), sort=False)
                .rolling(
                    window=window_size,
                    min_periods=min_periods,
                )
                .mean()
            )

            rolling_result.index = rolling_result.index.get_level_values(-1)

            result[feature_name] = (
                rolling_result.reindex(time_index).to_numpy()
            )

    # Variables d'interaction.
    result["pressure_gap_TP3_Reservoirs"] = (
        result["TP3"] - result["Reservoirs"]
    )

    result["pressure_gap_TP2_TP3"] = (
        result["TP2"] - result["TP3"]
    )

    return result
