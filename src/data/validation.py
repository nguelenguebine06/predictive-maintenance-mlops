
from typing import Any

import numpy as np
import pandas as pd

#test la validité du dataset
def validate_dataset(
    df: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:

    errors = []
    warnings = []

    data_config = config["data"]
    features_config = config["features"]

    timestamp_column = data_config["timestamp_column"]

    expected_columns = (
        [timestamp_column]
        + features_config["numeric"]
        + features_config["categorical"]
    )

    #  Colonnes manquantes
    missing_columns = [
        column
        for column in expected_columns
        if column not in df.columns
    ]

    if missing_columns:
        errors.append({
            "check": "required_columns",
            "message": f"Colonnes manquantes: {missing_columns}",
        })
    # Colonne disponible
    available_columns = [
        column
        for column in expected_columns
        if column in df.columns
    ]

    if df.empty:
        errors.append({
            "check": "dataset vide",
            "message": "Le dataset contains ne contient pas de lignes.",
        })

    # 2. recherche de valeurs manquantes
    null_counts = df[available_columns].isna().sum()
    columns_with_nulls = null_counts[null_counts > 0].to_dict()

    if columns_with_nulls:
        errors.append({
            "check": "Valeurs manquantes",
            "message": "Valeurs manquantes détectés.",
            "details": columns_with_nulls,
        })

    # Vérification des valeurs infinie dans les features numérics
    numeric_columns = [
        column
        for column in features_config["numeric"]
        if column in df.columns
    ]

    infinite_counts = {}

    for column in numeric_columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            count = int(np.isinf(df[column].dropna()).sum())

            if count > 0:
                infinite_counts[column] = count

    if infinite_counts:
        errors.append({
            "check": "Valeurs infinis",
            "message": "Valeurs infinis détectés.",
            "details": infinite_counts,
        })

    # Vérification de la colonne timestamp
    timestamp_stats = {}

    if timestamp_column in df.columns:
        timestamps = pd.to_datetime(
            df[timestamp_column],
            errors="coerce",
        )

        invalid_timestamps = int(timestamps.isna().sum())

        if invalid_timestamps > 0:
            errors.append({
                "check": "timestamps invalide",
                "message": (
                    f"{invalid_timestamps} timestamps invalide detecté."
                ),
            })
        else:
            timestamp_stats = {
                "start": str(timestamps.min()),
                "end": str(timestamps.max()),
                # Vérifier si les dates sont par ordres croissants
                "is_monotonic": bool(timestamps.is_monotonic_increasing),
                "duplicate_count": int(timestamps.duplicated().sum()),
            }

            if not timestamps.is_monotonic_increasing:
                warnings.append({
                    "check": "Ordre timestamp",
                    "message": "Timestamps n'est pas chronologiquement ordonné.",
                })

            if timestamp_stats["duplicate_count"] > 0:
                warnings.append({
                    "check": "duplicate_timestamps",
                    "message": (
                        f"{timestamp_stats['duplicate_count']} "
                        "duplicate timestamps detected."
                    ),
                })

            time_differences = timestamps.diff().dt.total_seconds()

            positive_differences = time_differences[
                time_differences > 0
            ]

            if not positive_differences.empty:
                timestamp_stats["median_interval_seconds"] = float(
                    positive_differences.median())
                timestamp_stats["maximum_interval_seconds"] = float(
                    positive_differences.max())

    # Les lignes dupliquées
    duplicate_rows = int(df.duplicated().sum())

    if duplicate_rows > 0:
        warnings.append({
            "check": "Lignes_dupliquées",
            "message": f"{duplicate_rows} ligne dupliquées détectées.",
        })

    return {
        "valid": len(errors) == 0,
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "errors": errors,
        "warnings": warnings,
        "timestamp_statistics": timestamp_stats,
    }
