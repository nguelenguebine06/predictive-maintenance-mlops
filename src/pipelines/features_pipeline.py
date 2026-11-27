
import numpy as np
import pandas as pd

from src.features.engineering import engineer_features


def prepare_fold_data(raw_df, labeled_df, fold, config):
    data_config = config["data"]

    timestamp_col = data_config["timestamp_column"]
    target_col = data_config["target_column"]
    event_id_col = data_config["event_id_column"]
    feature_columns = config["features"]["baseline"]

    # Vérifier l'alignement des deux datasets.
    raw_timestamps = pd.to_datetime(raw_df[timestamp_col], errors="coerce")
    labeled_timestamps = pd.to_datetime(
        labeled_df[timestamp_col], errors="coerce"
    )

    if raw_timestamps.isna().any() or labeled_timestamps.isna().any():
        raise ValueError("Timestamps invalides.")

    if not np.array_equal(
        raw_timestamps.to_numpy(),
        labeled_timestamps.to_numpy(),
    ):
        raise ValueError("Les datasets brut et labellisé ne sont pas alignés.")

    if not raw_timestamps.is_monotonic_increasing:
        raise ValueError("Le dataset doit être trié chronologiquement.")

    train_mask = fold["train_mask"]
    test_mask = fold["test_mask"]

    if len(train_mask) != len(raw_df) or len(test_mask) != len(raw_df):
        raise ValueError("Dimensions des masques incompatibles.")

    if np.any(train_mask & test_mask):
        raise ValueError("Chevauchement entre train et test.")

    # Sélection des seules colonnes nécessaires.
    sensor_columns = [timestamp_col, *feature_columns]

    # Features train : calcul uniquement sur les observations train.
    train_raw = raw_df.loc[train_mask, sensor_columns]

    train = engineer_features(
        train_raw,
        feature_columns=feature_columns,
        timestamp_column=timestamp_col,
    )

    # Features test : historique causal de 5 minutes avant le test.
    context_start = fold["test_start"] - pd.Timedelta("5min")
    context_mask = (
        raw_timestamps.ge(context_start)
        & raw_timestamps.le(fold["test_end"])
    )

    test_raw = raw_df.loc[context_mask, sensor_columns]

    test = engineer_features(
        test_raw,
        feature_columns=feature_columns,
        timestamp_column=timestamp_col,
    )

    test = test.loc[
        test[timestamp_col].between(
            fold["test_start"],
            fold["test_end"],
            inclusive="both",
        )
    ].reset_index(drop=True)

    # Rattachement des labels après le Feature Engineering.
    labels = labeled_df[
        [timestamp_col, target_col, event_id_col]
    ].copy()

    labels[timestamp_col] = labeled_timestamps
    labels = labels.set_index(timestamp_col)

    for dataset in (train, test):
        aligned_labels = labels.reindex(
            pd.DatetimeIndex(dataset[timestamp_col])
        )

        if aligned_labels[target_col].isna().any():
            raise ValueError("Labels manquants après rattachement.")

        dataset[target_col] = aligned_labels[target_col].to_numpy()
        dataset[event_id_col] = aligned_labels[event_id_col].to_numpy()

    if (train[event_id_col] == fold["event_id"]).any():
        raise ValueError("L'événement testé est présent dans le train.")

    if not (test[event_id_col] == fold["event_id"]).any():
        raise ValueError("Événement absent du dataset test.")

    return train, test
