
import pandas as pd


def build_loeo_folds(df: pd.DataFrame, config: dict) -> list[dict]:

    validation = config["validation"]
    data_config = config["data"]

    timestamp_col = data_config["timestamp_column"]
    event_id_col = data_config["event_id_column"]
    target_col = data_config["target_column"]

    required = {timestamp_col, event_id_col, target_col}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Colonnes manquantes : {sorted(missing)}")

    timestamps = pd.to_datetime(df[timestamp_col], errors="coerce")

    if timestamps.isna().any():
        raise ValueError("Timestamps invalides.")

    if timestamps.duplicated().any():
        raise ValueError("Timestamps dupliqués.")

    if not timestamps.is_monotonic_increasing:
        raise ValueError("Le dataset doit être trié chronologiquement.")

    before = pd.Timedelta(
        hours=validation["test_window"]["hours_before_event"]
    )
    after = pd.Timedelta(
        hours=validation["test_window"]["hours_after_event"]
    )

    folds = []

    for event in config["target"]["events"]:
        event_id = event["event_id"]
        start = pd.Timestamp(event["start"])
        end = pd.Timestamp(event["end"])

        test_start = start - before
        test_end = end + after

        test_mask = (
            timestamps.between(test_start, test_end, inclusive="both")
            .to_numpy()
        )

        train_mask = ~test_mask

        if not test_mask.any() or not train_mask.any():
            raise ValueError(f"Fold {event_id} vide.")

        if not (df.loc[test_mask, event_id_col] == event_id).any():
            raise ValueError(f"Événement {event_id} absent du test.")

        folds.append({
            "event_id": event_id,
            "train_mask": train_mask,
            "test_mask": test_mask,
            "test_start": test_start,
            "test_end": test_end,
        })

    return folds
