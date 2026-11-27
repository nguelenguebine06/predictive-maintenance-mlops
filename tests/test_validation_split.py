import pandas as pd
from src.pipelines.validation import build_loeo_folds

def test_build_loeo_folds():
    events = [
        {"event_id": 1, "start": "2020-04-18 00:00:00", "end": "2020-04-18 23:59:00"},
        {"event_id": 2, "start": "2020-05-29 23:30:00", "end": "2020-05-30 06:00:00"},
        {"event_id": 3, "start": "2020-06-05 10:00:00", "end": "2020-06-07 14:30:00"},
        {"event_id": 4, "start": "2020-07-15 14:30:00", "end": "2020-07-15 19:00:00"},
    ]

    rows = []

    for event in events:
        start = pd.Timestamp(event["start"])
        end = pd.Timestamp(event["end"])

        rows.extend([
            {
                "timestamp": start - pd.Timedelta(hours=25),
                "event_id": None,
                "target_event": 0,
            },
            {
                "timestamp": start,
                "event_id": event["event_id"],
                "target_event": 1,
            },
            {
                "timestamp": end + pd.Timedelta(hours=25),
                "event_id": None,
                "target_event": 0,
            },
        ])

    df = pd.DataFrame(rows)

    config = {
        "data": {
            "timestamp_column": "timestamp",
            "event_id_column": "event_id",
            "target_column": "target_event",
        },
        "target": {"events": events},
        "validation": {
            "test_window": {
                "hours_before_event": 24,
                "hours_after_event": 24,
            }
        },
    }

    folds = build_loeo_folds(df, config)

    assert len(folds) == 4

    for fold in folds:
        train_mask = fold["train_mask"]
        test_mask = fold["test_mask"]

        assert train_mask.any()
        assert test_mask.any()

        assert len(train_mask) == len(df)
        assert len(test_mask) == len(df)

        # Pas de chevauchement entre entraînement et test
        assert not (train_mask & test_mask).any()

        # L'événement est présent dans le test
        assert (
            df.loc[test_mask, "event_id"] == fold["event_id"]
        ).any()

        # L'événement est absent de l'entraînement
        assert not (
            df.loc[train_mask, "event_id"] == fold["event_id"]
        ).any()