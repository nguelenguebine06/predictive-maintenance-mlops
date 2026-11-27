
import pandas as pd
import numpy as np

from src.pipelines.features_pipeline import prepare_fold_data


def test_prepare_fold_data():
    timestamps = pd.date_range(
        "2020-01-01", periods=100, freq="10s"
    )

    raw = pd.DataFrame({
        "timestamp": timestamps,
        "TP2": range(100),
        "TP3": range(100),
        "H1": range(100),
        "DV_pressure": range(100),
        "Reservoirs": range(100),
        "Oil_temperature": range(100),
        "Motor_current": range(100),
        "COMP": [0] * 100,
        "DV_eletric": [0] * 100,
        "Towers": [0] * 100,
        "MPG": [0] * 100,
        "LPS": [0] * 100,
        "Pressure_switch": [0] * 100,
        "Oil_level": [0] * 100,
        "Caudal_impulses": range(100),
    })

    labeled = raw[["timestamp"]].copy()
    labeled["target_event"] = 0
    labeled["event_id"] = None

    labeled.loc[50, "target_event"] = 1
    labeled.loc[50, "event_id"] = 1

    fold = {
        "event_id": 1,
        "train_mask": np.array(
            [i < 20 or i > 80 for i in range(100)]
        ),
        "test_mask": np.array(
            [20 <= i <= 80 for i in range(100)]
        ),
        "test_start": timestamps[20],
        "test_end": timestamps[80],
    }

    config = {
        "data": {
            "timestamp_column": "timestamp",
            "target_column": "target_event",
            "event_id_column": "event_id",
        },
        "features": {
            "baseline": [
                "TP2", "TP3", "H1", "DV_pressure", "Reservoirs",
                "Oil_temperature", "Motor_current", "COMP",
                "DV_eletric", "Towers", "MPG", "LPS",
                "Pressure_switch", "Oil_level", "Caudal_impulses",
            ]
        },
    }

    train, test = prepare_fold_data(raw, labeled, fold, config)

    assert len(train) == 39
    assert len(test) == 61

    assert "target_event" in train.columns
    assert "event_id" in train.columns

    assert "target_event" in test.columns
    assert "event_id" in test.columns

    assert "target_event" not in raw.columns
    assert "event_id" not in raw.columns

    assert not (train["event_id"] == 1).any()
    assert (test["event_id"] == 1).any()
