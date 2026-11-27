
from pathlib import Path
import pandas as pd

from src.data.ingestion import load_config
from src.pipelines.validation import build_loeo_folds


def test_loeo_validation_on_real_dataset():
    config = load_config()
    data_config = config["data"]

    dataset_path = (
        Path("config.yaml").resolve().parent
        / data_config["processed_path"]
        / data_config["processed_dataset_name"]
    )

    df = pd.read_csv(dataset_path)
    folds = build_loeo_folds(df, config)

    assert len(folds) == 4

    for fold in folds:
        train_mask = fold["train_mask"]
        test_mask = fold["test_mask"]
        event_id = fold["event_id"]

        assert train_mask.sum() > 0
        assert test_mask.sum() > 0
        assert not (train_mask & test_mask).any()
        assert (train_mask | test_mask).all()

        assert (df.loc[test_mask, "event_id"] == event_id).any()
        assert not (df.loc[train_mask, "event_id"] == event_id).any()

        assert df.loc[test_mask, "target_event"].nunique() == 2
        assert df.loc[train_mask, "target_event"].nunique() == 2

        print(f"\nFold {event_id}")
        print(f"Train : {train_mask.sum():,}")
        print(f"Test : {test_mask.sum():,}")
        print(
            "Positifs test :",
            int(df.loc[test_mask, "target_event"].sum())
        )

    print("\nValidation LOEO réelle réussie.")
