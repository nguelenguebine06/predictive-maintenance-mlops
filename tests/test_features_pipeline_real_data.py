
from pathlib import Path
import gc
import pandas as pd

from src.data.ingestion import load_config
from src.pipelines.validation import build_loeo_folds
from src.pipelines.features_pipeline import prepare_fold_data


def test_features_pipeline_all_folds():
    config = load_config()
    root = Path("config.yaml").resolve().parent
    data_config = config["data"]

    # Colonnes nécessaires au pipeline
    feature_columns = config["features"]["baseline"]

    raw_columns = [
        data_config["timestamp_column"],
        *feature_columns,
    ]

    labeled_columns = [
        data_config["timestamp_column"],
        data_config["target_column"],
        data_config["event_id_column"],
    ]

    raw_df = pd.read_csv(
        root / data_config["raw_path"] / data_config["dataset_name"],
        usecols=raw_columns,
    )

    labeled_df = pd.read_csv(
        root / data_config["processed_path"] / data_config["processed_dataset_name"],
        usecols=labeled_columns,
    )

    folds = build_loeo_folds(labeled_df, config)

    expected_positives = {
        1: 8657,
        2: 2360,
        3: 17315,
        4: 1622,
    }

    for fold in folds:
        event_id = fold["event_id"]

        train, test = prepare_fold_data(
            raw_df, labeled_df, fold, config
        )

        assert len(train.columns) == 29
        assert len(test.columns) == 29

        assert int(test["target_event"].sum()) == expected_positives[event_id]
        assert not (train["event_id"] == event_id).any()
        assert (test["event_id"] == event_id).any()

        assert "target_event" in train.columns
        assert "event_id" in train.columns

        print(f"\nFold {event_id}")
        print("Train :", train.shape)
        print("Test :", test.shape)
        print("Positifs train :", int(train["target_event"].sum()))
        print("Positifs test :", int(test["target_event"].sum()))

        del train, test
        gc.collect()

    print("\nLes quatre folds ont été validés.")
