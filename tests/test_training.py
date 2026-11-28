
import numpy as np
import pandas as pd
import pytest

from src.models.training import (
    build_training_pipeline,
    split_features_target,
    train_model,
)


@pytest.fixture
def config():
    return {
        "data": {
            "target_column": "target_event",
        },
        "features": {
            "baseline": ["TP2", "TP3", "Motor_current"],
        },
        "model": {
            "parameters": {
                "class_weight": "balanced",
                "max_iter": 2000,
                "solver": "lbfgs",
                "random_state": 42,
            },
            "preprocessing": {
                "scaling": "standard",
            },
        },
    }


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(42)
    n_samples = 100

    df = pd.DataFrame({
        "TP2": rng.normal(5, 2, n_samples),
        "TP3": rng.normal(8, 1, n_samples),
        "Motor_current": rng.normal(4, 1, n_samples),
        "timestamp": pd.date_range(
            "2020-01-01",
            periods=n_samples,
            freq="10s",
        ),
        "event_id": [None] * n_samples,
        "target_event": [0] * 80 + [1] * 20,
    })

    return df


def test_build_training_pipeline(config):
    pipeline = build_training_pipeline(config)

    assert pipeline is not None
    assert list(pipeline.named_steps.keys()) == [
        "scaler",
        "classifier",
    ]

    assert pipeline.named_steps["scaler"] is not None
    assert pipeline.named_steps["classifier"] is not None

    assert "imputer" not in pipeline.named_steps


def test_split_features_target(config, sample_data):
    X, y = split_features_target(sample_data, config)

    assert list(X.columns) == ["TP2", "TP3", "Motor_current"]
    assert "timestamp" not in X.columns
    assert "event_id" not in X.columns
    assert "target_event" not in X.columns

    assert len(X) == len(y)
    assert set(y.unique()) == {0, 1}


def test_missing_feature_is_rejected(config, sample_data):
    sample_data = sample_data.drop(columns=["TP2"])

    with pytest.raises(ValueError, match="Colonnes manquantes"):
        split_features_target(sample_data, config)


def test_missing_target_values_are_rejected(config, sample_data):
    sample_data.loc[0, "target_event"] = np.nan

    with pytest.raises(ValueError, match="valeurs manquantes"):
        split_features_target(sample_data, config)


def test_invalid_target_values_are_rejected(config, sample_data):
    sample_data.loc[0, "target_event"] = 2

    with pytest.raises(ValueError, match="uniquement 0 et 1"):
        split_features_target(sample_data, config)


def test_single_class_target_is_rejected(config, sample_data):
    sample_data["target_event"] = 0

    with pytest.raises(ValueError, match="deux classes"):
        split_features_target(sample_data, config)


def test_unsupported_scaling_is_rejected(config):
    config["model"]["preprocessing"]["scaling"] = "minmax"

    with pytest.raises(ValueError, match="Scaling non supporté"):
        build_training_pipeline(config)


def test_train_model(sample_data, config):
    X, y = split_features_target(sample_data, config)
    pipeline = build_training_pipeline(config)

    trained_model = train_model(pipeline, X, y)

    assert trained_model is pipeline
    assert hasattr(trained_model, "predict")
    assert hasattr(trained_model, "predict_proba")


def test_train_model_rejects_missing_values(sample_data, config):
    X, y = split_features_target(sample_data, config)

    X.loc[0, "TP2"] = np.nan
    X.loc[1, "Motor_current"] = np.nan

    pipeline = build_training_pipeline(config)

    with pytest.raises(ValueError, match="Missing values"):
        train_model(pipeline, X, y)


def test_train_model_rejects_empty_data(config):
    X = pd.DataFrame(columns=["TP2", "TP3", "Motor_current"])
    y = pd.Series(dtype=int)

    pipeline = build_training_pipeline(config)

    with pytest.raises(ValueError, match="vide"):
        train_model(pipeline, X, y)


def test_train_model_rejects_misaligned_data(config):
    X = pd.DataFrame({
        "TP2": [1, 2, 3],
        "TP3": [4, 5, 6],
        "Motor_current": [2, 3, 4],
    })
    y = pd.Series([0, 1])

    pipeline = build_training_pipeline(config)

    with pytest.raises(ValueError, match="pas alignés"):
        train_model(pipeline, X, y)
