import pytest
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

from sklearn.linear_model import LogisticRegression

from src.tracking.mlflow_tracking import (
    setup_mlflow,
    start_run,
    log_parameters,
    log_metrics,
    log_model,
    log_artifact,
)


@pytest.fixture
def mlflow_environment(tmp_path):
    tracking_uri = f"sqlite:///{tmp_path / 'test_mlflow.db'}"
    artifact_location = (tmp_path / "artifacts").as_uri()

    mlflow.set_tracking_uri(tracking_uri)

    setup_mlflow(
        experiment_name="test-predictive-maintenance",
        artifact_location=artifact_location,
    )

    yield

    if mlflow.active_run() is not None:
        mlflow.end_run()


def test_setup_mlflow(mlflow_environment):
    experiment = mlflow.get_experiment_by_name(
        "test-predictive-maintenance"
    )

    assert experiment is not None
    assert experiment.experiment_id is not None


def test_start_run(mlflow_environment):
    with start_run(run_name="test-run") as run:
        assert run.info.run_id is not None

    assert mlflow.active_run() is None


def test_log_parameters(mlflow_environment):
    with start_run():
        log_parameters({
            "model": "LogisticRegression",
            "max_iter": 2000,
        })

        run_id = mlflow.active_run().info.run_id

    client = MlflowClient()
    run = client.get_run(run_id)

    assert run.data.params["model"] == "LogisticRegression"
    assert run.data.params["max_iter"] == "2000"


def test_log_metrics(mlflow_environment):
    with start_run():
        log_metrics({
            "precision": 0.90,
            "recall": 0.95,
            "f2": 0.94,
        })

        run_id = mlflow.active_run().info.run_id

    client = MlflowClient()
    run = client.get_run(run_id)

    assert run.data.metrics["precision"] == 0.90
    assert run.data.metrics["recall"] == 0.95
    assert run.data.metrics["f2"] == 0.94

def test_log_metrics_rejects_invalid_values(mlflow_environment):
    with start_run():
        with pytest.raises(ValueError):
            log_metrics({"precision": float("nan")})


def test_log_model(mlflow_environment):
    X = pd.DataFrame({
        "feature_1": [0, 1, 2, 3],
        "feature_2": [1, 0, 1, 0],
    })

    y = [0, 0, 1, 1]

    model = LogisticRegression()
    model.fit(X, y)

    with start_run():
        model_info = log_model(model)
        run_id = mlflow.active_run().info.run_id

    assert model_info.model_uri is not None
    assert model_info.model_id is not None

    loaded_model = mlflow.sklearn.load_model(
        model_info.model_uri
    )

    predictions = loaded_model.predict(X)

    assert len(predictions) == len(y)


def test_log_artifact(mlflow_environment, tmp_path):
    report = tmp_path / "report.txt"
    report.write_text("Test report", encoding="utf-8")

    with start_run():
        log_artifact(report)

        run_id = mlflow.active_run().info.run_id

    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(run_id)

    assert any(
        artifact.path == "report.txt"
        for artifact in artifacts
    )


def test_log_artifact_rejects_missing_file(mlflow_environment):
    with start_run():
        with pytest.raises(FileNotFoundError):
            log_artifact("missing_report.txt")
