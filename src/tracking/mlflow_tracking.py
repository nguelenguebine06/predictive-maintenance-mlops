
from pathlib import Path
from numbers import Real

import mlflow
import mlflow.sklearn


DEFAULT_EXPERIMENT_NAME = "predictive-maintenance-mlops"


def setup_mlflow(
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    tracking_uri: str | None = None,
    artifact_location: str | None = None,
) -> str:
    """
    Configure MLflow et retourne l'identifiant de l'expérience.
    """

    if tracking_uri is not None:
        mlflow.set_tracking_uri(tracking_uri)

    if not experiment_name.strip():
        raise ValueError("experiment_name cannot be empty.")

    if artifact_location is None:
        artifact_path = Path("mlflow_artifacts").resolve()
        artifact_path.mkdir(parents=True, exist_ok=True)
        artifact_location = artifact_path.as_uri()

    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        experiment_id = mlflow.create_experiment(
            name=experiment_name,
            artifact_location=artifact_location,
        )
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_name)

    return experiment_id


def start_run(
    run_name: str | None = None,
    tags: dict | None = None,
):
    """
    Démarre une nouvelle expérience MLflow.
    """

    if mlflow.active_run() is not None:
        raise RuntimeError(
            "An MLflow run is already active."
        )

    return mlflow.start_run(
        run_name=run_name,
        tags=tags,
    )


def log_parameters(parameters: dict) -> None:
    """
    Enregistre les paramètres d'entraînement.
    """

    if not isinstance(parameters, dict):
        raise TypeError("parameters must be a dictionary.")

    mlflow.log_params(parameters)


def log_metrics(metrics: dict, step: int | None = None) -> None:
    """
    Enregistre les métriques d'évaluation.
    """

    if not isinstance(metrics, dict):
        raise TypeError("metrics must be a dictionary.")

    for name, value in metrics.items():
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(
                f"Metric '{name}' must be numeric."
            )

        if not float("-inf") < float(value) < float("inf"):
            raise ValueError(
                f"Metric '{name}' must be finite."
            )

    mlflow.log_metrics(metrics, step=step)


def log_model(
    model,
    artifact_name: str = "model",
):
    """
    Enregistre un modèle Scikit-learn dans MLflow.
    """

    if not artifact_name.strip():
        raise ValueError("artifact_name cannot be empty.")

    return mlflow.sklearn.log_model(
        sk_model=model,
        name=artifact_name,
    )


def log_artifact(file_path: str | Path) -> None:
    """
    Enregistre un fichier comme artefact MLflow.
    """

    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Artifact not found: {path}"
        )

    mlflow.log_artifact(str(path))
