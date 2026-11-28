import numpy as np
import pandas as pd

from src.tracking.mlflow_tracking import (
    setup_mlflow,
    start_run,
    log_parameters,
    log_metrics,
    log_model,
)

from numbers import Real

from src.models.training import (
    build_training_pipeline,
    split_features_target,
    train_model,
)

from src.pipelines.validation import build_loeo_folds
from src.pipelines.features_pipeline import prepare_fold_data

from src.evaluation.threshold import select_threshold
from src.evaluation.metrics import evaluate_predictions


def get_positive_probabilities(model, X):
    """
    Retourne les probabilités associées à la classe positive (1).
    """

    classifier = model.named_steps["classifier"]
    classes = classifier.classes_

    positive_indices = np.where(classes == 1)[0]

    if len(positive_indices) != 1:
        raise ValueError("La classe positive 1 est introuvable.")

    return model.predict_proba(X)[:, positive_indices[0]]


def run_loeo_experiment(
    raw_df: pd.DataFrame,
    labeled_df: pd.DataFrame,
    config: dict,
    outer_event_ids: list[int] | None = None,
) -> list[dict]:
    """
    Exécute une validation imbriquée Leave-One-Event-Out.

    Pour chaque fold externe :
    - 3 validations internes ;
    - sélection du seuil sur les prédictions internes regroupées ;
    - réentraînement sur le train externe complet ;
    - évaluation sur le test externe.
    """

    if len(raw_df) != len(labeled_df):
        raise ValueError("Les datasets brut et labellisé ne sont pas alignés.")

    timestamp_col = config["data"]["timestamp_column"]

    if not np.array_equal(
        pd.to_datetime(raw_df[timestamp_col]).to_numpy(),
        pd.to_datetime(labeled_df[timestamp_col]).to_numpy(),
    ):
        raise ValueError("Les timestamps des datasets ne correspondent pas.")

    setup_mlflow(
        experiment_name=config["projet"]["name"]
    )

    outer_folds = build_loeo_folds(labeled_df, config)

    # Permet de limiter l'exécution à certains folds externes.
    if outer_event_ids is not None:
        requested_ids = set(outer_event_ids)

        available_ids = {
            fold["event_id"] for fold in outer_folds
        }

        unknown_ids = requested_ids - available_ids

        if unknown_ids:
            raise ValueError(
                f"Événements inconnus : {sorted(unknown_ids)}"
            )

        outer_folds = [
            fold for fold in outer_folds
            if fold["event_id"] in requested_ids
        ]

    results = []

    for outer_fold in outer_folds:

        outer_event_id = outer_fold["event_id"]

        with start_run(
                run_name=f"LOEO-fold-event-{outer_event_id}",
                tags={
                    "validation_strategy": "LOEO",
                    "outer_event_id": str(outer_event_id),
                    "model_type": config["model"]["name"],
                },
        ):
            print(f"\n===== FOLD EXTERNE {outer_event_id} =====")

            # --------------------------------------------------
            # 1. Préparation des données d'entraînement externe
            # --------------------------------------------------

            outer_train_raw = raw_df.loc[
                outer_fold["train_mask"]
            ].reset_index(drop=True)

            outer_train_labeled = labeled_df.loc[
                outer_fold["train_mask"]
            ].reset_index(drop=True)

            # Configuration interne : exclusion de l'événement externe.
            inner_config = {
                **config,
                "target": {
                    **config["target"],
                    "events": [
                        event
                        for event in config["target"]["events"]
                        if event["event_id"] != outer_event_id
                    ],
                },
            }

            inner_folds = build_loeo_folds(
                outer_train_labeled,
                inner_config,
            )

            inner_y_true = []
            inner_y_proba = []

            # --------------------------------------------------
            # 2. Validations internes
            # --------------------------------------------------
            inner_fold_details = []

            for inner_fold in inner_folds:
                inner_train, inner_valid = prepare_fold_data(
                    outer_train_raw,
                    outer_train_labeled,
                    inner_fold,
                    config,
                )

                X_train, y_train = split_features_target(
                    inner_train,
                    config,
                )

                X_valid, y_valid = split_features_target(
                    inner_valid,
                    config,
                )

                model = build_training_pipeline(config)

                train_model(
                    model,
                    X_train,
                    y_train,
                )

                probabilities = get_positive_probabilities(
                    model,
                    X_valid,
                )

                inner_fold_details.append({
                    "event_id": inner_fold["event_id"],
                    "samples": len(y_valid),
                    "positive_samples": int(y_valid.sum()),
                })

                inner_y_true.append(y_valid.to_numpy())
                inner_y_proba.append(probabilities)

                print(
                    f"Validation interne événement "
                    f"{inner_fold['event_id']} terminée."
                )

            # --------------------------------------------------
            # 3. Sélection du seuil sur les prédictions internes
            # --------------------------------------------------

            y_inner = np.concatenate(inner_y_true)
            proba_inner = np.concatenate(inner_y_proba)

            threshold_result = select_threshold(
                y_inner,
                proba_inner,
                config,
            )

            selected_threshold = threshold_result["threshold"]

            print(
                f"Seuil sélectionné : {selected_threshold:.2f}"
            )

            # --------------------------------------------------
            # 4. Préparation du train et test externes
            # --------------------------------------------------

            outer_train, outer_test = prepare_fold_data(
                raw_df,
                labeled_df,
                outer_fold,
                config,
            )

            X_train, y_train = split_features_target(
                outer_train,
                config,
            )

            X_test, y_test = split_features_target(
                outer_test,
                config,
            )

            # --------------------------------------------------
            # 5. Réentraînement sur le train externe complet
            # --------------------------------------------------

            final_model = build_training_pipeline(config)

            train_model(
                final_model,
                X_train,
                y_train,
            )

            # --------------------------------------------------
            # 6. Évaluation sur le test externe
            # --------------------------------------------------

            test_probabilities = get_positive_probabilities(
                final_model,
                X_test,
            )

            test_predictions = (
                    test_probabilities >= selected_threshold
            ).astype(int)

            metrics = evaluate_predictions(
                y_true=y_test,
                y_pred=test_predictions,
                y_proba=test_probabilities,
                beta=config["validation"]["threshold_selection"]["beta"],
            )

            # Paramètres du run
            mlflow_parameters = {
                "outer_event_id": outer_event_id,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "inner_folds": len(inner_folds),
                "inner_validation_samples": len(y_inner),
                "selected_threshold": selected_threshold,
                "inner_f2": threshold_result["f_beta"],
                "threshold_beta": config["validation"]["threshold_selection"]["beta"],
                "model_name": config["model"]["name"],
                "features": ",".join(config["features"]["baseline"]),
            }

            # Paramètres du modèle
            for name, value in config["model"]["parameters"].items():
                mlflow_parameters[f"model_{name}"] = (
                    str(value) if isinstance(value, (dict, list, tuple)) else value
                )

            # Informations sur les validations internes
            for index, detail in enumerate(inner_fold_details, start=1):
                mlflow_parameters[f"inner_{index}_event_id"] = detail["event_id"]
                mlflow_parameters[f"inner_{index}_samples"] = detail["samples"]
                mlflow_parameters[f"inner_{index}_positive_samples"] = detail["positive_samples"]

            log_parameters(mlflow_parameters)

            # Métriques internes et externes
            mlflow_metrics = get_numeric_metrics(metrics)
            mlflow_metrics["inner_f2"] = float(threshold_result["f_beta"])
            mlflow_metrics["selected_threshold"] = float(selected_threshold)

            log_metrics(mlflow_metrics)

            # Sauvegarde du modèle final du fold
            log_model(
                final_model,
                artifact_name="model",
            )

            fold_result = {
                "event_id": outer_event_id,
                "threshold": selected_threshold,
                "inner_f2": threshold_result["f_beta"],
                "metrics": metrics,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "inner_validation_samples": len(y_inner),
            }

            results.append(fold_result)

            print(f"F2 externe : {metrics['f2']:.4f}")
            print(f"Rappel externe : {metrics['recall']:.4f}")
            print(f"Précision externe : {metrics['precision']:.4f}")

    return results

def get_numeric_metrics(metrics: dict) -> dict:
    numeric_metrics = {}

    for name, value in metrics.items():
        if isinstance(value, (bool, np.bool_)):
            continue

        if isinstance(value, (Real, np.integer, np.floating)):
            if np.isfinite(value):
                numeric_metrics[name] = float(value)

    return numeric_metrics