import numpy as np
import pandas as pd

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

    outer_folds = build_loeo_folds(labeled_df, config)

    results = []

    for outer_fold in outer_folds:

        outer_event_id = outer_fold["event_id"]

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