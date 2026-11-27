import numpy as np

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_predictions(
    y_true,
    y_pred,
    y_proba,
    beta: float = 2.0,
) -> dict:
    """
    Calcule les métriques de classification binaire.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba)

    if not (len(y_true) == len(y_pred) == len(y_proba)):
        raise ValueError("Les dimensions des données ne correspondent pas.")

    if len(y_true) == 0:
        raise ValueError("Les données d'évaluation sont vides.")

    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("y_true doit contenir uniquement 0 et 1.")

    if not np.isin(y_pred, [0, 1]).all():
        raise ValueError("y_pred doit contenir uniquement 0 et 1.")

    if not np.isfinite(y_proba).all():
        raise ValueError("Les probabilités contiennent des valeurs invalides.")

    if not np.isin(y_true, [0, 1]).all() or len(np.unique(y_true)) != 2:
        raise ValueError("Les deux classes doivent être présentes dans y_true.")

    if not 0 < beta:
        raise ValueError("beta doit être strictement positif.")

    if ((y_proba < 0) | (y_proba > 1)).any():
        raise ValueError("Les probabilités doivent être comprises entre 0 et 1.")

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=beta, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "average_precision": float(average_precision_score(y_true, y_proba)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["normal", "event"],
            output_dict=True,
            zero_division=0,
        ),
    }