import numpy as np

from sklearn.metrics import fbeta_score


def select_threshold(
    y_true,
    y_proba,
    config: dict,
) -> dict:
    """
    Sélectionne le seuil maximisant le F-beta.
    En cas d'égalité, retient le seuil le plus élevé.
    """

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    if len(y_true) != len(y_proba):
        raise ValueError("y_true et y_proba ne sont pas alignés.")

    if len(y_true) == 0:
        raise ValueError("Les données de validation sont vides.")

    if len(np.unique(y_true)) != 2:
        raise ValueError("Les deux classes doivent être présentes.")

    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("y_true doit contenir uniquement 0 et 1.")

    if not np.isfinite(y_proba).all():
        raise ValueError("Les probabilités contiennent des valeurs invalides.")

    if ((y_proba < 0) | (y_proba > 1)).any():
        raise ValueError("Les probabilités doivent être comprises entre 0 et 1.")

    threshold_config = config["validation"]["threshold_selection"]

    if threshold_config["tie_break"] != "higher_threshold":
        raise ValueError("Stratégie de départage non supportée.")

    beta = float(threshold_config["beta"])
    search = threshold_config["search"]

    minimum = float(search["start"])
    maximum = float(search["stop"])
    step = float(search["step"])

    if beta <= 0 or step <= 0 or minimum > maximum:
        raise ValueError("Configuration de recherche du seuil invalide.")

    thresholds = np.arange(
        minimum,
        maximum + step / 2,
        step,
    )

    results = []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)

        score = fbeta_score(
            y_true,
            y_pred,
            beta=beta,
            zero_division=0,
        )

        results.append({
            "threshold": float(threshold),
            "f_beta": float(score),
        })

    best_score = max(item["f_beta"] for item in results)

    best_candidates = [
        item for item in results
        if np.isclose(item["f_beta"], best_score)
    ]

    best = max(
        best_candidates,
        key=lambda item: item["threshold"],
    )

    return {
        "threshold": best["threshold"],
        "f_beta": best["f_beta"],
        "beta": beta,
        "search_results": results,
    }