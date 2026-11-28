
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_training_pipeline(config: dict) -> Pipeline:

    model_config = config["model"]
    parameters = model_config["parameters"]
    preprocessing = model_config["preprocessing"]

    if preprocessing["scaling"] != "standard":
        raise ValueError("Scaling non supporté.")

    pipeline = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(**parameters)),
    ])

    return pipeline


def split_features_target(
    df: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Sépare les variables explicatives X et la cible y.
    """

    data_config = config["data"]
    feature_config = config["features"]

    target_column = data_config["target_column"]

    feature_columns = feature_config["baseline"]

    required_columns = [
        *feature_columns,
        target_column,
    ]

    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Colonnes manquantes : {sorted(missing_columns)}"
        )

    X = df[feature_columns].copy()
    y = df[target_column].copy()

    if y.isna().any():
        raise ValueError("La cible contient des valeurs manquantes.")

    if not y.isin([0, 1]).all():
        raise ValueError("La cible doit contenir uniquement 0 et 1.")

    if y.nunique() != 2:
        raise ValueError("La cible doit contenir les deux classes.")

    return X, y


def train_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """
    Entraîne le pipeline sur les données d'apprentissage.
    """

    if len(X_train) != len(y_train):
        raise ValueError("X_train et y_train ne sont pas alignés.")

    if X_train.empty:
        raise ValueError("Le jeu d'entraînement est vide.")

    if X_train.isna().all(axis=None):
        raise ValueError("Toutes les variables sont manquantes.")

    if y_train.nunique() != 2:
        raise ValueError("Les deux classes sont nécessaires.")

    if X_train.isna().any().any():
        raise ValueError(
            "Missing values detected in X_train. "
            "Imputation must be handled explicitly."
        )

    pipeline.fit(X_train, y_train)

    return pipeline
