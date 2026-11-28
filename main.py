
import sys

from src.data.ingestion import load_config, load_dataset
from src.pipelines.training_pipeline import run_loeo_experiment


def main():

    print("=" * 60)
    print("PREDICTIVE MAINTENANCE - MLOPS")
    print("=" * 60)

    print("\nENVIRONNEMENT")
    print("Version Python :", sys.version)
    print("Interpréteur :", sys.executable)

    print("\nCHARGEMENT DE LA CONFIGURATION")

    config_path = "config.yaml"
    config = load_config(config_path)

    print("Projet :", config["projet"]["name"])
    print("Modèle :", config["model"]["name"])
    print("Stratégie :", config["validation"]["strategy"])

    print("\nCHARGEMENT DES DONNÉES BRUTES")

    raw_df = load_dataset(
        config_path=config_path,
        directory_key="raw_path",
        filename_key="dataset_name",
    )

    print("Dimensions :", raw_df.shape)

    print("\nCHARGEMENT DES DONNÉES LABELLISÉES")

    labeled_df = load_dataset(
        config_path=config_path,
        directory_key="processed_path",
        filename_key="processed_dataset_name",
    )

    print("Dimensions :", labeled_df.shape)

    print("\nVÉRIFICATION DES DONNÉES")

    if len(raw_df) != len(labeled_df):
        raise ValueError(
            "Les datasets brut et labellisé n'ont pas "
            "le même nombre de lignes."
        )

    print("Nombre de lignes cohérent :", len(raw_df))

    print("\nLANCEMENT DE L'EXPÉRIENCE")

    print("Fold externe sélectionné : événement 1")
    print("Entraînement externe : 3 événements")
    print("Test externe : événement 1")
    print("Sélection du seuil : validations internes")

    results = run_loeo_experiment(
        raw_df=raw_df,
        labeled_df=labeled_df,
        config=config,
        outer_event_ids=[4],
    )

    print("\n" + "=" * 60)
    print("RÉSULTATS DE L'EXPÉRIENCE")
    print("=" * 60)

    for result in results:

        print(f"\nÉvénement testé : {result['event_id']}")
        print(f"Seuil sélectionné : {result['threshold']:.4f}")
        print(f"F2 interne : {result['inner_f2']:.4f}")
        print(f"Échantillons train : {result['train_samples']}")
        print(f"Échantillons test : {result['test_samples']}")

        print("\nMétriques externes :")

        for metric, value in result["metrics"].items():

            if isinstance(value, (int, float)):
                print(f"{metric} : {value:.4f}")
            else:
                print(f"{metric} : {value}")

    print("\nExpérience terminée.")
    print("Consulte MLflow pour retrouver les résultats et le modèle.")


if __name__ == "__main__":
    main()
