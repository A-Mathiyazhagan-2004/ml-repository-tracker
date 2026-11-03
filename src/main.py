
from data_loader import load_data
from train import train_models
from evaluate import evaluate_model

import mlflow
import argparse
import pandas as pd
from pathlib import Path
import mlflow.sklearn
import os
import re

def main():

    parser = argparse.ArgumentParser(description="Automated ML Experiment Tracker")
    
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to the dataset CSV file"
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Name of the target column in the dataset"
    )

    args = parser.parse_args()
    data = args.data
    target_column = args.target

    if args.data is None and args.target is None:
        dataset_name = "breast_cancer"
        experiment_name = "Breast Cancer Classification"

    elif args.data is not None and args.target is not None:
        dataset_name = re.sub(
            r"[^a-zA-Z0-9_-]",
            "_",
            Path(data).stem
        )
        experiment_name = f"{dataset_name} Classification"

    else:
        parser.error("--data and --target must be provided together.")

    x_train, x_test, y_train, y_test = load_data(data, target_column)
    trained_models, training_times = train_models(x_train, y_train)
    results = evaluate_model(trained_models, x_test, y_test, training_times)

    mlflow.set_tracking_uri(
        os.environ.get(
            "MLFLOW_TRACKING_URI",
            f"sqlite:///{Path('mlflow.db').resolve().as_posix()}"
        )
    )

    mlflow.set_experiment(experiment_name)

    for model_name, model in trained_models.items():

        with mlflow.start_run(run_name=model_name):
            
            mlflow.set_tag("model_name", model_name)
            mlflow.set_tag("dataset_name", dataset_name)
            for param_name, value in model.get_params(deep=True).items():
                if not hasattr(value, "fit") and not hasattr(value, "get_params"):
                    mlflow.log_param(param_name, str(value))
            mlflow.log_metrics(results[model_name])
            safe_model_name = re.sub(
                r"[^a-zA-Z0-9_-]",
                "_",
                model_name
            )

            
            mlflow.sklearn.log_model(
                sk_model=model,
                name=safe_model_name,
                serialization_format="skops",
                skops_trusted_types=[
                    "sklearn.neural_network._stochastic_optimizers.AdamOptimizer",
                    "sklearn.tree._tree.Tree"
                ],
                pip_requirements=[
                    f"scikit-learn=={__import__('sklearn').__version__}",
                    f"numpy=={__import__('numpy').__version__}",
                    "scipy",
                    "pandas",
                    "skops",
                ]
            )


            print(f"MLflow recorded: {model_name}")

    
    leaderboard = pd.DataFrame.from_dict(
        results, orient="index"
    )

    leaderboard.index.name = "Model"

    leaderboard["accuracy_rank"] = leaderboard["accuracy"].rank(
        ascending=False, method="average"
    )

    leaderboard["precision_rank"] = leaderboard["precision"].rank(
        ascending=False, method="average"
    )

    leaderboard["recall_rank"] = leaderboard["recall"].rank(
    ascending=False, method="average"
    )

    leaderboard["f1_rank"] = leaderboard["f1"].rank(
        ascending=False, method="average"
    )


    leaderboard["training_time_rank"] = leaderboard["training time"].rank(
        ascending=True, method="average"
    )

    rank_columns = [
        "accuracy_rank",
        "precision_rank",
        "recall_rank",
        "f1_rank",
        "training_time_rank"
    ]

    leaderboard["total_rank"] = leaderboard[rank_columns].sum(axis=1)

    leaderboard["overall_rank"] = leaderboard["total_rank"].rank(
        ascending=True, method="min"
    )

    leaderboard = leaderboard.sort_values(
        by=["total_rank", "f1"],
        ascending=[True, False]
    )

    print("\nMULTI-METRIC MODEL LEADERBOARD")
    print(leaderboard.to_string())

    best_model = leaderboard.index[0]

    print("\nBEST OVERALL MODEL")
    print(f"Model: {best_model}")
    print(f"Total Rank: {leaderboard.iloc[0]['total_rank']:.2f}")

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    if args.data is None:
        output_path = output_dir / "leaderboard.csv"
    else:
        output_path = output_dir / f"{dataset_name}_leaderboard.csv"

    leaderboard.to_csv(output_path)

    print(f"\nLeaderboard saved: {output_path}")

if __name__ == "__main__":
    main()
