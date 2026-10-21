
from data_loader import load_data
from train import train_models
from evaluate import evaluate_model

import mlflow
import pandas as pd
from pathlib import Path
import mlflow.sklearn
import os
import re

def main():

    x_train, x_test, y_train, y_test = load_data()
    trained_models, training_times = train_models(x_train, y_train)
    results = evaluate_model(trained_models, x_test, y_test)

    mlflow.set_tracking_uri(
        os.environ.get(
            "MLFLOW_TRACKING_URI",
            f"sqlite:///{Path('mlflow.db').resolve().as_posix()}"
        )
    )

    mlflow.set_experiment("Breast Cancer Classification")

    for model_name, model in trained_models.items():
        with mlflow.start_run(run_name=model_name):
            mlflow.set_tag("model_name", model_name)
            params = model.get_params()
            for param_name, value in params.items():
                if "__" in param_name:
                    if not hasattr(value, "get_params"):
                        mlflow.log_param(param_name, str(value))
            mlflow.log_metrics(results[model_name])
            mlflow.log_metric("training_time", training_times[model_name])
            safe_model_name = re.sub(
                r"[^a-zA-Z0-9_-]",
                "_",
                model_name
            )

            mlflow.sklearn.log_model(
                sk_model=model,
                name=safe_model_name,
                serialization_format="skops",
                skops_trusted_types=["sklearn.neural_network._stochastic_optimizers.AdamOptimizer"]
            )

            print(f"MLflow recorded: {model_name}")

    leaderboard = pd.DataFrame.from_dict(
        results, orient="index"
    )

    leaderboard.index.name = "Model"

    leaderboard = leaderboard.sort_values(
        by="f1", ascending=False
    )

    leaderboard.insert(
        0, "Rank", range(1, len(leaderboard) + 1)
    )

    print("\nMODEL LEADERBOARD")
    print(leaderboard.to_string())

    # Step 7: Save leaderboard
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    leaderboard.to_csv(
        output_dir / "leaderboard.csv"
    )

    print("\nLeaderboard saved successfully!")


if __name__ == "__main__":
    main()
