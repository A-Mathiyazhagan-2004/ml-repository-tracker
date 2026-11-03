
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

def evaluate_model(trained_models, x_test, y_test, training_times):

    results = {}

    for model_name, model in trained_models.items():

        y_pred = model.predict(x_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "training time": training_times[model_name]
        }

        results[model_name] = metrics

        print(f"\nModel: {model_name}")

        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")

    return results
