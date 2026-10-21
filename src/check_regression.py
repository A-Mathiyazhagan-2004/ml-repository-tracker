
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

BASELINE_PATH = ROOT / "baselines" / "baseline.json"
LEADERBOARD_PATH = ROOT / "results" / "leaderboard.csv"

TOLERANCE = 0.02


def check_regression():

    with open(BASELINE_PATH) as file:
        baseline = json.load(file)

    leaderboard = pd.read_csv(LEADERBOARD_PATH)

    if leaderboard["Model"].duplicated().any():
        raise ValueError("Duplicate model names in leaderboard")

    current = leaderboard.set_index("Model")

    failed = False

    # Ensure every baseline model is evaluated
    missing = set(baseline) - set(current.index)

    if missing:
        print(f"Missing models: {sorted(missing)}")
        failed = True

    for model_name, baseline_f1 in baseline.items():

        if model_name not in current.index:
            continue

        current_f1 = float(current.loc[model_name, "f1"])

        drop = baseline_f1 - current_f1

        print(f"\nModel: {model_name}")
        print(f"Baseline F1: {baseline_f1:.4f}")
        print(f"Current F1:  {current_f1:.4f}")
        print(f"F1 drop:     {drop:.4f}")

        if pd.isna(current_f1) or drop > TOLERANCE:
            print("REGRESSION DETECTED")
            failed = True
        else:
            print("PASSED")

    if failed:
        print("\nRegression check FAILED")
        sys.exit(1)

    print("\nAll regression checks PASSED")


if __name__ == "__main__":
    check_regression()
