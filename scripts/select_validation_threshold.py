from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "results" / "validation_predictions.csv"


def main():
    df = pd.read_csv(INPUT_PATH)

    y_true = (df["actual_label"] == "DEFECTIVE").astype(int)
    scores = df["anomaly_score"]

    best_threshold = None
    best_f1 = -1
    best_metrics = None

    for threshold in [i / 1000 for i in range(0, 1001)]:
        y_pred = (scores >= threshold).astype(int)

        f1 = f1_score(y_true, y_pred, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

            best_metrics = {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(
                    y_true, y_pred, zero_division=0
                ),
                "recall": recall_score(
                    y_true, y_pred, zero_division=0
                ),
                "f1": f1,
            }

    print("\n======================================")
    print("VALIDATION THRESHOLD SELECTION")
    print("======================================")

    print(f"Threshold : {best_threshold:.3f}")
    print(f"Accuracy  : {best_metrics['accuracy']:.4f}")
    print(f"Precision : {best_metrics['precision']:.4f}")
    print(f"Recall    : {best_metrics['recall']:.4f}")
    print(f"F1 Score  : {best_metrics['f1']:.4f}")


if __name__ == "__main__":
    main()