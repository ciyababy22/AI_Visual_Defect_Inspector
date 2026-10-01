from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "final_evaluation_predictions.csv"
)


def main():
    df = pd.read_csv(INPUT_PATH)

    threshold = 0.327

    y_true = (
        df["actual_label"] == "DEFECTIVE"
    ).astype(int)

    y_pred = (
        df["anomaly_score"] >= threshold
    ).astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(
        y_true, y_pred, zero_division=0
    )
    recall = recall_score(
        y_true, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_true, y_pred, zero_division=0
    )
    roc_auc = roc_auc_score(
        y_true, df["anomaly_score"]
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    print("\n======================================")
    print("FINAL EVALUATION")
    print("======================================")

    print(f"Threshold : {threshold:.3f}")
    print(f"Images    : {len(df)}")

    print("\nMetrics:")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nPrediction counts:")
    print(
        pd.Series(y_pred)
        .map({0: "OK", 1: "DEFECTIVE"})
        .value_counts()
    )


if __name__ == "__main__":
    main()