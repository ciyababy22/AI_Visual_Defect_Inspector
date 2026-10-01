from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "final_evaluation_predictions.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "final_confusion_matrix.png"
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

    cm = confusion_matrix(y_true, y_pred)

    labels = ["OK", "DEFECTIVE"]

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.imshow(cm)

    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted Label",
        ylabel="Actual Label",
        title="PatchCore Final Evaluation\nMVTec AD - Bottle",
    )

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                fontsize=16,
            )

    plt.tight_layout()

    fig.savefig(
        OUTPUT_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()

    print("\nConfusion matrix saved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()