from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = PROJECT_ROOT / "data" / "MVTecAD"

CHECKPOINT = (
    PROJECT_ROOT
    / "results"
    / "patchcore"
    / "Patchcore"
    / "MVTecAD"
    / "bottle"
    / "v0"
    / "weights"
    / "lightning"
    / "model.ckpt"
)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("Creating MVTec AD datamodule...")

    datamodule = MVTecAD(
        root=str(DATASET_ROOT),
        category="bottle",
        train_batch_size=8,
        eval_batch_size=8,
        num_workers=0,
    )

    print("Creating PatchCore model...")

    model = Patchcore(
        backbone="wide_resnet50_2",
    )

    print("Loading trained PatchCore model...")

    engine = Engine()

    print("\nRunning inference on test set...")

    predictions = engine.predict(
        model=model,
        datamodule=datamodule,
        ckpt_path=str(CHECKPOINT),
    )

    print("Inference completed.")

    # -----------------------------------------------------
    # Collect scores and labels
    # -----------------------------------------------------

    true_labels = []
    anomaly_scores = []

    for batch in predictions:

        true_labels.extend(
            batch.gt_label.cpu().numpy().astype(int).tolist()
        )

        anomaly_scores.extend(
            batch.pred_score.cpu().numpy().tolist()
        )

    true_labels = np.array(true_labels)
    anomaly_scores = np.array(anomaly_scores)

    # -----------------------------------------------------
    # Test different thresholds
    # -----------------------------------------------------

    thresholds = np.linspace(
        anomaly_scores.min(),
        anomaly_scores.max(),
        100,
    )

    results = []

    for threshold in thresholds:

        predicted_labels = (
            anomaly_scores >= threshold
        ).astype(int)

        accuracy = accuracy_score(
            true_labels,
            predicted_labels,
        )

        precision = precision_score(
            true_labels,
            predicted_labels,
            zero_division=0,
        )

        recall = recall_score(
            true_labels,
            predicted_labels,
            zero_division=0,
        )

        f1 = f1_score(
            true_labels,
            predicted_labels,
            zero_division=0,
        )

        results.append(
            (
                threshold,
                accuracy,
                precision,
                recall,
                f1,
            )
        )

    # -----------------------------------------------------
    # Find threshold with highest F1
    # -----------------------------------------------------

    best_result = max(
        results,
        key=lambda item: item[4],
    )

    best_threshold = best_result[0]
    best_accuracy = best_result[1]
    best_precision = best_result[2]
    best_recall = best_result[3]
    best_f1 = best_result[4]

    print("\n" + "=" * 55)
    print("THRESHOLD ANALYSIS")
    print("=" * 55)

    print(f"Current score range : "
          f"{anomaly_scores.min():.4f} - "
          f"{anomaly_scores.max():.4f}")

    print("\nBest F1 threshold:")
    print(f"Threshold          : {best_threshold:.4f}")
    print(f"Accuracy            : {best_accuracy:.4f}")
    print(f"Precision           : {best_precision:.4f}")
    print(f"Recall              : {best_recall:.4f}")
    print(f"F1 Score            : {best_f1:.4f}")

    # -----------------------------------------------------
    # Plot F1 vs threshold
    # -----------------------------------------------------

    result_array = np.array(results)

    plt.figure(figsize=(9, 6))

    plt.plot(
        result_array[:, 0],
        result_array[:, 4],
        label="F1 Score",
    )

    plt.axvline(
        best_threshold,
        linestyle="--",
        label=f"Best Threshold = {best_threshold:.4f}",
    )

    plt.xlabel("Anomaly Score Threshold")
    plt.ylabel("F1 Score")
    plt.title("PatchCore Threshold Analysis")

    plt.legend()
    plt.grid(True)

    output_path = (
        PROJECT_ROOT
        / "results"
        / "threshold_analysis.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()

    print("\nThreshold analysis plot saved to:")
    print(output_path)


if __name__ == "__main__":
    main()