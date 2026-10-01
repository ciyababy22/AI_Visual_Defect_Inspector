from pathlib import Path

import json
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
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

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


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

    print("\nRunning inference on the complete test set...")

    predictions = engine.predict(
        model=model,
        datamodule=datamodule,
        ckpt_path=str(CHECKPOINT),
    )

    print("Inference completed.")

    # -----------------------------------------------------
    # Collect predictions
    # -----------------------------------------------------

    true_labels = []
    predicted_labels = []
    anomaly_scores = []
    image_paths = []

    for batch in predictions:

        true_labels.extend(
            batch.gt_label.cpu().numpy().astype(int).tolist()
        )

        predicted_labels.extend(
            batch.pred_label.cpu().numpy().astype(int).tolist()
        )

        anomaly_scores.extend(
            batch.pred_score.cpu().numpy().tolist()
        )

        image_paths.extend(batch.image_path)

    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    anomaly_scores = np.array(anomaly_scores)

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

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

    roc_auc = roc_auc_score(
        true_labels,
        anomaly_scores,
    )

    # -----------------------------------------------------
    # Print results
    # -----------------------------------------------------

    print("\n" + "=" * 50)
    print("PATCHCORE EVALUATION RESULTS")
    print("=" * 50)

    print(f"Total test images : {len(true_labels)}")
    print(f"Normal images     : {np.sum(true_labels == 0)}")
    print(f"Defective images  : {np.sum(true_labels == 1)}")

    print("\nMetrics:")

    print(f"Accuracy          : {accuracy:.4f}")
    print(f"Precision         : {precision:.4f}")
    print(f"Recall            : {recall:.4f}")
    print(f"F1 Score          : {f1:.4f}")
    print(f"ROC-AUC           : {roc_auc:.4f}")

    # -----------------------------------------------------
    # Confusion matrix
    # -----------------------------------------------------

    cm = confusion_matrix(
        true_labels,
        predicted_labels,
    )

    print("\nConfusion Matrix:")
    print(cm)

    # -----------------------------------------------------
    # Save confusion matrix
    # -----------------------------------------------------

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["OK", "DEFECTIVE"],
    )

    display.plot()

    plt.title("PatchCore Confusion Matrix")
    plt.tight_layout()

    confusion_path = RESULTS_DIR / "confusion_matrix.png"

    plt.savefig(
        confusion_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"\nConfusion matrix saved to:")
    print(confusion_path)

    # -----------------------------------------------------
    # Save metrics JSON
    # -----------------------------------------------------

    metrics = {
        "model": "PatchCore",
        "dataset": "MVTec AD",
        "category": "bottle",
        "total_test_images": int(len(true_labels)),
        "normal_images": int(np.sum(true_labels == 0)),
        "defective_images": int(np.sum(true_labels == 1)),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
    }

    metrics_path = RESULTS_DIR / "metrics.json"

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
        )

    print(f"Metrics saved to:")
    print(metrics_path)

    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    main()