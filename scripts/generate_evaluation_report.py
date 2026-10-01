from pathlib import Path

import pandas as pd

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore


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

OUTPUT_PATH = PROJECT_ROOT / "results" / "predictions.csv"


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

    print("\nRunning inference on test dataset...")

    engine = Engine()

    predictions = engine.predict(
        model=model,
        datamodule=datamodule,
        ckpt_path=str(CHECKPOINT),
    )

    print("Inference completed.")

    rows = []

    for batch in predictions:
        for i in range(len(batch.pred_score)):
            image_path = Path(batch.image_path[i])

            actual_label = bool(batch.gt_label[i])
            anomaly_score = float(batch.pred_score[i])

            predicted_label = anomaly_score >= 0.3285

            rows.append(
                {
                    "image": image_path.name,
                    "category": image_path.parent.name,
                    "actual_label": "DEFECTIVE"
                    if actual_label
                    else "OK",
                    "anomaly_score": anomaly_score,
                    "predicted_label": "DEFECTIVE"
                    if predicted_label
                    else "OK",
                    "correct": actual_label == predicted_label,
                    "image_path": str(image_path),
                }
            )

    df = pd.DataFrame(rows)

    df = df.sort_values(
        by=["category", "image"]
    ).reset_index(drop=True)

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nEvaluation report created successfully.")
    print(f"Total images: {len(df)}")
    print(f"Correct predictions: {df['correct'].sum()}")
    print(f"Incorrect predictions: {(~df['correct']).sum()}")

    print(f"\nSaved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()