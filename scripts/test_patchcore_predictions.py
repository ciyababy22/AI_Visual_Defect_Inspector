from pathlib import Path

from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = PROJECT_ROOT / "data" / "MVTecAD"

CHECKPOINT_PATH = (
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


def main():
    print("Checking PatchCore checkpoint...")

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT_PATH}"
        )

    print(f"Checkpoint found:\n{CHECKPOINT_PATH}")

    print("\nCreating MVTec AD datamodule...")

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

    print("\nRunning PatchCore prediction...")

    engine = Engine()

    predictions = engine.predict(
        model=model,
        datamodule=datamodule,
        ckpt_path=str(CHECKPOINT_PATH),
    )

    print("\nPrediction completed successfully.")

    print(f"Number of prediction batches: {len(predictions)}")

    if predictions:
        first_batch = predictions[0]

        print("\nFirst prediction batch:")
        print(f"Prediction type: {type(first_batch)}")

        print("\nAnomaly scores:")
        print(first_batch.pred_score)

        print("\nPredicted labels:")
        print(first_batch.pred_label)

        print("\nImage paths:")
        for path in first_batch.image_path:
            print(f"  {path}")


if __name__ == "__main__":
    main()