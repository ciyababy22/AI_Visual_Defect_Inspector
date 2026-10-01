from pathlib import Path

from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data" / "MVTecAD"
RESULTS_ROOT = PROJECT_ROOT / "results" / "patchcore"


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

    print("Creating Anomalib engine...")

    engine = Engine(
        max_epochs=1,
        default_root_dir=str(RESULTS_ROOT),
    )

    print("\nStarting PatchCore training...")
    print("Training uses the normal bottle images from train/good.")

    engine.fit(
        model=model,
        datamodule=datamodule,
    )

    print("\nPatchCore training completed successfully.")


if __name__ == "__main__":
    main()