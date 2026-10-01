from pathlib import Path

from anomalib.data import MVTecAD


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data" / "MVTecAD"


def main():
    print("Creating MVTec AD datamodule...")

    datamodule = MVTecAD(
        root=str(DATASET_ROOT),
        category="bottle",
        train_batch_size=8,
        eval_batch_size=8,
        num_workers=0,
    )

    print("Preparing dataset...")
    datamodule.prepare_data()

    print("Setting up datamodule...")
    datamodule.setup()

    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))

    print("\nTraining batch loaded successfully.")
    print(f"Batch type: {type(batch)}")

    print(f"\nImage tensor shape: {batch.image.shape}")
    print(f"Label shape: {batch.gt_label.shape}")
    print(f"Mask shape: {batch.gt_mask.shape}")

    print(f"\nImage paths:")
    for path in batch.image_path:
        print(f"  {path}")


if __name__ == "__main__":
    main()