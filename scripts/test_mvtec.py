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

    print("MVTec AD datamodule created successfully.")
    print(f"Dataset root: {DATASET_ROOT}")
    print("Category: bottle")

    print("\nPreparing dataset...")
    datamodule.prepare_data()

    print("Dataset preparation complete.")


if __name__ == "__main__":
    main()