from pathlib import Path
from PIL import Image


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data" / "MVTecAD"
CATEGORY = "bottle"

CATEGORY_ROOT = DATASET_ROOT / CATEGORY


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def count_images(folder: Path) -> int:
    """Count PNG images directly inside a folder."""
    if not folder.exists():
        return 0

    return len(list(folder.glob("*.png")))


def check_images(folder: Path) -> tuple[int, int, set[tuple[int, int]]]:
    """
    Check images for readability and collect their dimensions.

    Returns:
        total_images
        corrupted_images
        image_dimensions
    """
    total = 0
    corrupted = 0
    dimensions = set()

    if not folder.exists():
        return total, corrupted, dimensions

    for image_path in folder.glob("*.png"):
        total += 1

        try:
            with Image.open(image_path) as image:
                image.verify()

            # Re-open after verify() to safely read dimensions.
            with Image.open(image_path) as image:
                dimensions.add(image.size)

        except Exception:
            corrupted += 1
            print(f"WARNING: Could not read {image_path}")

    return total, corrupted, dimensions


def inspect_split(split_name: str) -> None:
    """Inspect one dataset split."""
    split_root = CATEGORY_ROOT / split_name

    print(f"\n{'=' * 60}")
    print(f"{split_name.upper()}")
    print(f"{'=' * 60}")

    if not split_root.exists():
        print(f"ERROR: Folder does not exist: {split_root}")
        return

    total_images = 0

    for defect_folder in sorted(split_root.iterdir()):

        if not defect_folder.is_dir():
            continue

        total, corrupted, dimensions = check_images(defect_folder)

        total_images += total

        print(f"\nCategory: {defect_folder.name}")
        print(f"  Images:    {total}")
        print(f"  Corrupted: {corrupted}")
        print(f"  Dimensions: {dimensions}")

    print(f"\nTotal images: {total_images}")


# ---------------------------------------------------------
# Ground-truth inspection
# ---------------------------------------------------------

def inspect_ground_truth() -> None:
    """Inspect ground-truth anomaly masks."""
    ground_truth_root = CATEGORY_ROOT / "ground_truth"

    print(f"\n{'=' * 60}")
    print("GROUND TRUTH")
    print(f"{'=' * 60}")

    if not ground_truth_root.exists():
        print(f"ERROR: Folder does not exist: {ground_truth_root}")
        return

    total_masks = 0

    for defect_folder in sorted(ground_truth_root.iterdir()):

        if not defect_folder.is_dir():
            continue

        count = count_images(defect_folder)
        total_masks += count

        print(f"{defect_folder.name}: {count} masks")

    print(f"\nTotal masks: {total_masks}")


# ---------------------------------------------------------
# Dataset summary
# ---------------------------------------------------------

def main() -> None:
    print("\nAI Visual Defect Inspector")
    print("Dataset Inspection")
    print("-" * 60)

    print(f"Dataset:  {DATASET_ROOT}")
    print(f"Category: {CATEGORY}")

    if not CATEGORY_ROOT.exists():
        print("\nERROR: Dataset category does not exist.")
        print(f"Expected location: {CATEGORY_ROOT}")
        return

    # Training data
    inspect_split("train")

    # Test data
    inspect_split("test")

    # Ground truth
    inspect_ground_truth()

    print(f"\n{'=' * 60}")
    print("DATASET INSPECTION COMPLETE")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()