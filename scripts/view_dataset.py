from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = PROJECT_ROOT / "data" / "MVTecAD" / "bottle"


# ---------------------------------------------------------
# Image paths
# ---------------------------------------------------------

normal_image = DATASET_ROOT / "train" / "good" / "000.png"

test_good = DATASET_ROOT / "test" / "good" / "000.png"

broken_large = DATASET_ROOT / "test" / "broken_large" / "000.png"

broken_small = DATASET_ROOT / "test" / "broken_small" / "000.png"

contamination = DATASET_ROOT / "test" / "contamination" / "000.png"

broken_large_mask = (
    DATASET_ROOT / "ground_truth" / "broken_large" / "000.png"
)


# ---------------------------------------------------------
# Helper
# ---------------------------------------------------------

def load_image(path: Path):
    """Load an image and verify that it exists."""
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    return Image.open(path)


def show_image(image_path: Path, title: str) -> None:
    """Display a single image."""
    image = load_image(image_path)

    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("AI Visual Defect Inspector")
    print("Dataset Visual Inspection")
    print("-" * 60)

    # 1. Normal training image
    show_image(
        normal_image,
        "Training Image - Normal Bottle"
    )

    # 2. Normal test image
    show_image(
        test_good,
        "Test Image - Normal Bottle"
    )

    # 3. Broken large
    show_image(
        broken_large,
        "Defect - Broken Large"
    )

    # 4. Broken small
    show_image(
        broken_small,
        "Defect - Broken Small"
    )

    # 5. Contamination
    show_image(
        contamination,
        "Defect - Contamination"
    )

    # 6. Ground truth mask
    mask = load_image(broken_large_mask)

    plt.figure(figsize=(6, 6))
    plt.imshow(mask, cmap="gray")
    plt.title("Ground Truth Mask - Broken Large")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()