from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data" / "MVTecAD" / "bottle"

IMAGE_PATH = DATASET_ROOT / "test" / "broken_large" / "000.png"
MASK_PATH = DATASET_ROOT / "ground_truth" / "broken_large" / "000_mask.png"


def main():
    image = Image.open(IMAGE_PATH).convert("RGB")
    mask = Image.open(MASK_PATH).convert("L")
    # Figure 1: Original image
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.title("1. Original Defective Image")
    plt.axis("off")
    plt.tight_layout()


    # Figure 2: Ground-truth mask
    plt.figure(figsize=(6, 6))
    plt.imshow(mask, cmap="gray")
    plt.title("2. Ground-Truth Defect Mask")
    plt.axis("off")
    plt.tight_layout()
 

    # Figure 3: Overlay
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.imshow(mask, cmap="Reds", alpha=0.45)
    plt.title("3. Defect Overlay")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()