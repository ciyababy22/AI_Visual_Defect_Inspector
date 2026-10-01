from pathlib import Path
import sys

import matplotlib.pyplot as plt

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.predictor import PatchCorePredictor
from src.visualization import create_heatmap, create_overlay


IMAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "MVTecAD"
    / "bottle"
    / "test"
    / "broken_large"
    / "000.png"
)


def main():
    print("Loading PatchCore predictor...")

    predictor = PatchCorePredictor()

    print("Running prediction...")

    result = predictor.predict(
        IMAGE_PATH
    )

    image = result["image"]
    anomaly_map = result["anomaly_map"]

    heatmap = create_heatmap(
        anomaly_map
    )

    overlay = create_overlay(
        image,
        anomaly_map,
    )

    print(
        f"Anomaly score : "
        f"{result['anomaly_score']:.4f}"
    )

    print(
        f"Prediction    : "
        f"{result['predicted_label']}"
    )

    # Original image
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")

    # Heatmap
    plt.figure(figsize=(6, 6))
    plt.imshow(heatmap)
    plt.title("PatchCore Anomaly Heatmap")
    plt.axis("off")

    # Overlay
    plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    plt.title("Anomaly Overlay")
    plt.axis("off")

    plt.show()


if __name__ == "__main__":
    main()