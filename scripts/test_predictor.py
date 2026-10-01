from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.predictor import PatchCorePredictor

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_IMAGE = (
    PROJECT_ROOT
    / "data"
    / "inference"
    / "000.png"
)


def main():
    print("Initializing predictor...")

    predictor = PatchCorePredictor()

    print("\nRunning prediction...")
    
    result = predictor.predict(TEST_IMAGE)

    print("\nPrediction completed.")

    print(f"Image shape    : {result['image'].shape}")
    print(f"Anomaly score  : {result['anomaly_score']:.4f}")
    print(f"Prediction     : {result['predicted_label']}")
    print(f"Heatmap shape  : {result['anomaly_map'].shape}")


if __name__ == "__main__":
     main()