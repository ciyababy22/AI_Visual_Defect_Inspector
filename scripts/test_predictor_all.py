from pathlib import Path
import shutil
import sys

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.predictor import PatchCorePredictor


TEST_ROOT = (
    PROJECT_ROOT
    / "data"
    / "MVTecAD"
    / "bottle"
    / "test"
)

INFERENCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "inference"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "predictor_test_results.csv"
)


def main():
    print("Initializing PatchCore predictor...")

    predictor = PatchCorePredictor()

    INFERENCE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_paths = sorted(
        TEST_ROOT.glob("*/*.png")
    )

    print(
        f"\nFound {len(image_paths)} test images."
    )

    rows = []

    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):
        category = image_path.parent.name

        print(
            f"[{index}/{len(image_paths)}] "
            f"{category}/{image_path.name}"
        )

        # Give every image a unique filename.
        unique_name = (
            f"{category}_{image_path.name}"
        )

        inference_image = (
            INFERENCE_DIR
            / unique_name
        )

        # Copy the image to the inference folder.
        shutil.copyfile(
            image_path,
            inference_image,
        )

        # Run prediction.
        result = predictor.predict(
            inference_image
        )

        actual_label = (
            "OK"
            if category == "good"
            else "DEFECTIVE"
        )

        predicted_label = (
            result["predicted_label"]
        )

        rows.append(
            {
                "image": image_path.name,
                "category": category,
                "actual_label": actual_label,
                "anomaly_score": result[
                    "anomaly_score"
                ],
                "predicted_label": predicted_label,
                "correct": (
                    actual_label
                    == predicted_label
                ),
            }
        )

    df = pd.DataFrame(rows)

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    correct_count = int(
        df["correct"].sum()
    )

    incorrect_count = (
        len(df) - correct_count
    )

    accuracy = (
        correct_count / len(df)
        if len(df) > 0
        else 0
    )

    print(
        "\n======================================"
    )
    print(
        "PREDICTOR TEST COMPLETED"
    )
    print(
        "======================================"
    )

    print(
        f"Total images : {len(df)}"
    )

    print(
        f"Correct      : {correct_count}"
    )

    print(
        f"Incorrect    : {incorrect_count}"
    )

    print(
        f"Accuracy     : {accuracy:.4f}"
    )

    print("\nPrediction counts:")

    print(
        df["predicted_label"].value_counts()
    )

    print("\nResults saved to:")

    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()