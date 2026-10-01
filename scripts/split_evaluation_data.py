from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "results" / "predictions.csv"
VALIDATION_PATH = PROJECT_ROOT / "results" / "validation_predictions.csv"
EVALUATION_PATH = PROJECT_ROOT / "results" / "final_evaluation_predictions.csv"


def main():
    print("Loading predictions...")

    df = pd.read_csv(INPUT_PATH)

    print(f"Total images: {len(df)}")

    validation_df, evaluation_df = train_test_split(
        df,
        test_size=0.5,
        random_state=42,
        stratify=df["actual_label"],
    )

    validation_df = validation_df.sort_values(
        by=["category", "image"]
    ).reset_index(drop=True)

    evaluation_df = evaluation_df.sort_values(
        by=["category", "image"]
    ).reset_index(drop=True)

    validation_df.to_csv(
        VALIDATION_PATH,
        index=False,
    )

    evaluation_df.to_csv(
        EVALUATION_PATH,
        index=False,
    )

    print("\nSplit completed.")

    print(f"Validation images: {len(validation_df)}")
    print(f"Evaluation images: {len(evaluation_df)}")

    print("\nValidation class distribution:")
    print(validation_df["actual_label"].value_counts())

    print("\nEvaluation class distribution:")
    print(evaluation_df["actual_label"].value_counts())

    print("\nSaved:")
    print(VALIDATION_PATH)
    print(EVALUATION_PATH)


if __name__ == "__main__":
    main()