from pathlib import Path
import shutil
import tempfile

import numpy as np
from PIL import Image

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT = (
    PROJECT_ROOT
    / "results"
    / "patchcore"
    / "Patchcore"
    / "MVTecAD"
    / "bottle"
    / "v0"
    / "weights"
    / "lightning"
    / "model.ckpt"
)

THRESHOLD = 0.327


class PatchCorePredictor:

    def __init__(self):
        print("Loading PatchCore model...")

        self.model = Patchcore(
            backbone="wide_resnet50_2",
        )

        self.engine = Engine()

        print("PatchCore predictor initialized.")

    def predict(self, image_path):

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        image = Image.open(
            image_path
        ).convert("RGB")

        # Create a temporary folder containing
        # ONLY the image being predicted.
        with tempfile.TemporaryDirectory(
            dir=PROJECT_ROOT / "data"
        ) as temp_dir:

            temp_dir = Path(temp_dir)

            inference_image = (
                temp_dir
                / image_path.name
            )

            shutil.copyfile(
                image_path,
                inference_image,
            )

            datamodule = Folder(
                name="single_image",
                root=str(temp_dir),
                normal_dir=".",
                normal_test_dir=".",
                train_batch_size=1,
                eval_batch_size=1,
                num_workers=0,
            )

            predictions = self.engine.predict(
                model=self.model,
                datamodule=datamodule,
                ckpt_path=str(CHECKPOINT),
            )

        if not predictions:
            raise RuntimeError(
                "No prediction was returned."
            )

        prediction = predictions[0]

        score = float(
            prediction.pred_score[0]
        )

        predicted_label = (
            "DEFECTIVE"
            if score >= THRESHOLD
            else "OK"
        )

        anomaly_map = (
            prediction.anomaly_map[0]
            .cpu()
            .numpy()
        )

        return {
            "image": np.array(image),
            "anomaly_score": score,
            "predicted_label": predicted_label,
            "anomaly_map": anomaly_map,
        }