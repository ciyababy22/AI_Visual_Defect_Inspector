from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = PROJECT_ROOT / "data" / "MVTecAD"

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

# ---------------------------------------------------------
# Load dataset & model
# ---------------------------------------------------------

datamodule = MVTecAD(
    root=str(DATASET_ROOT),
    category="bottle",
    train_batch_size=8,
    eval_batch_size=8,
    num_workers=0,
)

model = Patchcore(backbone="wide_resnet50_2")
engine = Engine()

print("Running inference...")

predictions = engine.predict(
    model=model,
    datamodule=datamodule,
    ckpt_path=str(CHECKPOINT),
)

print("Prediction completed.")

# ---------------------------------------------------------
# Select first image from first prediction batch
# ---------------------------------------------------------

batch = predictions[0]

image = batch.image[0].permute(1, 2, 0).cpu().numpy()

heatmap = batch.anomaly_map[0].cpu().numpy()

mask = batch.gt_mask[0].cpu().numpy()

score = float(batch.pred_score[0])

label = bool(batch.pred_label[0])

# Normalize image for display
image = (image - image.min()) / (image.max() - image.min())

# ---------------------------------------------------------
# Display
# ---------------------------------------------------------

fig = plt.figure(figsize=(15, 5))

# Figure 1
ax1 = fig.add_subplot(1, 3, 1)
ax1.imshow(image)
ax1.set_title("1. Original Image")
ax1.axis("off")

# Figure 2
ax2 = fig.add_subplot(1, 3, 2)
ax2.imshow(heatmap, cmap="jet")
ax2.set_title(f"2. AI Heatmap\nScore = {score:.3f}")
ax2.axis("off")

# Figure 3
ax3 = fig.add_subplot(1, 3, 3)
ax3.imshow(image)
ax3.imshow(mask, cmap="Reds", alpha=0.45)
ax3.set_title(
    "3. Ground Truth\n"
    + ("DEFECTIVE" if label else "OK")
)
ax3.axis("off")

plt.tight_layout()
plt.show()