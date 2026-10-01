# 🔍 AI Visual Defect Inspector

An AI-powered visual inspection system for detecting manufacturing defects using computer vision and anomaly detection.

The project uses **PatchCore**, an industrial anomaly detection method, to learn the visual characteristics of normal products and identify images that contain potential defects.

---

## 🚀 Features

- AI-based visual defect detection
- PatchCore anomaly detection
- Wide ResNet-50-2 backbone
- MVTec AD bottle dataset
- Image-level anomaly scoring
- Configurable decision threshold
- Anomaly heatmap visualization
- Anomaly overlay visualization
- Streamlit web interface
- Reusable Python prediction API
- Evaluation metrics and confusion matrix
- CPU-compatible inference

---

## 🧠 How It Works

The system follows this pipeline:

```text
                Input Image
                     │
                     ▼
             Image Preprocessing
                     │
                     ▼
              PatchCore Model
                     │
                     ▼
             Anomaly Score
                     │
              ┌──────┴──────┐
              │             │
       Score < 0.327   Score >= 0.327
              │             │
              ▼             ▼
             OK         DEFECTIVE
              │             │
              └──────┬──────┘
                     │
                     ▼
             Anomaly Heatmap
                     │
                     ▼
              Visual Overlay