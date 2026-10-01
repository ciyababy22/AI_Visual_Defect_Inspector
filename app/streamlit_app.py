from pathlib import Path
import sys
import uuid

import streamlit as st
from PIL import Image

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.predictor import PatchCorePredictor
from src.visualization import create_heatmap, create_overlay


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="AI Visual Defect Inspector",
    page_icon="🔍",
    layout="wide",
)


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------

@st.cache_resource
def load_predictor():
    return PatchCorePredictor()


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

def display_header():

    st.title("🔍 AI Visual Defect Inspector")

    st.markdown(
        """
        **AI-powered visual inspection using PatchCore**

        Upload a manufacturing image to identify potential
        visual defects and visualize the regions that contribute
        to the anomaly score.
        """
    )

    st.divider()


# ---------------------------------------------------------
# Model information
# ---------------------------------------------------------

def display_model_info():

    st.sidebar.header("Model Information")

    st.sidebar.write(
        "**Model:** PatchCore"
    )

    st.sidebar.write(
        "**Backbone:** Wide ResNet-50-2"
    )

    st.sidebar.write(
        "**Dataset:** MVTec AD"
    )

    st.sidebar.write(
        "**Category:** Bottle"
    )

    st.sidebar.write(
        "**Decision threshold:** 0.327"
    )

    st.sidebar.divider()

    st.sidebar.caption(
        "PatchCore is an anomaly detection model "
        "trained primarily on normal images."
    )


# ---------------------------------------------------------
# Main application
# ---------------------------------------------------------

def main():

    display_header()
    display_model_info()

    st.subheader("Upload Image")

    uploaded_file = st.file_uploader(
        "Choose a manufacturing image",
        type=["png", "jpg", "jpeg"],
        help="Upload a PNG or JPEG image for inspection.",
    )

    if uploaded_file is None:

        st.info(
            "Upload an image to begin the inspection."
        )

        st.markdown(
            """
            ### How it works

            1. Upload an image.
            2. PatchCore analyzes its visual features.
            3. An anomaly score is generated.
            4. The score is compared with the decision threshold.
            5. An anomaly heatmap highlights suspicious regions.
            """
        )

        return

    # -----------------------------------------------------
    # Display uploaded image
    # -----------------------------------------------------

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.subheader("Input Image")

    st.image(
        image,
        caption=uploaded_file.name,
        use_container_width=True,
    )

    st.divider()

    # -----------------------------------------------------
    # Run inspection
    # -----------------------------------------------------

    if st.button(
        "🔍 Inspect Image",
        type="primary",
        use_container_width=True,
    ):

        with st.spinner(
            "Running PatchCore inspection..."
        ):

            predictor = load_predictor()

            inference_dir = (
                PROJECT_ROOT
                / "data"
                / "inference"
            )

            inference_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            # Use a unique filename so that previous
            # inference files cannot cause file conflicts.
            unique_name = (
                f"upload_{uuid.uuid4().hex}.png"
            )

            image_path = (
                inference_dir
                / unique_name
            )

            image.save(
                image_path,
                format="PNG",
            )

            result = predictor.predict(
                image_path
            )

        # -------------------------------------------------
        # Extract prediction
        # -------------------------------------------------

        score = float(
            result["anomaly_score"]
        )

        prediction = result[
            "predicted_label"
        ]

        heatmap = create_heatmap(
            result["anomaly_map"]
        )

        overlay = create_overlay(
            result["image"],
            result["anomaly_map"],
        )

        # -------------------------------------------------
        # Result
        # -------------------------------------------------

        st.subheader("Inspection Result")

        if prediction == "DEFECTIVE":

            st.error(
                "⚠️ DEFECTIVE",
                icon="⚠️",
            )

            st.write(
                "The anomaly score is above the "
                "configured decision threshold."
            )

        else:

            st.success(
                "✅ OK",
                icon="✅",
            )

            st.write(
                "The anomaly score is below the "
                "configured decision threshold."
            )

        # -------------------------------------------------
        # Metrics
        # -------------------------------------------------

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Anomaly Score",
                f"{score:.4f}",
            )

        with col2:

            st.metric(
                "Decision Threshold",
                "0.327",
            )

        with col3:

            difference = score - 0.327

            st.metric(
                "Score vs Threshold",
                f"{difference:+.4f}",
            )

        st.divider()

        # -------------------------------------------------
        # Visualization
        # -------------------------------------------------

        st.subheader(
            "Anomaly Visualization"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                heatmap,
                caption="PatchCore Anomaly Heatmap",
                use_container_width=True,
            )

        with col2:

            st.image(
                overlay,
                caption="Anomaly Overlay",
                use_container_width=True,
            )

        st.divider()

        # -------------------------------------------------
        # Interpretation
        # -------------------------------------------------

        st.subheader(
            "Interpretation"
        )

        if prediction == "DEFECTIVE":

            st.warning(
                f"The anomaly score ({score:.4f}) "
                f"is above the threshold (0.327). "
                "The image is therefore classified as "
                "**DEFECTIVE**."
            )

        else:

            st.info(
                f"The anomaly score ({score:.4f}) "
                f"is below the threshold (0.327). "
                "The image is therefore classified as "
                "**OK**."
            )

        # -------------------------------------------------
        # Technical details
        # -------------------------------------------------

        with st.expander(
            "Technical Details"
        ):

            st.write(
                f"**Model:** PatchCore"
            )

            st.write(
                f"**Backbone:** Wide ResNet-50-2"
            )

            st.write(
                f"**Dataset:** MVTec AD"
            )

            st.write(
                f"**Category:** Bottle"
            )

            st.write(
                f"**Anomaly score:** {score:.6f}"
            )

            st.write(
                f"**Decision threshold:** 0.327"
            )

            st.write(
                f"**Prediction:** {prediction}"
            )

        st.caption(
            "Note: The anomaly score is a model score, "
            "not a calibrated probability."
        )


# ---------------------------------------------------------
# Run application
# ---------------------------------------------------------

if __name__ == "__main__":
    main()