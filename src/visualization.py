import cv2
import numpy as np


def create_heatmap(anomaly_map):
    """
    Convert an anomaly map into a colored heatmap.
    """

    anomaly_map = np.asarray(
        anomaly_map,
        dtype=np.float32,
    )

    anomaly_map = cv2.normalize(
        anomaly_map,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    )

    anomaly_map = anomaly_map.astype(
        np.uint8
    )

    heatmap = cv2.applyColorMap(
        anomaly_map,
        cv2.COLORMAP_JET,
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB,
    )

    return heatmap


def create_overlay(
    image,
    anomaly_map,
    alpha=0.45,
):
    """
    Overlay the anomaly heatmap
    on the original image.
    """

    image = np.asarray(
        image,
        dtype=np.uint8,
    )

    heatmap = create_heatmap(
        anomaly_map
    )

    heatmap = cv2.resize(
        heatmap,
        (
            image.shape[1],
            image.shape[0],
        ),
    )

    overlay = cv2.addWeighted(
        image,
        1 - alpha,
        heatmap,
        alpha,
        0,
    )

    return overlay