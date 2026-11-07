
import cv2
import numpy as np

from .preprocessing import IMG_SIZE


def create_overlay(image, heatmap):
    """Generate colored Grad-CAM heatmap and overlay."""

    # Resize heatmap
    heatmap_resized = cv2.resize(
        heatmap,
        (IMG_SIZE, IMG_SIZE)
    )

    # Convert heatmap to 8-bit
    heatmap_uint8 = np.uint8(
        255 * heatmap_resized
    )

    # Apply JET color map
    colored_heatmap = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    # Convert BGR to RGB
    colored_heatmap = cv2.cvtColor(
        colored_heatmap,
        cv2.COLOR_BGR2RGB
    )

    # Combine original image and heatmap
    overlay = cv2.addWeighted(
        image.astype(np.uint8),
        0.55,
        colored_heatmap,
        0.45,
        0
    )

    return colored_heatmap, overlay
