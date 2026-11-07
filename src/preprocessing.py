
import cv2
import numpy as np

# Image configuration
IMG_SIZE = 224


def preprocess_image(image):
    """Resize and normalize an uploaded chest X-ray image."""

    if image is None:
        raise ValueError("Please upload a chest X-ray image.")

    image = np.asarray(image)

    # Handle grayscale images
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)

    # Handle RGBA images
    if image.shape[-1] == 4:
        image = image[:, :, :3]

    # Resize image
    image = cv2.resize(
        image.astype(np.uint8),
        (IMG_SIZE, IMG_SIZE)
    )

    return image


def apply_clahe(image):
    """Apply CLAHE contrast enhancement."""

    image = np.clip(image, 0, 255).astype(np.uint8)

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    enhanced = np.stack(
        [enhanced] * 3,
        axis=-1
    )

    return enhanced.astype(np.float32)
