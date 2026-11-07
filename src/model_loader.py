
from pathlib import Path
import tensorflow as tf

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Model paths
BASELINE_PATH = BASE_DIR / "models" / "efficientnetb0_baseline.keras"
CLAHE_PATH = BASE_DIR / "models" / "efficientnetb0_clahe.keras"


def load_models():
    """Load both trained EfficientNetB0 models."""

    if not BASELINE_PATH.exists():
        raise FileNotFoundError(f"Baseline model missing: {BASELINE_PATH}")

    if not CLAHE_PATH.exists():
        raise FileNotFoundError(f"CLAHE model missing: {CLAHE_PATH}")

    print("Loading Baseline EfficientNetB0...")
    baseline_model = tf.keras.models.load_model(
        BASELINE_PATH,
        compile=False
    )

    print("Loading CLAHE EfficientNetB0...")
    clahe_model = tf.keras.models.load_model(
        CLAHE_PATH,
        compile=False
    )

    print("Both models loaded successfully!")

    return baseline_model, clahe_model
