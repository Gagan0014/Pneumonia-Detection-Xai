
import numpy as np

from .preprocessing import preprocess_image, apply_clahe
from .gradcam import generate_gradcam
from .visualization import create_overlay


def predict_and_explain(
    input_image,
    model_choice,
    baseline_model,
    clahe_model,
    baseline_feature_extractor,
    clahe_feature_extractor
):
    """Perform pneumonia prediction and generate Grad-CAM."""

    if input_image is None:
        raise ValueError("Please upload a chest X-ray image.")

    # Preprocess uploaded image
    image = preprocess_image(input_image)

    # Select model and preprocessing pipeline
    if model_choice == "Baseline EfficientNetB0":

        model = baseline_model
        feature_extractor = baseline_feature_extractor
        processed_image = image.astype(np.float32)

    else:

        model = clahe_model
        feature_extractor = clahe_feature_extractor
        processed_image = apply_clahe(image)

    # Prepare model input
    model_input = np.expand_dims(
        processed_image,
        axis=0
    )

    # Generate prediction
    probability = float(
        model.predict(
            model_input,
            verbose=0
        )[0][0]
    )

    predicted_class = (
        "PNEUMONIA"
        if probability >= 0.5
        else "NORMAL"
    )

    # Generate Grad-CAM
    heatmap, _ = generate_gradcam(
        processed_image,
        model,
        feature_extractor,
        predicted_class=int(probability >= 0.5)
    )

    # Generate visualizations
    colored_heatmap, overlay = create_overlay(
        processed_image,
        heatmap
    )

    # Prepare result
    result = f"""
### Prediction: {predicted_class}

- Pneumonia model score: {probability:.2%}
- Normal model score: {(1 - probability):.2%}
- Selected model: {model_choice}
- Grad-CAM: Generated successfully

**Important:** This is an experimental academic model,
not a medical diagnosis.
"""

    return (
        result,
        image,
        processed_image.astype(np.uint8),
        colored_heatmap,
        overlay
    )
