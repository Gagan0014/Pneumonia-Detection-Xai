
import numpy as np
import tensorflow as tf


def build_gradcam_model(model):
    """Build feature extractor for EfficientNetB0."""

    backbone = model.get_layer("efficientnetb0")

    feature_extractor = tf.keras.Model(
        inputs=backbone.input,
        outputs=backbone.get_layer("top_activation").output
    )

    return feature_extractor


def generate_gradcam(
    image,
    model,
    feature_extractor,
    predicted_class=None
):
    """Generate Grad-CAM heatmap for the predicted class."""

    image_tensor = tf.convert_to_tensor(
        np.expand_dims(image, axis=0),
        dtype=tf.float32
    )

    with tf.GradientTape() as tape:

        conv_outputs = feature_extractor(
            image_tensor,
            training=False
        )

        x = model.get_layer("global_pool")(conv_outputs)

        x = model.get_layer("dropout")(
            x,
            training=False
        )

        predictions = model.get_layer(
            "pneumonia_output"
        )(x)

        probability = predictions[:, 0]

        if predicted_class is None:
            predicted_class = int(
                probability.numpy()[0] >= 0.5
            )

        if predicted_class == 1:
            class_score = probability
        else:
            class_score = 1.0 - probability

    gradients = tape.gradient(
        class_score,
        conv_outputs
    )

    if gradients is None:
        raise ValueError("Grad-CAM gradients are None.")

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    feature_maps = conv_outputs[0]

    heatmap = tf.reduce_sum(
        feature_maps * pooled_gradients,
        axis=-1
    )

    heatmap = tf.maximum(heatmap, 0)

    heatmap = heatmap / (
        tf.reduce_max(heatmap) + 1e-8
    )

    return heatmap.numpy(), float(
        probability.numpy()[0]
    )
