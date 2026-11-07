
import os
import gradio as gr

from src.model_loader import load_models
from src.gradcam import build_gradcam_model
from src.prediction import predict_and_explain


# ==========================================
# 1. LOAD TRAINED MODELS
# ==========================================

print("Initializing Pneumonia Detection System...")

baseline_model, clahe_model = load_models()

print("Building Grad-CAM feature extractors...")

baseline_feature_extractor = build_gradcam_model(baseline_model)
clahe_feature_extractor = build_gradcam_model(clahe_model)

print("Application initialized successfully!")


# ==========================================
# 2. PREDICTION WRAPPER
# ==========================================

def run_prediction(input_image, model_choice):

    return predict_and_explain(
        input_image,
        model_choice,
        baseline_model,
        clahe_model,
        baseline_feature_extractor,
        clahe_feature_extractor
    )


# ==========================================
# 3. PROFESSIONAL LIGHT THEME
# ==========================================

THEME = gr.themes.Base(
    primary_hue="blue",
    secondary_hue="sky",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
    radius_size="lg",
    spacing_size="md"
).set(

    body_background_fill="#F4F7FB",
    body_background_fill_dark="#F4F7FB",

    body_text_color="#20374D",
    body_text_color_dark="#20374D",
    body_text_color_subdued="#718397",
    body_text_color_subdued_dark="#718397",

    background_fill_primary="#FFFFFF",
    background_fill_primary_dark="#FFFFFF",
    background_fill_secondary="#F7FAFD",
    background_fill_secondary_dark="#F7FAFD",

    block_background_fill="#FFFFFF",
    block_background_fill_dark="#FFFFFF",
    block_border_color="#DFE8F0",
    block_border_color_dark="#DFE8F0",
    block_title_text_color="#20374D",
    block_title_text_color_dark="#20374D",
    block_label_text_color="#52677C",
    block_label_text_color_dark="#52677C",

    input_background_fill="#FFFFFF",
    input_background_fill_dark="#FFFFFF",
    input_border_color="#D8E3ED",
    input_border_color_dark="#D8E3ED",

    button_primary_background_fill="#397FA3",
    button_primary_background_fill_dark="#397FA3",
    button_primary_background_fill_hover="#2E6D8F",
    button_primary_background_fill_hover_dark="#2E6D8F",
    button_primary_text_color="#FFFFFF",
    button_primary_text_color_dark="#FFFFFF",

    border_color_primary="#DFE8F0",
    border_color_primary_dark="#DFE8F0"
)


# ==========================================
# 4. CUSTOM CSS
# ==========================================

CUSTOM_CSS = """

/* GLOBAL PAGE WIDTH */

.gradio-container {
    width: 96% !important;
    max-width: 1600px !important;
    margin: auto !important;
    padding: 28px !important;
    font-family: 'Inter', sans-serif !important;
}

main {
    width: 100% !important;
    max-width: none !important;
}

/* PROJECT HEADER */

#project-header {
    text-align: center;
    padding: 42px 20px;
    margin-bottom: 30px;

    background: linear-gradient(
        120deg,
        #FFFFFF,
        #EDF5FB
    );

    border: 1px solid #DCE8F1;
    border-radius: 18px;

    box-shadow: 0 4px 18px rgba(35, 70, 100, 0.04);
}

#project-header h1 {
    color: #193A56 !important;
    font-size: clamp(30px, 4vw, 44px) !important;
    font-weight: 800 !important;
    letter-spacing: -1px;
    margin: 0 !important;
}

#project-header .accent {
    color: #438DAE;
}

#project-header .accent-line {
    width: 60px;
    height: 4px;
    background: #438DAE;
    border-radius: 5px;
    margin: 18px auto 0;
}

/* SECTION HEADINGS */

#section-heading h2 {
    color: #193A56 !important;
    font-size: 23px !important;
    font-weight: 700 !important;
}

/* DESCRIPTIONS */

#section-description p {
    color: #718397 !important;
    font-size: 14px !important;
}

/* PREDICTION RESULT */

#prediction-output {
    background: #F7FAFD;
    border: 1px solid #DFE8F0;
    border-radius: 12px;
    padding: 22px;
    min-height: 160px;
}

#prediction-output p,
#prediction-output h3,
#prediction-output strong {
    color: #20374D !important;
}

/* VISUALIZATION CARDS */

#visual-card {
    background: #FFFFFF;
    border: 1px solid #DFE8F0;
    border-radius: 14px;
    padding: 14px;
}

/* ANALYZE BUTTON */

#analyze-button {
    margin-top: 12px;
}

/* FOOTER */

#project-footer {
    text-align: center;
    color: #8293A4;
    font-size: 13px;
    padding: 25px 10px;
    margin-top: 30px;
    border-top: 1px solid #DFE8F0;
}

/* MOBILE RESPONSIVENESS */

@media (max-width: 768px) {

    .gradio-container {
        width: 100% !important;
        padding: 12px !important;
    }

    #project-header {
        padding: 28px 12px;
    }

    #project-header h1 {
        font-size: 27px !important;
    }

}

"""


# ==========================================
# 5. GRADIO INTERFACE
# ==========================================

with gr.Blocks(
    theme=THEME,
    css=CUSTOM_CSS,
    fill_width=True
) as demo:


    # --------------------------------------
    # PROJECT HEADER
    # --------------------------------------

    gr.HTML("""
    <div id="project-header">

        <h1>
            Pneumonia Detection
            <span class="accent">& Explainable AI</span>
        </h1>

        <div class="accent-line"></div>

    </div>
    """)


    # --------------------------------------
    # IMAGE UPLOAD
    # --------------------------------------

    input_image = gr.Image(
        label="Upload Chest X-ray",
        type="numpy",
        height=350
    )


    # --------------------------------------
    # MODEL SELECTION
    # --------------------------------------

    model_choice = gr.Radio(
        choices=[
            "Baseline EfficientNetB0",
            "CLAHE EfficientNetB0"
        ],
        value="CLAHE EfficientNetB0",
        label="Select Model"
    )


    # --------------------------------------
    # ANALYZE BUTTON
    # --------------------------------------

    predict_button = gr.Button(
        "Analyze X-ray",
        variant="primary",
        size="lg",
        elem_id="analyze-button"
    )


    # --------------------------------------
    # PREDICTION RESULTS
    # --------------------------------------

    gr.Markdown(
        "## Prediction Results",
        elem_id="section-heading"
    )

    prediction_output = gr.Markdown(
        value="""
### Awaiting Analysis

Upload a chest X-ray and click Analyze X-ray
to view the prediction.
""",
        elem_id="prediction-output"
    )


    # --------------------------------------
    # EXPLAINABLE AI
    # --------------------------------------

    gr.Markdown(
        "## Explainable AI Visualization",
        elem_id="section-heading"
    )

    gr.Markdown(
        "Visual representation of image processing and model attention regions.",
        elem_id="section-description"
    )


    # ORIGINAL AND PROCESSED IMAGES

    with gr.Row():

        with gr.Column(elem_id="visual-card"):

            original_output = gr.Image(
                label="Original X-ray",
                height=350
            )

        with gr.Column(elem_id="visual-card"):

            processed_output = gr.Image(
                label="Processed X-ray",
                height=350
            )


    # GRAD-CAM OUTPUTS

    with gr.Row():

        with gr.Column(elem_id="visual-card"):

            heatmap_output = gr.Image(
                label="Grad-CAM Heatmap",
                height=350
            )

        with gr.Column(elem_id="visual-card"):

            overlay_output = gr.Image(
                label="Grad-CAM Overlay",
                height=350
            )


    # --------------------------------------
    # PREDICTION EVENT
    # --------------------------------------

    predict_button.click(
        fn=run_prediction,
        inputs=[
            input_image,
            model_choice
        ],
        outputs=[
            prediction_output,
            original_output,
            processed_output,
            heatmap_output,
            overlay_output
        ]
    )


    # --------------------------------------
    # FOOTER
    # --------------------------------------

    gr.HTML("""
    <div id="project-footer">

        <strong>Pneumonia Detection & Explainable AI</strong>

        <br><br>

        EfficientNetB0 · CLAHE · Grad-CAM

        <br>
        Academic Research Project

        <br><br>
        For research and educational demonstration only.
        Not intended for clinical diagnosis.

    </div>
    """)


# ==========================================
# 6. START SERVER
# ==========================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False
    )
