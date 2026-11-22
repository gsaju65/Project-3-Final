"""Drone Object Classifier: the Gradio frontend for the model API on Google Cloud Run.

Pick an example crop or upload your own. The app sends it to the API's /predict endpoint and shows the class probabilities. The API address comes from the API_URL variable in the Space settings.
"""

import base64
import io
import os
from pathlib import Path

import gradio as gr
import requests


try:
    import spaces  # installed automatically on Hugging Face ZeroGPU Spaces
except ImportError:  # running locally
    spaces = None

if spaces is not None:
    @spaces.GPU
    def _zero_gpu_placeholder():
        """ZeroGPU Spaces won't start without one @spaces.GPU function. The model runs on Cloud Run, so this app never needs a GPU and this function is never called."""

API_URL = os.environ.get("API_URL", "https://drone-clf-api-513191811205.us-east1.run.app").rstrip("/")
EXAMPLES = Path(__file__).parent / "examples"
CLASS_NAMES = {"car_van": "Car / van", "person": "Person", "truck_bus": "Truck / bus", "two_wheeler": "Two-wheeler"}
DEFAULT_FRAME = (1360, 765)  # median VisDrone frame size, from models/metadata.json


def classify(image, frame_w, frame_h):
    """Send the crop to the API and return the class probabilities and a short note."""
    if image is None:
        return None, "Pick an example or upload a crop first."
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    body = {"image_base64": base64.b64encode(buf.getvalue()).decode(),
            "frame_width": int(frame_w or DEFAULT_FRAME[0]), "frame_height": int(frame_h or DEFAULT_FRAME[1])}
    try:
        r = requests.post(f"{API_URL}/predict", json=body, timeout=60)  # first call after a quiet period takes a few seconds
        r.raise_for_status()
    except requests.RequestException as e:
        return None, f"Couldn't get a prediction from the API: {e}"
    out = r.json()
    note = (f"**{out['display_label']}**, {max(out['probabilities'].values()):.0%} sure. "
            f"The model ({out['model']['name'].upper()}) got {out['model']['test_accuracy']:.0%} of crops right "
            "on test flights it never saw.")
    return out["probabilities"], note


examples = [[str(p), CLASS_NAMES[p.stem.rsplit("_", 1)[0]]] for p in sorted(EXAMPLES.glob("*.png"))]

with gr.Blocks() as demo:
    gr.Markdown(
        "# Drone Object Classifier\n"
        "Is it a person, a car, a truck or a motorbike? Pick a crop from a drone image below, or upload your own, "
        "and the model guesses from the object's outline. The examples come from drone flights the model never trained on."
    )
    with gr.Row():
        with gr.Column():
            image = gr.Image(type="pil", label="Object crop", height=300)
            truth = gr.Textbox(label="True class (examples only)", interactive=False)
            with gr.Accordion("Original frame size", open=False):
                gr.Markdown("The model also looks at how big the object is in the full drone frame. "
                            "Leave these alone unless you know the size of the frame your crop came from.")
                frame_w = gr.Number(value=DEFAULT_FRAME[0], label="Frame width (px)", precision=0)
                frame_h = gr.Number(value=DEFAULT_FRAME[1], label="Frame height (px)", precision=0)
            button = gr.Button("Classify", variant="primary")
        with gr.Column():
            probs = gr.Label(num_top_classes=4, label="Prediction")
            note = gr.Markdown()

    gr.Examples(examples=examples, inputs=[image, truth], label="Example crops from test flights")
    gr.Markdown(
        "**How it works:** an SVM trained on HOG shape features of 4,209 crops from the VisDrone dataset. "
        "It often calls trucks and buses cars, because from straight above a small truck has about the same outline as a van. "
        f"Predictions come from the model API on Google Cloud Run ({API_URL}). Images are classified in memory and not stored. "
        "Code: [github.com/gsaju65/Project-3-Final](https://github.com/gsaju65/Project-3-Final)"
    )

    button.click(classify, inputs=[image, frame_w, frame_h], outputs=[probs, note])
    image.upload(lambda: "", inputs=None, outputs=truth)  # your own image has no known class

demo.launch()