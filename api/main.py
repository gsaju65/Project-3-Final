"""Serve the trained classifier as a web API.

Endpoints:
    GET  /         short description and a link to the docs
    GET  /health   is the API up, and which model is loaded
    POST /predict  classify one object crop sent as base64 in JSON

The model and its metadata are read from models/ (or the folder in MODEL_DIR) once at startup. Images are classified in memory and never saved or logged.
"""

import base64
import binascii
import io
import json
import os
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

from src.features import features

MODEL_DIR = Path(os.environ.get("MODEL_DIR", Path(__file__).resolve().parents[1] / "models"))
MAX_BASE64_CHARS = 2_800_000  # about a 2 MB image
MAX_SIDE_PX = 4000            # one object crop is never this big

model = joblib.load(MODEL_DIR / "model.joblib")
meta = json.loads((MODEL_DIR / "metadata.json").read_text())

app = FastAPI(title="Drone Object Classifier", version="1.0",
              description="Classifies an object cropped from a drone image as a person, car/van, truck/bus or two-wheeler.")


class PredictRequest(BaseModel):
    image_base64: str = Field(max_length=MAX_BASE64_CHARS, description="PNG or JPEG of one object crop, base64-encoded")
    frame_width: int | None = Field(None, gt=0, le=10000, description="Width in px of the full drone frame the crop came from")
    frame_height: int | None = Field(None, gt=0, le=10000, description="Height in px of the full drone frame")


class PredictResponse(BaseModel):
    label: str
    display_label: str
    probabilities: dict[str, float]
    model: dict


def decode_image(image_base64):
    """Turn the base64 string into an RGB image, or stop with a 400 error."""
    if image_base64.startswith("data:"):  # accept data URLs from browsers too
        image_base64 = image_base64.split(",", 1)[-1]
    try:
        img = Image.open(io.BytesIO(base64.b64decode(image_base64, validate=True)))
        if max(img.size) > MAX_SIDE_PX:
            raise HTTPException(400, f"The image is over {MAX_SIDE_PX} px on a side. Send a crop of one object.")
        return img.convert("RGB")
    except (binascii.Error, ValueError, OSError):
        raise HTTPException(400, "image_base64 is not a valid PNG or JPEG image")


@app.get("/")
def root():
    return {"message": "Drone Object Classifier API. POST an object crop to /predict.", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "model": meta["model_name"], "mlflow_run_id": meta["mlflow_run_id"]}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    img = decode_image(req.image_base64)
    frame_w = req.frame_width or meta["default_frame"][0]
    frame_h = req.frame_height or meta["default_frame"][1]
    x = features(img, frame_w * frame_h, meta["features"]).reshape(1, -1)

    label = str(model.predict(x)[0])  # same rule the test scores were computed with
    proba = model.predict_proba(x)[0]
    return {
        "label": label,
        "display_label": meta["display"][label],
        "probabilities": {meta["display"][str(c)]: round(float(p), 4) for c, p in zip(model.classes_, proba)},
        "model": {"name": meta["model_name"], "mlflow_run_id": meta["mlflow_run_id"],
                  "test_accuracy": meta["test_accuracy"], "test_macro_f1": meta["test_macro_f1"]},
    }