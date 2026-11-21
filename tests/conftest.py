"""Shared test setup. The real model isn't in git, so the API tests use a tiny stand-in model."""

import base64
import io
import json
import os

import numpy as np
import pytest
from PIL import Image
from sklearn.linear_model import LogisticRegression

from src.config import load_config

CLASSES = ["car_van", "person", "truck_bus", "two_wheeler"]
DISPLAY = {"car_van": "Car / van", "person": "Person", "truck_bus": "Truck / bus", "two_wheeler": "Two-wheeler"}


@pytest.fixture(scope="session")
def client(tmp_path_factory):
    """An API test client with a small model trained on random 902-number features."""
    import joblib

    model_dir = tmp_path_factory.mktemp("models")
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(40, 902)), np.repeat(CLASSES, 10)
    joblib.dump(LogisticRegression(max_iter=500).fit(X, y), model_dir / "model.joblib")
    (model_dir / "metadata.json").write_text(json.dumps({
        "model_name": "test_model", "classes": CLASSES, "display": DISPLAY,
        "features": load_config()["features"], "default_frame": [1360, 765], "mlflow_run_id": "test",
        "val_accuracy": 0.5, "test_accuracy": 0.5, "test_macro_f1": 0.5}))
    os.environ["MODEL_DIR"] = str(model_dir)  # must be set before the API module loads

    from fastapi.testclient import TestClient
    from api.main import app
    return TestClient(app)


@pytest.fixture
def png_base64():
    """A 30x60 test image, base64-encoded the way the front-end sends it."""
    buf = io.BytesIO()
    Image.new("RGB", (30, 60), (120, 90, 60)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()