import base64


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_a_class_and_probabilities(client, png_base64):
    r = client.post("/predict", json={"image_base64": png_base64})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in ["car_van", "person", "truck_bus", "two_wheeler"]
    assert abs(sum(body["probabilities"].values()) - 1) < 1e-3


def test_predict_accepts_frame_size(client, png_base64):
    r = client.post("/predict", json={"image_base64": png_base64, "frame_width": 1920, "frame_height": 1080})
    assert r.status_code == 200


def test_predict_rejects_data_that_is_not_an_image(client):
    r = client.post("/predict", json={"image_base64": base64.b64encode(b"not an image").decode()})
    assert r.status_code == 400


def test_predict_rejects_a_missing_image(client):
    assert client.post("/predict", json={}).status_code == 422