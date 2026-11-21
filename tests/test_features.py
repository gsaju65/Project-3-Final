from PIL import Image

from src.config import load_config
from src.features import features

FEATURES = load_config()["features"]


def test_feature_vector_has_902_numbers():
    assert features(Image.new("RGB", (30, 60)), 1360 * 765, FEATURES).shape == (902,)


def test_tall_objects_have_negative_log_aspect_ratio():
    # people are taller than they are wide, so log(w / h) < 0
    assert features(Image.new("RGB", (20, 60)), 1360 * 765, FEATURES)[-2] < 0