"""Turn an object crop into the feature vector the models use.

Training (src/train.py) and the API (api/main.py) both call this, so a crop is always described the same way.

Features (902 with the settings in config.yaml): 
    log aspect ratio of the crop: people are tall and thin, buses are long
    log of crop area / frame area: how big the object looks from the drone
"""

import numpy as np
from skimage.feature import hog


def features(crop, frame_area, cfg):
    """Return the features of one PIL crop. cfg is the "features" block of config.yaml."""
    w, h = crop.size
    px = cfg["crop_px"]
    pixels = np.asarray(crop.convert("L").resize((px, px)), dtype=np.float32) / 255
    shape = hog(pixels, orientations=cfg["orientations"],
                pixels_per_cell=(cfg["pixels_per_cell"],) * 2, cells_per_block=(cfg["cells_per_block"],) * 2)
    return np.append(shape, [np.log(w / h), np.log(w * h / frame_area)]).astype(np.float32)