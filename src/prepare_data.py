"""Clean the raw VisDrone labels and save every kept object as its own PNG crop.

This one-time cleaning step runs locally. Its output, data/clean/visdrone-crops-v1.zip, is uploaded to Google Cloud Storage, and the training pipeline downloads it from there (src/ingest.py).

An object is kept if it is in one of the classes below, is at least min_side_px on each side and is not heavily occluded. VisDrone images come in sequences from the same flight, so at most max_per_sequence
objects of each class are kept per flight. The filter values are in config.yaml under "prepare".

Classes (VisDrone category in brackets):
    person       pedestrian (1), people (2)
    car_van      car (4), van (5)
    truck_bus    truck (6), bus (9)
    two_wheeler  bicycle (3), motor (10)
Tricycles (7, 8), ignored regions (0) and "others" (11) are left out.

Output: data/clean/visdrone-crops-v1.zip with crops/<crop_id>.png and labels.csv (crop_id, label, sequence, image_id, w, h, frame_w, frame_h).

"""

import shutil
import tempfile
from pathlib import Path

import pandas as pd
from PIL import Image

from src.config import ROOT, load_config
from src.download_raw import sha256

SOURCE = ROOT / "data" / "raw" / "VisDrone2019-DET-test-dev"
CLEAN = ROOT / "data" / "clean"

COLUMNS = ["x", "y", "w", "h", "score", "category", "truncation", "occlusion"]
CLASSES = {1: "person", 2: "person", 4: "car_van", 5: "car_van",
           6: "truck_bus", 9: "truck_bus", 3: "two_wheeler", 10: "two_wheeler"}


def load_labels(label_path, min_side_px, max_occlusion):
    """Return the objects in one image that are in a class and pass the size and occlusion filters."""
    labels = pd.read_csv(label_path, header=None, names=COLUMNS)
    keep = (labels["category"].isin(list(CLASSES)) & (labels["score"] == 1)
            & (labels[["w", "h"]].min(axis=1) >= min_side_px) & (labels["occlusion"] <= max_occlusion))
    return labels[keep].assign(image_id=label_path.stem)


def main():
    cfg = load_config()
    prep = cfg["prepare"]

    # 1. List every usable object
    label_paths = sorted((SOURCE / "annotations").glob("*.txt"))
    if not label_paths:
        raise SystemExit(f"No labels found in {SOURCE}. Run python -m src.download_raw first.")
    objects = pd.concat([load_labels(p, prep["min_side_px"], prep["max_occlusion"]) for p in label_paths],
                        ignore_index=True)
    objects["sequence"] = objects["image_id"].str.split("_").str[0]
    objects["label"] = objects["category"].map(CLASSES)

    # 2. Shuffle with a fixed seed, then keep the first max_per_sequence objects of each class in each sequence
    sample = (objects.sample(frac=1, random_state=cfg["seed"])
              .groupby(["sequence", "label"]).head(prep["max_per_sequence"]))

    # 3. Save each crop as a PNG, in a temp folder so OneDrive doesn't sync thousands of files
    CLEAN.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        (staging / "crops").mkdir()
        rows = []
        for image_id, boxes in sample.groupby("image_id"):
            with Image.open(SOURCE / "images" / f"{image_id}.jpg") as im:
                for box in boxes.itertuples():
                    crop_id = f"{image_id}_{box.Index}"
                    im.crop((box.x, box.y, box.x + box.w, box.y + box.h)).save(staging / "crops" / f"{crop_id}.png")
                    rows.append({"crop_id": crop_id, "label": box.label, "sequence": box.sequence,
                                 "image_id": image_id, "w": box.w, "h": box.h,
                                 "frame_w": im.width, "frame_h": im.height})
        assert len(rows) == len(sample)
        pd.DataFrame(rows).sort_values("crop_id").to_csv(staging / "labels.csv", index=False)

        # 4. Zip crops/ and labels.csv together
        archive = Path(shutil.make_archive(str(CLEAN / "visdrone-crops-v1"), "zip", root_dir=staging))

    print(f"Usable objects: {len(objects)} from {objects['sequence'].nunique()} sequences")
    print(f"Saved {len(rows)} crops to {archive.relative_to(ROOT)} ({archive.stat().st_size / 1e6:.1f} MB)")
    print("Crops per class:")
    print(sample["label"].value_counts().to_string())
    print(f"\nSHA256 (paste into config.yaml -> data.sha256):\n{sha256(archive)}")


if __name__ == "__main__":
    main()