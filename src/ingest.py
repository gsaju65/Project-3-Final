"""Download the cleaned VisDrone crops from Google Cloud Storage: step 1 of the training pipeline.

The zip is public, so no Google credentials are needed. Its SHA256 is checked against config.yaml so a changed file can't slip into training, then it is extracted to data/cloud/.
"""

import shutil
import urllib.request
import zipfile

from src.config import ROOT, load_config
from src.download_raw import sha256


def ingest(cfg):
    """Make sure data/cloud/ holds the crops and labels.csv named in config.yaml, and return that folder."""
    data = cfg["data"]
    out = ROOT / data["local_dir"]
    marker = out / ".sha256"
    if marker.exists() and marker.read_text() == data["sha256"]:
        print(f"Already downloaded: {out.relative_to(ROOT)}")
        return out

    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    zip_path = out / data["object"]
    print(f"Downloading {data['url']}")
    urllib.request.urlretrieve(data["url"], zip_path)

    # Stop if the file in the bucket is not byte-for-byte the one in config.yaml
    if sha256(zip_path) != data["sha256"]:
        zip_path.unlink()
        raise SystemExit("Checksum mismatch: the file in the bucket is not the one in config.yaml.")

    # Refuse any path that would land outside data/cloud
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if not (out / name).resolve().is_relative_to(out.resolve()):
                raise SystemExit(f"Unsafe path in zip: {name}")
        z.extractall(out)
    zip_path.unlink()
    marker.write_text(data["sha256"])

    n_crops = len(list((out / "crops").glob("*.png")))
    print(f"Done: {n_crops} crops and labels.csv in {out.relative_to(ROOT)}")
    return out


if __name__ == "__main__":
    ingest(load_config())