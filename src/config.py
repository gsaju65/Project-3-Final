"""Load the pipeline settings from config.yaml at the repository root."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=ROOT / "config.yaml"):
    """Return config.yaml as a dict."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)