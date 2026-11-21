"""Turn an image into the JSON body that POST /predict expects, for testing the API with curl.

Usage:
    python scripts/make_request.py frontend/examples/car_van_1.png --out request.json
"""

import argparse
import base64
import json
from pathlib import Path

parser = argparse.ArgumentParser(description="Write the /predict JSON body for one image.")
parser.add_argument("image", type=Path)
parser.add_argument("--out", type=Path, default=Path("request.json"))
parser.add_argument("--frame-width", type=int)
parser.add_argument("--frame-height", type=int)
args = parser.parse_args()

body = {"image_base64": base64.b64encode(args.image.read_bytes()).decode()}
if args.frame_width and args.frame_height:
    body |= {"frame_width": args.frame_width, "frame_height": args.frame_height}
args.out.write_text(json.dumps(body))
print(f"Wrote {args.out} for {args.image.name}")