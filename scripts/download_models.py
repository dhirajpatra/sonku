#!/usr/bin/env python3
"""Download the two small face models (about 37 MB total) into ./models."""
import sys
from pathlib import Path

import requests

BASE = "https://github.com/opencv/opencv_zoo/raw/main/models"
FILES = {
    "face_detection_yunet_2023mar.onnx": f"{BASE}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx": f"{BASE}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}
dest = Path(__file__).resolve().parent.parent / "models"
dest.mkdir(exist_ok=True)
failed = False
for name, url in FILES.items():
    target = dest / name
    if target.exists() and target.stat().st_size > 50_000:
        print(f"already have {name}")
        continue
    print(f"downloading {name} ...")
    try:
        r = requests.get(url, timeout=120, allow_redirects=True)
        r.raise_for_status()
        if len(r.content) < 50_000:  # a Git LFS pointer file instead of the model
            raise RuntimeError(f"got only {len(r.content)} bytes (not the model file)")
        target.write_bytes(r.content)
        print(f"  saved {target} ({len(r.content) / 1e6:.1f} MB)")
    except Exception as exc:
        failed = True
        print(f"  FAILED: {exc}\n  Download it manually from {url} and put it in {dest}/")
sys.exit(1 if failed else 0)
