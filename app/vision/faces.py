"""Face detection + recognition with OpenCV's YuNet detector and SFace embeddings.

Both are small ONNX models that run in real time on a Pi 5 CPU with no dlib/torch.
Only the 128-float SFace embedding is kept; the photo is discarded immediately.
"""
from __future__ import annotations

import logging
import threading

import cv2
import numpy as np

from app.config import Settings

log = logging.getLogger("sonku.faces")

DETECTOR_FILE = "face_detection_yunet_2023mar.onnx"
RECOGNIZER_FILE = "face_recognition_sface_2021dec.onnx"


class FaceEngine:
    def __init__(self, s: Settings):
        self.s = s
        self._lock = threading.Lock()
        self.det = None
        self.rec = None

    def load(self) -> None:
        det_path = self.s.models_dir / DETECTOR_FILE
        rec_path = self.s.models_dir / RECOGNIZER_FILE
        for p in (det_path, rec_path):
            if not p.exists() or p.stat().st_size < 50_000:
                raise FileNotFoundError(f"{p} missing - run: python scripts/download_models.py")
        self.det = cv2.FaceDetectorYN.create(str(det_path), "", (self.s.camera_width, self.s.camera_height), 0.8, 0.3, 5000)
        self.rec = cv2.FaceRecognizerSF.create(str(rec_path), "")
        log.info("face models loaded")

    def detect(self, frame: np.ndarray) -> list[np.ndarray]:
        """Faces big enough to try, largest first. Each row: x,y,w,h, 10 landmark coords, score.

        FACE_MIN_WIDTH_PX is meant for a 640-pixel-wide frame and is scaled for other sizes. Large frames are
        searched on a smaller copy (FACE_DETECT_MAX_SIDE) to stay fast, but the rows are in full-size
        coordinates, so recognition still uses the sharp full-size pixels."""
        h, w = frame.shape[:2]
        scale, img = 1.0, frame
        if self.s.face_detect_max_side and max(h, w) > self.s.face_detect_max_side:
            scale = self.s.face_detect_max_side / max(h, w)
            img = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        ih, iw = img.shape[:2]
        with self._lock:
            self.det.setInputSize((iw, ih))
            _ret, faces = self.det.detect(img)
        if faces is None:
            return []
        if scale != 1.0:
            faces = faces.copy()
            faces[:, :14] /= scale  # box and landmarks back to full-size pixels (the score stays)
        min_w = self.s.face_min_width_px * (w / 640.0)
        rows = [f for f in faces if f[2] >= min_w]
        rows.sort(key=lambda f: float(f[2] * f[3]), reverse=True)
        return rows

    def embed(self, frame: np.ndarray, row: np.ndarray) -> np.ndarray:
        with self._lock:
            aligned = self.rec.alignCrop(frame, row)
            feat = self.rec.feature(aligned)
        v = np.asarray(feat, dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v

    def match(self, emb: np.ndarray, gallery: list[tuple[int, np.ndarray]]) -> tuple[int | None, float]:
        """Best cosine match per person; returns (person_id, score) or (None, best_score) below threshold."""
        best_pid, best = None, -1.0
        for pid, ref in gallery:
            n = float(np.linalg.norm(ref))
            score = float(np.dot(emb, ref / n)) if n else -1.0
            if score > best:
                best_pid, best = pid, score
        if best_pid is not None and best >= self.s.face_match_threshold:
            return best_pid, best
        return None, best
