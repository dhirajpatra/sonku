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
        """Faces wide enough to be 'nearby', largest first. Each row: x,y,w,h, 10 landmark coords, score."""
        h, w = frame.shape[:2]
        with self._lock:
            self.det.setInputSize((w, h))
            _ret, faces = self.det.detect(frame)
        if faces is None:
            return []
        rows = [f for f in faces if f[2] >= self.s.face_min_width_px]
        rows.sort(key=lambda f: float(f[2] * f[3]), reverse=True)
        return rows

    def embed(self, frame: np.ndarray, row: np.ndarray) -> np.ndarray:
        with self._lock:
            aligned = self.rec.alignCrop(frame, row)
            feat = self.rec.feature(aligned)
        v = np.asarray(feat, dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v

    def match_face(self, target_embedding, stored_embeddings):
        """Best cosine match per person; returns (person_id, score) or (None, best_score) below threshold."""
        best_score = -1.0
        best_name = None
        
        for name, sample_embedding in stored_embeddings.items():
            # Compute Cosine Similarity between embeddings
            score = self.recognizer.match(target_embedding, sample_embedding, cv2.FaceRecognizerSF_FR_COSINE)
            if score > best_score:
                best_score = score
                if score >= 0.363:
                    best_name = name
                    return best_name, best_score
        return best_name, best_score