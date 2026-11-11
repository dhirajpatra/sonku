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

    def match(self, target_embedding: np.ndarray, gallery: dict[int, list[np.ndarray]] | list) -> tuple[int | None, float]:
        """Matches target embedding against gallery items.
        
        Handles gallery passed either as:
        - dict: {person_id: [emb1, emb2, ...]}
        - list: [(person_id, emb), ...] or list of embeddings
        """
        best_score = -1.0
        best_pid = None
        threshold = getattr(self.s, "face_sim_threshold", 0.363)

        with self._lock:
            # Handle gallery passed as a list
            if isinstance(gallery, list):
                for item in gallery:
                    if isinstance(item, (tuple, list)) and len(item) == 2:
                        pid, sample = item
                    else:
                        pid, sample = None, item
                    
                    score = float(self.rec.match(target_embedding, sample, cv2.FaceRecognizerSF_FR_COSINE))
                    if score > best_score:
                        best_score = score
                        if score >= threshold:
                            best_pid = pid
            # Handle gallery passed as a dict
            elif isinstance(gallery, dict):
                for pid, embeddings in gallery.items():
                    for sample in embeddings:
                        score = float(self.rec.match(target_embedding, sample, cv2.FaceRecognizerSF_FR_COSINE))
                        if score > best_score:
                            best_score = score
                            if score >= threshold:
                                best_pid = pid

        return (best_pid, best_score)