"""One shared camera used by presence detection, enrolment and "what do you see?" snapshots.

Backends (CAMERA_BACKEND=auto tries them in this order):
  picamera2 - keeps the camera open, fast frames (needs `apt install python3-picamera2`)
  rpicam    - spawns rpicam-still per frame (slow, ~1 s; the old robot_main.py approach)
  opencv    - any V4L2 webcam (also handy for laptop development)
"""
from __future__ import annotations

import base64
import logging
import os
import shutil
import subprocess
import tempfile
import threading

import cv2
import numpy as np

from app.config import Settings

log = logging.getLogger("sonku.camera")


class Camera:
    def __init__(self, s: Settings):
        self.s = s
        self.backend: str | None = None
        self.fast = False
        self._cam = None
        self._lock = threading.Lock()
        self._tmp = os.path.join(tempfile.gettempdir(), "sonku_frame.jpg")

    def start(self) -> None:
        order = ["picamera2", "rpicam", "opencv"] if self.s.camera_backend == "auto" else [self.s.camera_backend]
        errors = []
        for name in order:
            try:
                getattr(self, f"_start_{name}")()
                self.backend = name
                log.info("camera ready (backend=%s, %dx%d)", name, self.s.camera_width, self.s.camera_height)
                return
            except Exception as exc:  # try the next backend
                errors.append(f"{name}: {exc}")
        raise RuntimeError("no camera backend worked: " + "; ".join(errors))

    def _start_picamera2(self) -> None:
        from libcamera import Transform  # type: ignore
        from picamera2 import Picamera2  # type: ignore

        cam = Picamera2()
        transform = Transform(hflip=1, vflip=1) if self.s.camera_rotate_180 else Transform()
        cfg = cam.create_video_configuration(
            main={"size": (self.s.camera_width, self.s.camera_height), "format": "RGB888"}, transform=transform
        )
        cam.configure(cfg)
        cam.start()
        self._cam = cam
        self.fast = True

    def _start_rpicam(self) -> None:
        if not shutil.which("rpicam-still"):
            raise RuntimeError("rpicam-still not found")
        self.fast = False

    def _start_opencv(self) -> None:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("no /dev/video0 device")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.s.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.s.camera_height)
        self._cam = cap
        self.fast = True

    def read(self) -> np.ndarray | None:
        """Return a BGR frame, or None if the camera could not deliver one."""
        with self._lock:
            try:
                if self.backend == "picamera2":
                    return self._cam.capture_array("main")  # "RGB888" is already BGR byte order for OpenCV
                if self.backend == "opencv":
                    ok, frame = self._cam.read()
                    if not ok:
                        return None
                    return cv2.rotate(frame, cv2.ROTATE_180) if self.s.camera_rotate_180 else frame
                if self.backend == "rpicam":
                    cmd = ["rpicam-still", "-n", "-t", "1", "--immediate", "-o", self._tmp,
                           "--width", str(self.s.camera_width), "--height", str(self.s.camera_height)]
                    if self.s.camera_rotate_180:
                        cmd += ["--hflip", "--vflip"]
                    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=10)
                    return cv2.imread(self._tmp)
            except Exception as exc:
                log.warning("frame capture failed: %s", exc)
        return None

    def jpeg(self, frame: np.ndarray | None = None, quality: int = 75, max_side: int = 0) -> bytes | None:
        frame = self.read() if frame is None else frame
        if frame is None:
            return None
        longest = max(frame.shape[:2])
        if max_side and longest > max_side:  # never upscale; only shrink big frames
            scale = max_side / longest
            frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes() if ok else None

    def jpeg_b64(self, quality: int = 80) -> str | None:
        data = self.jpeg(quality=quality, max_side=self.s.vision_max_side)
        return base64.b64encode(data).decode() if data else None

    def stop(self) -> None:
        with self._lock:
            try:
                if self.backend == "picamera2" and self._cam:
                    self._cam.stop()
                    self._cam.close()
                elif self.backend == "opencv" and self._cam:
                    self._cam.release()
            except Exception:
                log.exception("camera shutdown error")
            self._cam = None
