"""Background thread that notices people and works out who they are.

Status flow:  none -> checking -> known | unknown -> none (after the person leaves)
`seq` increments each time a known person is newly confirmed, so the orchestrator can
greet exactly once per arrival.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, replace

import cv2
import numpy as np

from app.config import Settings
from app.memory.store import MemoryStore
from app.vision.camera import Camera
from app.vision.faces import FaceEngine

log = logging.getLogger("sonku.presence")


@dataclass
class Presence:
    face_present: bool = False
    status: str = "none"  # none | checking | known | unknown
    person_id: int | None = None
    name: str | None = None
    score: float = 0.0
    since: float = 0.0
    last_face_ts: float = 0.0
    seq: int = 0


class PresenceWatcher(threading.Thread):
    def __init__(self, s: Settings, camera: Camera, faces: FaceEngine, store: MemoryStore):
        super().__init__(daemon=True, name="presence")
        self.s, self.camera, self.faces, self.store = s, camera, faces, store
        self._p = Presence()
        self._plock = threading.Lock()
        self._stop_evt = threading.Event()
        self._gallery: list = []
        self._gallery_ver = -1
        self._votes: list[int | None] = []
        self._last_vote_ts = 0.0
        self.last_jpeg: bytes | None = None

    # -- public -----------------------------------------------------------------
    def snapshot(self) -> Presence:
        with self._plock:
            return replace(self._p)

    def reset_identity(self) -> None:
        with self._plock:
            self._p = Presence(seq=self._p.seq)
            self._votes.clear()

    def stop(self) -> None:
        self._stop_evt.set()

    def capture_embeddings(self, n: int = 3, spacing: float = 0.4, timeout: float = 6.0) -> list[np.ndarray]:
        """Grab fresh frames and embed the largest face in each (used for enrolment)."""
        out: list[np.ndarray] = []
        end = time.time() + timeout
        while len(out) < n and time.time() < end:
            frame = self.camera.read()
            if frame is not None:
                rows = self.faces.detect(frame)
                if rows:
                    out.append(self.faces.embed(frame, rows[0]))
            time.sleep(spacing)
        return out

    # -- thread -----------------------------------------------------------------
    def run(self) -> None:
        interval = self.s.presence_interval_sec if self.camera.fast else max(2.0, self.s.presence_interval_sec)
        log.info("presence watcher started (every %.1fs)", interval)
        while not self._stop_evt.is_set():
            t0 = time.time()
            try:
                frame = self.camera.read()
                if frame is None:
                    time.sleep(1.0)
                    continue
                self._step(frame, t0)
            except Exception:
                log.exception("presence step failed")
                time.sleep(1.0)
            self._stop_evt.wait(max(0.05, interval - (time.time() - t0)))

    def _reload_gallery(self) -> None:
        if self._gallery_ver != self.store.gallery_version:
            self._gallery = self.store.gallery()
            self._gallery_ver = self.store.gallery_version

    def _step(self, frame: np.ndarray, now: float) -> None:
        rows = self.faces.detect(frame)
        label = ""
        with self._plock:
            p = self._p
            if not rows:
                p.face_present = False
                if p.status != "none" and now - p.last_face_ts > self.s.absence_sec:
                    log.info("person left (%s)", p.name or p.status)
                    self._p = Presence(seq=p.seq)
                    self._votes.clear()
                self._publish(frame, rows, label)
                return
            p.face_present = True
            p.last_face_ts = now
            if p.status == "none":
                p.status, p.since = "checking", now
            interval = 10.0 if p.status == "known" else (1.0 if p.status == "checking" else 3.0)
            due = now - self._last_vote_ts >= interval
        if due:
            self._vote(frame, rows[0], now)
        with self._plock:
            label = self._p.name or self._p.status
        self._publish(frame, rows, label)

    def _vote(self, frame: np.ndarray, row: np.ndarray, now: float) -> None:
        self._reload_gallery()
        emb = self.faces.embed(frame, row)
        pid, score = self.faces.match(emb, self._gallery)
        self._last_vote_ts = now
        self._votes = (self._votes + [pid])[-4:]
        with self._plock:
            p = self._p
            last2 = self._votes[-2:]
            if p.status == "known":
                if len(last2) == 2 and last2[0] == last2[1] and last2[0] != p.person_id:
                    log.info("identity changed (%s -> %s)", p.person_id, last2[0])
                    p.status, p.person_id, p.name = "checking", None, None
                else:
                    p.score = score if pid == p.person_id else p.score
                    return
            if len(last2) == 2 and last2[0] is not None and last2[0] == last2[1]:
                person = self.store.get_person(last2[0])
                if person:
                    p.status, p.person_id, p.name, p.score = "known", person["id"], person["name"], score
                    p.seq += 1
                    self.store.touch_seen(person["id"])
                    log.info("recognised %s (score %.2f)", person["name"], score)
                    known_pid = person["id"]
                else:
                    return
            elif len(self._votes) >= 4 and all(v is None for v in self._votes):
                if p.status != "unknown":
                    log.info("unknown person (best score %.2f)", score)
                p.status, p.person_id, p.name, p.score = "unknown", None, None, score
                return
            else:
                return
        self._refresh_template(known_pid, emb, score)

    def _refresh_template(self, pid: int, emb: np.ndarray, score: float) -> None:
        """Monthly overwrite: once the newest template is old enough, add a fresh one and drop the oldest."""
        newest = self.store.newest_face_ts(pid)
        age_days = (time.time() - newest) / 86400 if newest else 999
        if age_days >= self.s.face_refresh_days and score >= self.s.face_match_threshold + 0.1:
            self.store.add_face(pid, emb)
            dropped = self.store.prune_faces(pid, self.s.faces_per_person)
            log.info("refreshed face template for person %s (dropped %d old)", pid, dropped)

    def _publish(self, frame: np.ndarray, rows: list, label: str) -> None:
        """Annotated preview for the admin panel."""
        try:
            img = frame.copy()
            for r in rows[:3]:
                x, y, w, h = (int(v) for v in r[:4])
                cv2.rectangle(img, (x, y), (x + w, y + h), (80, 160, 20), 2)
                if label:
                    cv2.putText(img, label, (x, max(15, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 160, 20), 2)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 60])
            if ok:
                self.last_jpeg = buf.tobytes()
        except Exception:
            pass
