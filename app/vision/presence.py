"""Background thread that notices people and works out who they are.

Every face in view (up to MAX_FACES) is recognised, not only the biggest one:
  people   - known people confirmed in view (two matching checks in a row), closest first
  unknown  - faces nobody recognises in the latest check
  faces    - faces in view right now
`status`/`person_id`/`name` describe the primary person (the largest known face) for older callers.
`seq` increments each time a known person is newly confirmed.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field, replace

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
    faces: int = 0
    people: list = field(default_factory=list)  # [{"id": 3, "name": "Asha"}], closest first
    unknown: int = 0
    unknown_since: float = 0.0  # when an unknown face first showed up and stayed (0 = none)


class PresenceWatcher(threading.Thread):
    MAX_FACES = 4

    def __init__(self, s: Settings, camera: Camera, faces: FaceEngine, store: MemoryStore, cache=None):
        super().__init__(daemon=True, name="presence")
        self.s, self.camera, self.faces, self.store, self.cache = s, camera, faces, store, cache
        self._p = Presence()
        self._plock = threading.Lock()
        self._stop_evt = threading.Event()
        self._gallery: list = []
        self._gallery_ver = -1
        self._seen: dict[int, dict] = {}
        self._unknown_votes = 0
        self._unknown_first = 0.0
        self._last_vote_ts = 0.0
        self._last_n = 0
        self._last_learn: dict[int, float] = {}
        self._labels: list[tuple[float, float, str]] = []
        self.last_jpeg: bytes | None = None

    # -- public -----------------------------------------------------------------
    def snapshot(self) -> Presence:
        with self._plock:
            return replace(self._p)

    def reset_identity(self) -> None:
        with self._plock:
            self._p = Presence(seq=self._p.seq, last_face_ts=self._p.last_face_ts)
            self._seen.clear()
            self._unknown_votes, self._unknown_first = 0, 0.0

    def stop(self) -> None:
        self._stop_evt.set()

    def _match(self, emb: np.ndarray) -> tuple[int | None, float]:
        pid, score = self.cache.match(emb) if self.cache else (None, -1.0)  # hot people first
        if pid is None:
            pid, score = self.faces.match(emb, self._gallery)
        return pid, score

    def capture_embeddings(self, n: int = 3, spacing: float = 0.4, timeout: float = 6.0,
                           unknown_only: bool = False) -> list[np.ndarray]:
        """Grab fresh frames and embed one face in each. With unknown_only the largest face that nobody
        recognises is used, so a known person standing next to a newcomer is never enrolled by mistake."""
        out: list[np.ndarray] = []
        end = time.time() + timeout
        self._reload_gallery()
        while len(out) < n and time.time() < end:
            frame = self.camera.read()
            if frame is not None:
                rows = self.faces.detect(frame)[: self.MAX_FACES]
                cands = []
                for row in rows:
                    emb = self.faces.embed(frame, row)
                    if unknown_only and self._match(emb)[0] is not None:
                        continue
                    cands.append((float(row[2] * row[3]), emb))
                if cands:
                    out.append(max(cands, key=lambda c: c[0])[1])
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
        with self._plock:
            p = self._p
            p.faces = len(rows)
            if not rows:
                p.face_present = False
                if (p.status != "none" or p.people) and now - p.last_face_ts > self.s.absence_sec:
                    log.info("everyone left the frame (%s)", p.name or p.status)
                    self._p = Presence(seq=p.seq, last_face_ts=p.last_face_ts)  # keep it: sessions measure absence from here
                    self._seen.clear()
                    self._unknown_votes, self._unknown_first = 0, 0.0
                self._publish(frame, rows)
                return
            p.face_present = True
            p.last_face_ts = now
            if p.status == "none":
                p.status, p.since = "checking", now
            stable_known = bool(p.people) and p.unknown == 0
            interval = 8.0 if stable_known else (2.0 if p.unknown and not p.people else 1.0)
            due = now - self._last_vote_ts >= interval or len(rows) != self._last_n
        if due:
            self._vote(frame, rows, now)
        self._publish(frame, rows)

    def _vote(self, frame: np.ndarray, rows: list, now: float) -> None:
        self._reload_gallery()
        self._last_vote_ts, self._last_n = now, len(rows)
        results = []  # (row, pid, score, emb)
        for row in rows[: self.MAX_FACES]:
            emb = self.faces.embed(frame, row)
            pid, score = self._match(emb)
            results.append((row, pid, score, emb))
        unknown_n = sum(1 for r in results if r[1] is None)
        best_unknown = max((r[2] for r in results if r[1] is None), default=0.0)
        newly: list = []
        learn: list = []
        with self._plock:
            p = self._p
            matched: dict[int, tuple] = {}
            for row, pid, score, emb in results:
                if pid is not None and (pid not in matched or score > matched[pid][0]):
                    matched[pid] = (score, emb, float(row[2] * row[3]))
            for pid in [k for k, v in self._seen.items() if not v["confirmed"] and k not in matched]:
                del self._seen[pid]  # a person must match on consecutive checks to count
            for pid, (score, emb, area) in matched.items():
                e = self._seen.get(pid)
                if e is None:
                    e = self._seen[pid] = {"hits": 0, "confirmed": False, "name": None}
                e.update(last=now, score=score, area=area)
                e["hits"] += 1
                if not e["confirmed"] and e["hits"] >= 2:
                    person = self.store.get_person(pid)
                    if person:
                        e["confirmed"], e["name"] = True, person["name"]
                        p.seq += 1
                        self.store.touch_seen(pid)
                        log.info("recognised %s (score %.2f)", person["name"], score)
                        newly.append((pid, emb, score))
                if e["confirmed"]:
                    learn.append((pid, emb, score))
            for pid in [k for k, v in self._seen.items() if v["confirmed"] and now - v["last"] > self.s.absence_sec]:
                log.info("%s left the frame", self._seen[pid]["name"])
                del self._seen[pid]
            if unknown_n:
                self._unknown_votes += 1
                if self._unknown_votes == 1:
                    self._unknown_first = now
            else:
                self._unknown_votes, self._unknown_first = 0, 0.0
            if unknown_n and (p.unknown == 0):
                log.info("%d face(s) I do not know (best score %.2f)", unknown_n, best_unknown)
            p.unknown = unknown_n
            p.unknown_since = self._unknown_first if self._unknown_votes >= 2 else 0.0
            ranked = sorted(((v["area"], k) for k, v in self._seen.items() if v["confirmed"]), reverse=True)
            p.people = [{"id": k, "name": self._seen[k]["name"]} for _a, k in ranked]
            if p.people:
                top = p.people[0]["id"]
                p.status, p.person_id, p.name, p.score = "known", top, p.people[0]["name"], self._seen[top]["score"]
            elif p.unknown_since:
                p.status, p.person_id, p.name, p.score = "unknown", None, None, best_unknown
            else:
                p.status, p.person_id, p.name = "checking", None, None
            names = {k: v["name"] for k, v in self._seen.items() if v["confirmed"]}
            self._labels = [((r[0][0] + r[0][2] / 2), (r[0][1] + r[0][3] / 2), names.get(r[1], "?")) for r in results]
        for pid, emb, score in newly:
            self._refresh_template(pid, emb, score)
        for pid, emb, score in learn:
            self._learn_template(pid, emb, score, now)

    def _learn_template(self, pid: int, emb: np.ndarray, score: float, now: float) -> None:
        """Learn new looks (angle, light, glasses): store a template when the match is real but not strong,
        at most one every 5 minutes and never beyond FACES_PER_PERSON."""
        if not (self.s.face_match_threshold + 0.05 <= score < 0.6):
            return
        if now - self._last_learn.get(pid, 0.0) < 300:
            return
        if self.store.face_count(pid) >= self.s.faces_per_person:
            return
        self.store.add_face(pid, emb)
        self._last_learn[pid] = now
        log.info("learned a new look for person %s (score %.2f)", pid, score)

    def _refresh_template(self, pid: int, emb: np.ndarray, score: float) -> None:
        """Monthly overwrite: once the newest template is old enough, add a fresh one and drop the oldest."""
        newest = self.store.newest_face_ts(pid)
        age_days = (time.time() - newest) / 86400 if newest else 999
        if age_days >= self.s.face_refresh_days and score >= self.s.face_match_threshold + 0.1:
            self.store.add_face(pid, emb)
            dropped = self.store.prune_faces(pid, self.s.faces_per_person)
            log.info("refreshed face template for person %s (dropped %d old)", pid, dropped)

    def _publish(self, frame: np.ndarray, rows: list) -> None:
        """Annotated preview for the admin panel: a box and a name on every face."""
        try:
            img = frame.copy()
            for r in rows[: self.MAX_FACES]:
                x, y, w, h = (int(v) for v in r[:4])
                cx, cy = x + w / 2, y + h / 2
                near = min(self._labels, key=lambda t: (t[0] - cx) ** 2 + (t[1] - cy) ** 2, default=None)
                label = near[2] if near and (near[0] - cx) ** 2 + (near[1] - cy) ** 2 < 150 ** 2 else ""
                cv2.rectangle(img, (x, y), (x + w, y + h), (80, 160, 20), 2)
                if label:
                    cv2.putText(img, label, (x, max(15, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 160, 20), 2)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 60])
            if ok:
                self.last_jpeg = buf.tobytes()
        except Exception:
            pass
