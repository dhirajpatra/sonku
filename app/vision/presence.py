"""Background thread that notices people and works out who they are.

Every face in view is followed as its own *track* from frame to frame, so several people are recognised at
the same time and each one is identified independently:

  new track      -> sampled every frame; identified from the average of its last few embeddings.
                    One strong match is enough; a weaker match needs two samples that agree.
  known track    -> re-checked every VERIFY_SEC; disagreeing checks send it back to "new"
                    (a different person sat down in the same spot).
  unknown track  -> a face close enough to judge that nobody recognises; re-checked every few seconds.

  people   - known people currently in view, closest (largest face) first
  unknown  - strangers currently in view (only faces large enough to judge fairly count)
  faces    - faces in view right now
`status`/`person_id`/`name` describe the primary person (the largest known face) for older callers.
`seq` increments each time a known person is newly confirmed.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
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
    unknown_since: float = 0.0  # when the earliest current stranger first showed up (0 = none)


@dataclass
class Track:
    tid: int
    cx: float
    cy: float
    w: float
    h: float
    first_seen: float
    last_seen: float
    embs: deque = field(default_factory=lambda: deque(maxlen=5))
    votes: deque = field(default_factory=lambda: deque(maxlen=4))  # recent (person_id | None, score)
    state: str = "new"  # new | known | unknown
    pid: int | None = None
    name: str | None = None
    score: float = 0.0
    mismatch: int = 0
    last_sample: float = 0.0


class PresenceWatcher(threading.Thread):
    TRACK_TTL = 1.5          # a face unseen this long is gone
    VERIFY_SEC = 2.0         # how often a known face is re-checked
    UNKNOWN_RECHECK_SEC = 2.0
    STRONG_MARGIN = 0.15     # a match this far above the threshold is accepted from a single sample

    def __init__(self, s: Settings, camera: Camera, faces: FaceEngine, store: MemoryStore, cache=None):
        super().__init__(daemon=True, name="presence")
        self.s, self.camera, self.faces, self.store, self.cache = s, camera, faces, store, cache
        self._p = Presence()
        self._plock = threading.Lock()  # guards _p; tracks belong to the watcher thread alone
        self._stop_evt = threading.Event()
        self._gallery: list = []
        self._gallery_ver = -1
        self._tracks: list[Track] = []
        self._next_tid = 1
        self._reset_flag = False
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
        self._reset_flag = True

    def stop(self) -> None:
        self._stop_evt.set()

    def _match(self, emb: np.ndarray) -> tuple[int | None, float]:
        pid, score = self.cache.match(emb) if self.cache else (None, -1.0)  # hot people first
        if pid is None:
            pid, score = self.faces.match(emb, self._gallery)
        return pid, score

    def capture_embeddings(self, n: int = 3, spacing: float = 0.4, timeout: float = 8.0,
                           unknown_only: bool = False) -> list[np.ndarray]:
        """Grab fresh frames and embed one face in each. With unknown_only the largest face that nobody
        recognises is used, so a known person standing next to a newcomer is never enrolled by mistake."""
        out: list[np.ndarray] = []
        end = time.time() + timeout
        self._reload_gallery()
        while len(out) < n and time.time() < end:
            frame = self.camera.read()
            if frame is not None:
                rows = self.faces.detect(frame)[: self.s.face_max_people]
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
        log.info("presence watcher started (every %.1fs, up to %d faces)", interval, self.s.face_max_people)
        while not self._stop_evt.is_set():
            t0 = time.time()
            busy = False
            try:
                frame = self.camera.read()
                if frame is None:
                    time.sleep(1.0)
                    continue
                busy = self._step(frame, t0)
            except Exception:
                log.exception("presence step failed")
                time.sleep(1.0)
            fast = 0.25 if (busy and self.camera.fast) else interval  # look quicker while someone is being identified
            self._stop_evt.wait(max(0.05, fast - (time.time() - t0)))

    def _reload_gallery(self) -> None:
        if self._gallery_ver != self.store.gallery_version:
            self._gallery = self.store.gallery()
            self._gallery_ver = self.store.gallery_version

    # -- tracking ---------------------------------------------------------------
    def _associate(self, rows: list, now: float) -> dict[int, Track]:
        """Match each detected face to the track it most likely belongs to (nearest, similar size)."""
        live = [t for t in self._tracks if now - t.last_seen <= self.TRACK_TTL]
        pairs = []
        for ri, r in enumerate(rows):
            cx, cy, w = float(r[0] + r[2] / 2), float(r[1] + r[3] / 2), max(float(r[2]), 1.0)
            for t in live:
                dist = math.hypot(cx - t.cx, cy - t.cy) / max(w, t.w)
                size = abs(math.log(w / max(t.w, 1.0)))
                if dist < 0.8 and size < 0.5:
                    pairs.append((dist + size, ri, t))
        pairs.sort(key=lambda x: (x[0], x[1]))
        used_rows, used_tracks, assign = set(), set(), {}
        for _cost, ri, t in pairs:
            if ri in used_rows or id(t) in used_tracks:
                continue
            assign[ri] = t
            used_rows.add(ri)
            used_tracks.add(id(t))
        return assign

    def _due(self, t: Track, now: float) -> bool:
        if t.state == "new":
            return now - t.last_sample >= (0.0 if now - t.first_seen < 6.0 else 2.0)
        return now - t.last_sample >= (self.VERIFY_SEC if t.state == "known" else self.UNKNOWN_RECHECK_SEC)

    def _step(self, frame: np.ndarray, now: float) -> bool:
        """Process one frame. Returns True while someone is still being identified (the loop then runs faster)."""
        if self._reset_flag:
            self._tracks.clear()
            self._reset_flag = False
        rows = self.faces.detect(frame)[: self.s.face_max_people]
        if not rows:
            self._tracks = [t for t in self._tracks if now - t.last_seen <= self.TRACK_TTL]
            with self._plock:
                p = self._p
                p.faces, p.face_present = 0, False
                self._summarise(p, now)
                if (p.status != "none" or p.people) and now - p.last_face_ts > self.s.absence_sec:
                    log.info("everyone left the frame (%s)", p.name or p.status)
                    self._p = Presence(seq=p.seq, last_face_ts=p.last_face_ts)  # keep it: sessions measure absence from here
                    self._tracks.clear()
            self._publish(frame, rows)
            return False

        assign = self._associate(rows, now)
        current: list[tuple[Track, np.ndarray]] = []
        for ri, r in enumerate(rows):
            cx, cy, w, h = float(r[0] + r[2] / 2), float(r[1] + r[3] / 2), float(r[2]), float(r[3])
            t = assign.get(ri)
            if t is None:
                t = Track(self._next_tid, cx, cy, w, h, first_seen=now, last_seen=now)
                self._next_tid += 1
            t.cx, t.cy, t.w, t.h, t.last_seen = cx, cy, w, h, now
            current.append((t, r))
        keep = {id(t) for t, _ in current}
        self._tracks = [t for t, _ in current] + [t for t in self._tracks if id(t) not in keep and now - t.last_seen <= self.TRACK_TTL]

        self._reload_gallery()
        frame_scale = frame.shape[1] / 640.0
        newly: list = []
        learn: list = []
        for t, r in current:
            if self._due(t, now):
                self._process(t, frame, r, now, frame_scale, newly, learn)
        with self._plock:
            p = self._p
            p.faces, p.face_present, p.last_face_ts = len(rows), True, now
            self._summarise(p, now)
        for pid, emb, score in newly:
            self._refresh_template(pid, emb, score)
        for pid, emb, score in learn:
            self._learn_template(pid, emb, score, now)
        self._labels = [(t.cx, t.cy, t.name if t.state == "known" else ("?" if t.state == "unknown" else "...")) for t, _ in current]
        self._publish(frame, rows)
        return any(t.state == "new" and now - t.first_seen < 6.0 for t, _ in current)

    def _process(self, t: Track, frame: np.ndarray, row: np.ndarray, now: float, frame_scale: float,
                 newly: list, learn: list) -> None:
        emb = self.faces.embed(frame, row)
        t.embs.append(emb)
        t.last_sample = now
        fused = np.mean(list(t.embs)[-3:], axis=0)  # averaging a few frames smooths out blur and head turns
        norm = float(np.linalg.norm(fused))
        fused = fused / norm if norm > 0 else fused
        pid, score = self._match(fused)
        thr = self.s.face_match_threshold

        if t.state == "known":
            if pid == t.pid:
                t.mismatch, t.score = 0, score
                learn.append((pid, emb, score))
            else:  # another person now, or just a poor frame: a different match counts double
                t.mismatch += 2 if pid is not None else 1
                if t.mismatch >= 3:
                    log.info("identity changed (%s -> %s)", t.name, pid)
                    t.state, t.pid, t.name, t.mismatch, t.last_sample = "new", None, None, 0, 0.0
                    t.embs.clear()
                    t.embs.append(emb)
                    t.votes.clear()
            return

        t.votes.append((pid, score))
        agree = len(t.votes) >= 2 and t.votes[-1][0] is not None and t.votes[-1][0] == t.votes[-2][0]
        strong = pid is not None and score >= thr + self.STRONG_MARGIN
        if pid is not None and (strong or agree):
            person = self.store.get_person(pid)
            if person:
                was_unknown = t.state == "unknown"
                t.state, t.pid, t.name, t.score, t.mismatch = "known", pid, person["name"], score, 0
                with self._plock:
                    self._p.seq += 1
                self.store.touch_seen(pid)
                log.info("recognised %s (score %.2f%s)", person["name"], score, ", was unknown" if was_unknown else "")
                newly.append((pid, emb, score))
                learn.append((pid, emb, score))
            return
        unknown_min_w = self.s.face_unknown_min_width_px * frame_scale
        if (t.state == "new" and now - t.first_seen >= 2.0 and len(t.votes) >= 2
                and all(v[0] is None for v in list(t.votes)[-2:]) and t.w >= unknown_min_w):
            t.state = "unknown"
            t.embs.clear()  # a re-check must start from fresh evidence, not the blurry first impression
            log.info("unknown person (best score %.2f, face %dpx wide)", score, int(t.w))

    def _summarise(self, p: Presence, now: float) -> None:
        """Fill the public snapshot from the live tracks (call with the lock held)."""
        live = [t for t in self._tracks if now - t.last_seen <= self.TRACK_TTL]
        people, seen = [], set()
        for t in sorted((t for t in live if t.state == "known"), key=lambda t: t.w * t.h, reverse=True):
            if t.pid in seen:
                continue
            seen.add(t.pid)
            people.append({"id": t.pid, "name": t.name})
        strangers = [t for t in live if t.state == "unknown"]
        p.people, p.unknown = people, len(strangers)
        p.unknown_since = min((t.first_seen for t in strangers), default=0.0)
        if people:
            top = next(t for t in live if t.state == "known" and t.pid == people[0]["id"])
            p.status, p.person_id, p.name, p.score = "known", top.pid, top.name, top.score
        elif strangers:
            p.status, p.person_id, p.name = "unknown", None, None
        else:
            p.person_id, p.name = None, None
            p.status = "checking" if (p.face_present or p.status != "none") else "none"

    # -- templates --------------------------------------------------------------
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
            for r in rows[: self.s.face_max_people]:
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
