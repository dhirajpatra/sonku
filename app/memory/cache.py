"""Small LRU cache of the people who were around recently (at least 3).

Each entry keeps what a conversation needs, so a returning person costs no database reads:
  - their face templates (stacked, normalised) for a fast recognition path,
  - a compact summary of everything Sonku knows about them (capped, cut at a sentence end),
  - their last few turns, to resume the conversation.
The store tells the cache when something changes, so entries never go stale.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from app.config import Settings
from app.memory.store import MemoryStore

MIN_CAPACITY = 3
COMPACT_CHARS = 600


def compact(summary: str, limit: int = COMPACT_CHARS) -> str:
    summary = (summary or "").strip()
    if len(summary) <= limit:
        return summary
    cut = summary[:limit]
    end = max(cut.rfind(". "), cut.rfind("\u0964 "), cut.rfind("? "), cut.rfind("! "))
    return cut[: end + 1] if end > limit * 0.5 else cut.rstrip()


@dataclass
class CachedPerson:
    pid: int
    name: str
    language: str
    summary: str
    embeddings: np.ndarray  # (templates, dim), rows L2-normalised
    history: list | None  # [(role, text)]; None = refresh from the database on next use
    loaded_at: float
    last_used: float


class PersonCache:
    def __init__(self, store: MemoryStore, s: Settings):
        self.store, self.s = store, s
        self._d: OrderedDict[int, CachedPerson] = OrderedDict()
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        store.add_listener(self._on_change)

    @property
    def capacity(self) -> int:
        return max(MIN_CAPACITY, int(self.s.person_cache_size))

    # -- loading ----------------------------------------------------------------
    def _history(self, pid: int) -> list:
        hist: list = []
        if self.s.resume_turns > 0:
            for r in self.store.recent_interactions(pid, self.s.resume_turns):
                hist += [("user", r["user_text"]), ("assistant", r["robot_text"])]
        return hist

    def _load(self, pid: int) -> CachedPerson | None:
        person = self.store.get_person(pid)
        if not person:
            return None
        faces = self.store.faces_for(pid)
        if faces:
            m = np.vstack(faces).astype(np.float32)
            norms = np.linalg.norm(m, axis=1, keepdims=True)
            m = m / np.where(norms == 0, 1, norms)
        else:
            m = np.zeros((0, 128), dtype=np.float32)
        now = time.time()
        return CachedPerson(pid, person["name"], person["language"], compact(person["summary"]), m, self._history(pid), now, now)

    def get(self, pid: int | None) -> CachedPerson | None:
        if pid is None:
            return None
        with self._lock:
            e = self._d.get(pid)
            if e is not None:
                self._d.move_to_end(pid)
                if e.history is None:
                    e.history = self._history(pid)
                e.last_used = time.time()
                self.hits += 1
                return e
            self.misses += 1
            e = self._load(pid)
            if e is not None:
                self._d[pid] = e
                while len(self._d) > self.capacity:
                    self._d.popitem(last=False)  # least recently used leaves
            return e

    def warm(self) -> None:
        """Preload the most recently seen people so the first greeting after a restart is instant."""
        for p in reversed(self.store.recent_people(self.capacity)):
            self.get(p["id"])
        self.hits = self.misses = 0

    # -- recognition fast path -------------------------------------------------
    def match(self, emb: np.ndarray, margin: float = 0.10) -> tuple[int | None, float]:
        """Confident match against the hot people only; anything less sure falls back to the full gallery."""
        best_pid, best = None, -1.0
        with self._lock:
            for pid, e in self._d.items():
                if e.embeddings.shape[0]:
                    score = float((e.embeddings @ emb).max())
                    if score > best:
                        best_pid, best = pid, score
        if best_pid is not None and best >= self.s.face_match_threshold + margin:
            return best_pid, best
        return None, best

    # -- invalidation -----------------------------------------------------------
    def _on_change(self, event: str, pid: int | None) -> None:
        with self._lock:
            if event == "interaction":  # only the resumed turns are stale
                for p, e in self._d.items():
                    if pid is None or p == pid:
                        e.history = None
            elif pid is None:
                self._d.clear()
            else:
                self._d.pop(pid, None)

    def stats(self) -> dict:
        with self._lock:
            return {"capacity": self.capacity, "size": len(self._d), "hits": self.hits, "misses": self.misses,
                    "people": [e.name for e in reversed(self._d.values())]}
