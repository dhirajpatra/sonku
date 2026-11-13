"""SQLite-backed memory: people, face templates, conversations, runtime settings.

SQLite is a free, file-based database that needs no server, survives power loss in WAL
mode and is easy to inspect or back up on a Pi. Only face *embeddings* (128 floats) are
stored, never face photos or audio.
"""
from __future__ import annotations

import os
import sqlite3
import threading
import time
from typing import Iterable

import numpy as np

from app.config import Settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS people(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  language TEXT NOT NULL DEFAULT 'en',
  summary TEXT NOT NULL DEFAULT '',
  created_at REAL NOT NULL,
  last_seen REAL,
  last_greeted REAL
);
CREATE TABLE IF NOT EXISTS faces(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  person_id INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
  embedding BLOB NOT NULL,
  created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_faces_person ON faces(person_id);
CREATE TABLE IF NOT EXISTS interactions(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  person_id INTEGER REFERENCES people(id) ON DELETE CASCADE,
  ts REAL NOT NULL,
  lang TEXT,
  user_text TEXT,
  robot_text TEXT,
  backend TEXT,
  latency_ms INTEGER,
  visual INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_inter_person ON interactions(person_id, id);
CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def dir_size(path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


class MemoryStore:
    def __init__(self, s: Settings):
        self.s = s
        s.data_dir.mkdir(parents=True, exist_ok=True)
        s.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(s.db_path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.executescript(SCHEMA)
        self._db.commit()
        self.gallery_version = 0  # bumped whenever face templates change

    # -- helpers ----------------------------------------------------------------
    def _run(self, sql: str, args: Iterable = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._db.execute(sql, tuple(args))
            self._db.commit()
            return cur

    def _all(self, sql: str, args: Iterable = ()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._db.execute(sql, tuple(args)).fetchall()]

    def _one(self, sql: str, args: Iterable = ()) -> dict | None:
        rows = self._all(sql, args)
        return rows[0] if rows else None

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # -- people -----------------------------------------------------------------
    def add_person(self, name: str, language: str = "en") -> int:
        cur = self._run("INSERT INTO people(name, language, created_at) VALUES(?,?,?)", (name.strip(), language, time.time()))
        return int(cur.lastrowid)

    def get_person(self, pid: int) -> dict | None:
        return self._one("SELECT * FROM people WHERE id=?", (pid,))

    def list_people(self) -> list[dict]:
        return self._all(
            """SELECT p.*,
                      (SELECT COUNT(*) FROM faces f WHERE f.person_id=p.id) AS face_count,
                      (SELECT MAX(created_at) FROM faces f WHERE f.person_id=p.id) AS face_updated,
                      (SELECT COUNT(*) FROM interactions i WHERE i.person_id=p.id) AS interaction_count
               FROM people p ORDER BY p.name COLLATE NOCASE"""
        )

    def rename_person(self, pid: int, name: str) -> bool:
        return self._run("UPDATE people SET name=? WHERE id=?", (name.strip(), pid)).rowcount > 0

    def set_language(self, pid: int, lang: str) -> None:
        self._run("UPDATE people SET language=? WHERE id=?", (lang, pid))

    def set_summary(self, pid: int, summary: str) -> None:
        self._run("UPDATE people SET summary=? WHERE id=?", (summary, pid))

    def touch_seen(self, pid: int) -> None:
        self._run("UPDATE people SET last_seen=? WHERE id=?", (time.time(), pid))

    def mark_greeted(self, pid: int) -> None:
        self._run("UPDATE people SET last_greeted=?, last_seen=? WHERE id=?", (time.time(), time.time(), pid))

    def delete_person(self, pid: int) -> bool:
        ok = self._run("DELETE FROM people WHERE id=?", (pid,)).rowcount > 0  # cascades to faces + interactions
        self.gallery_version += 1
        return ok

    def delete_all_people(self) -> int:
        n = self._run("DELETE FROM people").rowcount
        self._run("DELETE FROM interactions")
        self.gallery_version += 1
        return n

    # -- faces ------------------------------------------------------------------
    def add_face(self, pid: int, emb: np.ndarray) -> None:
        self._run(
            "INSERT INTO faces(person_id, embedding, created_at) VALUES(?,?,?)",
            (pid, np.asarray(emb, dtype=np.float32).tobytes(), time.time()),
        )
        self.gallery_version += 1

    def gallery(self) -> list[tuple[int, np.ndarray]]:
        rows = self._all("SELECT person_id, embedding FROM faces")
        return [(r["person_id"], np.frombuffer(r["embedding"], dtype=np.float32).copy()) for r in rows]

    def face_count(self, pid: int) -> int:
        return self._one("SELECT COUNT(*) AS n FROM faces WHERE person_id=?", (pid,))["n"]

    def newest_face_ts(self, pid: int) -> float | None:
        r = self._one("SELECT MAX(created_at) AS t FROM faces WHERE person_id=?", (pid,))
        return r["t"] if r else None

    def prune_faces(self, pid: int, keep: int) -> int:
        """Overwrite policy: keep only the `keep` newest templates for a person."""
        n = self._run(
            "DELETE FROM faces WHERE person_id=? AND id NOT IN "
            "(SELECT id FROM faces WHERE person_id=? ORDER BY created_at DESC, id DESC LIMIT ?)",
            (pid, pid, keep),
        ).rowcount
        if n:
            self.gallery_version += 1
        return n

    def purge_old_faces(self, max_age_days: int) -> int:
        if max_age_days <= 0:
            return 0
        n = self._run("DELETE FROM faces WHERE created_at < ?", (time.time() - max_age_days * 86400,)).rowcount
        if n:
            self.gallery_version += 1
        return n

    # -- interactions -----------------------------------------------------------
    def add_interaction(self, person_id, lang, user_text, robot_text, backend, latency_ms, visual=False) -> None:
        self._run(
            "INSERT INTO interactions(person_id, ts, lang, user_text, robot_text, backend, latency_ms, visual) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (person_id, time.time(), lang, user_text, robot_text, backend, int(latency_ms), int(bool(visual))),
        )

    def list_interactions(self, limit: int = 100, person_id: int | None = None) -> list[dict]:
        where, args = ("WHERE i.person_id=?", [person_id]) if person_id is not None else ("", [])
        return self._all(
            f"SELECT i.*, p.name AS person_name FROM interactions i LEFT JOIN people p ON p.id=i.person_id "
            f"{where} ORDER BY i.id DESC LIMIT ?",
            args + [limit],
        )

    def count_interactions(self, person_id: int | None = None) -> int:
        if person_id is None:
            return self._one("SELECT COUNT(*) AS n FROM interactions")["n"]
        return self._one("SELECT COUNT(*) AS n FROM interactions WHERE person_id=?", (person_id,))["n"]

    def recent_interactions(self, person_id: int, n: int) -> list[dict]:
        """The newest `n` rows for a person, oldest first (used to resume a conversation)."""
        return self._all(
            "SELECT * FROM (SELECT * FROM interactions WHERE person_id=? ORDER BY id DESC LIMIT ?) ORDER BY id ASC",
            (person_id, n),
        )

    def people_over(self, n: int) -> list[int]:
        """People with at least `n` stored rows (time for a fresh summary)."""
        return [r["person_id"] for r in self._all(
            "SELECT person_id FROM interactions WHERE person_id IS NOT NULL GROUP BY person_id HAVING COUNT(*) >= ?", (n,))]

    def older_interactions(self, person_id: int, keep_recent: int) -> list[dict]:
        """Everything except the newest `keep_recent` rows, oldest first."""
        return self._all(
            "SELECT * FROM interactions WHERE person_id=? AND id NOT IN "
            "(SELECT id FROM interactions WHERE person_id=? ORDER BY id DESC LIMIT ?) ORDER BY id ASC",
            (person_id, person_id, keep_recent),
        )

    def delete_interactions(self, ids: list[int]) -> None:
        if ids:
            with self._lock:
                self._db.executemany("DELETE FROM interactions WHERE id=?", [(i,) for i in ids])
                self._db.commit()

    def clear_interactions(self, person_id: int | None = None) -> int:
        if person_id is None:
            return self._run("DELETE FROM interactions").rowcount
        return self._run("DELETE FROM interactions WHERE person_id=?", (person_id,)).rowcount

    def prune_oldest_interactions(self, n: int) -> int:
        return self._run("DELETE FROM interactions WHERE id IN (SELECT id FROM interactions ORDER BY id ASC LIMIT ?)", (n,)).rowcount

    def people_with_interactions(self) -> list[int]:
        return [r["person_id"] for r in self._all("SELECT DISTINCT person_id FROM interactions WHERE person_id IS NOT NULL")]

    # -- runtime settings -------------------------------------------------------
    def get_settings(self) -> dict[str, str]:
        return {r["key"]: r["value"] for r in self._all("SELECT key, value FROM settings")}

    def save_setting(self, key: str, value) -> None:
        self._run("INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, str(value)))

    # -- disk usage / maintenance -----------------------------------------------
    def db_bytes(self) -> int:
        base = str(self.s.db_path)
        return sum(os.path.getsize(p) for p in (base, base + "-wal", base + "-shm") if os.path.exists(p))

    def usage(self) -> dict:
        data = dir_size(self.s.data_dir)
        limit = int(self.s.max_memory_gb * 1024**3)
        pct = 100.0 * data / limit if limit else 0.0
        status = "full" if pct >= 100 else "warning" if pct >= 80 else "ok"
        return {
            "db_bytes": self.db_bytes(), "data_bytes": data, "limit_bytes": limit,
            "percent": round(pct, 3), "status": status,
            "people": self._one("SELECT COUNT(*) AS n FROM people")["n"],
            "faces": self._one("SELECT COUNT(*) AS n FROM faces")["n"],
            "interactions": self.count_interactions(),
        }

    def vacuum(self) -> None:
        with self._lock:
            self._db.commit()
            self._db.execute("VACUUM")
