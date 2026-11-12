"""Picks the microphone and speaker automatically.

With a sound server running (PipeWire/PulseAudio, the Raspberry Pi OS default) every connected
device is visible through `pactl`: Bluetooth headsets (bluez_*), USB headsets such as the Jabra
(alsa_*usb*) and the built-in outputs. Sonku ranks them by AUDIO_PRIORITY, makes the winners the
system defaults and re-checks every few seconds, so plugging in or connecting a device just works.

Without a sound server (or AUDIO_BACKEND=alsa) it falls back to the old direct-ALSA behaviour
using AUDIO_INPUT_MATCH / AUDIO_OUTPUT_DEVICE.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import time

from app.config import Settings

log = logging.getLogger("sonku.audio")
KINDS = ("bluetooth", "usb", "other")


def classify(name: str) -> str:
    n = name.lower()
    if "bluez" in n or "bluetooth" in n:
        return "bluetooth"
    if "usb" in n:
        return "usb"
    return "other"


def parse_short(text: str) -> list[dict]:
    """Parse `pactl list short sources|sinks`, dropping monitor sources and the dummy output."""
    rows = []
    for line in text.splitlines():
        parts = line.split("\t") if "\t" in line else line.split()
        if len(parts) < 2:
            continue
        name = parts[1]
        if name.endswith(".monitor") or "auto_null" in name:
            continue
        rows.append({"name": name, "kind": classify(name), "state": parts[-1]})
    return rows


class AudioRouter:
    def __init__(self, s: Settings):
        self.s = s
        self.mode = "alsa"  # pulse (sound server) | alsa (direct hardware)
        self.source: str | None = None
        self.sink: str | None = None
        self.sources: list[dict] = []
        self.sinks: list[dict] = []
        self._last_scan = 0.0
        self._lock = threading.Lock()

    # -- helpers ----------------------------------------------------------------
    def _pactl(self, *args: str) -> str | None:
        if not shutil.which("pactl"):
            return None
        try:
            r = subprocess.run(["pactl", *args], capture_output=True, text=True, timeout=3)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return r.stdout if r.returncode == 0 else None

    def _best(self, items: list[dict]) -> dict | None:
        prio = [p for p in self.s.audio_priority.split(",") if p]
        rank = lambda d: prio.index(d["kind"]) if d["kind"] in prio else len(prio)  # noqa: E731
        return min(items, key=rank) if items else None

    def _set_default(self, kind: str, name: str) -> None:
        cur = self._pactl(f"get-default-{kind}")
        if cur is None or cur.strip() != name:
            self._pactl(f"set-default-{kind}", name)

    # -- public -----------------------------------------------------------------
    def scan(self) -> None:
        with self._lock:
            self._last_scan = time.time()
            if self.s.audio_backend == "alsa":
                self.mode, self.source, self.sink, self.sources, self.sinks = "alsa", None, None, [], []
                return
            src_txt, snk_txt = self._pactl("list", "short", "sources"), self._pactl("list", "short", "sinks")
            if src_txt is None or snk_txt is None:
                if self.mode == "pulse" or self._last_scan == 0:
                    log.warning("sound server not reachable; using direct ALSA devices")
                self.mode = "pulse" if self.s.audio_backend == "pulse" else "alsa"
                self.source, self.sink, self.sources, self.sinks = None, None, [], []
                return
            self.mode = "pulse"
            self.sources, self.sinks = parse_short(src_txt), parse_short(snk_txt)
            src, snk = self._best(self.sources), self._best(self.sinks)
            new_src, new_snk = src["name"] if src else None, snk["name"] if snk else None
            if (new_src, new_snk) != (self.source, self.sink):
                log.info("audio devices: microphone=%s (%s), speaker=%s (%s)",
                         new_src or "NONE", src["kind"] if src else "-", new_snk or "NONE", snk["kind"] if snk else "-")
            self.source, self.sink = new_src, new_snk
            if new_src:
                self._set_default("source", new_src)
            if new_snk:
                self._set_default("sink", new_snk)

    def refresh(self, force: bool = False, max_age: float = 10.0) -> None:
        if force or time.time() - self._last_scan >= max_age:
            self.scan()

    def player_cmd(self, path: str) -> list[str]:
        self.refresh()
        if self.mode == "pulse":
            for tool in ("paplay", "pw-play"):
                if shutil.which(tool):
                    return [tool, path]
            return ["aplay", "-q", "-D", "default", path]
        return ["aplay", "-q", "-D", self.s.audio_output_device, path]

    def status(self) -> dict:
        return {
            "mode": self.mode, "backend": self.s.audio_backend, "priority": self.s.audio_priority,
            "microphone": self.source, "speaker": self.sink,
            "microphones": [f"{d['name']} ({d['kind']})" for d in self.sources],
            "speakers": [f"{d['name']} ({d['kind']})" for d in self.sinks],
        }
