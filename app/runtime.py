"""Builds and owns every component. Hardware failures degrade gracefully instead of crashing,
so the admin panel stays reachable to diagnose them."""
from __future__ import annotations

import logging
import time

from app import __version__
from app.audio.devices import AudioRouter
from app.audio.recorder import Recorder
from app.audio.stt import STT
from app.audio.tts import TTS
from app.config import Settings
from app.core.maintenance import Maintenance
from app.core.orchestrator import Orchestrator
from app.llm.decision import Decision
from app.llm.router import LLMRouter
from app.logging_setup import RING
from app.memory.store import MemoryStore
from app.vision.camera import Camera
from app.vision.faces import FaceEngine
from app.vision.presence import PresenceWatcher

log = logging.getLogger("sonku.runtime")


class Runtime:
    def __init__(self, s: Settings):
        self.s = s
        self.started = time.time()
        self.health: dict[str, str] = {}
        self.store = MemoryStore(s)
        self.router = LLMRouter(s)
        self.decision = Decision(s, self.router.ollama)
        self.audio = AudioRouter(s)
        self.stt = STT(s)
        self.tts: TTS | None = None
        self.recorder: Recorder | None = None
        self.camera: Camera | None = None
        self.faces: FaceEngine | None = None
        self.watcher: PresenceWatcher | None = None
        self.orch = Orchestrator(self)
        self.maint = Maintenance(self)
        self._started = False

    def _try(self, name: str, fn) -> bool:
        try:
            fn()
            self.health[name] = "ok"
            return True
        except Exception as exc:
            self.health[name] = f"unavailable: {exc}"
            log.error("%s unavailable: %s", name, exc)
            return False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        log.info("Sonku %s starting (headless=%s)", __version__, self.s.headless)
        for key, raw in self.store.get_settings().items():  # admin-panel overrides win over .env
            try:
                self.s.set_value(key, raw)
            except ValueError:
                log.warning("ignoring invalid stored setting %s=%r", key, raw)

        if self.s.headless:
            self.health.update({"camera": "headless", "microphone": "headless", "speaker": "headless"})
        else:
            def check_audio():
                self.audio.scan()
                if self.audio.mode == "pulse" and not (self.audio.source and self.audio.sink):
                    raise RuntimeError("sound server is running but no microphone and speaker were found "
                                       "(is the Bluetooth or USB device connected?)")

            self._try("audio", check_audio)
            self.tts = TTS(self.s, self.audio)
            self._try("speaker", self.tts.load)
            if not self.tts.ready:
                self.health["speaker"] = "unavailable: no TTS engine"
            self.recorder = Recorder(self.s, self.audio)
            if not self._try("microphone", self.recorder.open):
                self.recorder = None
            camera = Camera(self.s)
            faces = FaceEngine(self.s)

            def init_vision():
                camera.start()
                faces.load()

            if self._try("camera", init_vision):
                self.camera, self.faces = camera, faces
                self.watcher = PresenceWatcher(self.s, camera, faces, self.store)
                self.watcher.start()
            elif camera.backend:  # camera works but face models are missing: still allow "what do you see?"
                self.camera = camera

        self.router.warm_async()
        self.orch.start()
        self.maint.start()

    def stop(self) -> None:
        log.info("shutting down")
        for t in (self.orch, self.maint, self.watcher):
            if t:
                t.stop()
        if self.orch.is_alive():
            self.orch.join(timeout=4)  # let the mic read finish so PortAudio is not torn down mid-read
        if self.tts:
            self.tts.stop()
        if self.camera:
            self.camera.stop()
        self.store.close()

    def status(self) -> dict:
        w = self.watcher.snapshot() if self.watcher else None
        return {
            "version": __version__,
            "uptime_sec": round(time.time() - self.started),
            "health": self.health,
            "audio": self.audio.status(),
            "robot": self.orch.describe(),
            "presence": None if not w else {
                "face_present": w.face_present, "status": w.status, "name": w.name, "score": round(w.score, 3)},
            "llm": self.router.status(),
            "memory": self.store.usage(),
            "memory_alert": self.maint.alert,
            "log_errors": sum(1 for r in RING.buf if r["level"] in ("ERROR", "CRITICAL")),
        }
