"""Text to speech: Mithi v1 Flash (offline, 11 Indian languages) with an online gTTS fallback.

`speak_stream` overlaps work: the calling thread synthesises sentence N+1 while a player
thread is still speaking sentence N, so the first sentence starts as soon as it exists.
"""
from __future__ import annotations

import logging
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from typing import Iterable

from app.audio.devices import AudioRouter
from app.config import Settings
from app.utils.lang import MITHI_LANGS, normalize_lang
from app.utils.text import clean_for_tts

log = logging.getLogger("sonku.tts")
_DONE = object()


class TTS:
    def __init__(self, s: Settings, audio: AudioRouter | None = None):
        self.s = s
        self.audio = audio or AudioRouter(s)
        self.mithi = None
        self.ready = False
        self.first_audio_ts = 0.0
        self._synth_lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._counter = 0

    def load(self) -> None:
        try:
            sys.path.insert(0, self.s.mithi_model_path)
            from mithi_tts import Mithi  # type: ignore

            self.mithi = Mithi(self.s.mithi_model_path, device="cpu")
            log.info("Mithi TTS loaded")
        except Exception as exc:
            log.error("Mithi TTS unavailable (%s)%s", exc, "; falling back to gTTS (English, online)" if self.s.tts_fallback_gtts else "")
        self.ready = self.mithi is not None or self.s.tts_fallback_gtts

    # -- synthesis --------------------------------------------------------------
    def _tmp(self, ext: str) -> str:
        self._counter += 1
        return os.path.join(tempfile.gettempdir(), f"sonku_tts_{os.getpid()}_{self._counter}.{ext}")

    def _synth(self, text: str, lang: str) -> str | None:
        text = clean_for_tts(text)
        if not text:
            return None
        lang = normalize_lang(lang)
        if self.mithi is not None and lang in MITHI_LANGS:
            try:
                path = self._tmp("wav")
                with self._synth_lock:
                    audio = self.mithi.synthesize(text, lang=lang, voice=self.s.default_voice)
                    self.mithi.save(audio, path)
                return path
            except Exception as exc:
                log.error("Mithi synthesis failed (%s)", exc)
        if self.s.tts_fallback_gtts:
            try:
                from gtts import gTTS

                mp3, wav = self._tmp("mp3"), self._tmp("wav")
                gTTS(text=text, lang=lang if lang in MITHI_LANGS else "en").save(mp3)
                subprocess.run(["sox", mp3, "-r", "48000", "-c", "2", wav], check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                os.unlink(mp3)
                return wav
            except Exception as exc:
                log.error("gTTS fallback failed (%s)", exc)
        return None

    def _play(self, path: str) -> None:
        try:
            cmd = self.audio.player_cmd(path)
            self._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            _out, err = self._proc.communicate()
            if self._proc.returncode not in (0, -15):
                log.error("playback failed (%s): %s", self._proc.args[0] if self._proc else "?", (err or b"").decode(errors="replace").strip())
        finally:
            self._proc = None
            try:
                os.unlink(path)
            except OSError:
                pass

    # -- public -----------------------------------------------------------------
    def speak_stream(self, sentences: Iterable[str], lang: str = "en") -> str:
        """Speak sentences as they arrive; returns the full text that was spoken."""
        q: queue.Queue = queue.Queue(maxsize=3)
        self.first_audio_ts = 0.0

        def player() -> None:
            while True:
                item = q.get()
                if item is _DONE:
                    return
                if not self.first_audio_ts:
                    self.first_audio_ts = time.time()
                self._play(item)

        t = threading.Thread(target=player, daemon=True, name="tts-player")
        t.start()
        spoken: list[str] = []
        try:
            for sentence in sentences:
                spoken.append(sentence)
                path = self._synth(sentence, lang)
                if path:
                    q.put(path)
        finally:
            q.put(_DONE)
            t.join(timeout=120)
        log.info("[Sonku %s]: %s", lang, " ".join(spoken))
        time.sleep(0.3)  # let the DAC settle before the microphone reopens
        return " ".join(spoken)

    def stop(self) -> None:
        proc = self._proc
        if proc and proc.poll() is None:
            proc.terminate()
