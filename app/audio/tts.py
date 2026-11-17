"""Text to speech with Google's gTTS (online) played through the sound server.

Sonku speaks every language it answers in that gTTS supports; English uses an Indian accent by default
(GTTS_TLD=co.in). `speak_stream` overlaps work: the calling thread synthesises sentence N+1 while a player
thread is still speaking sentence N, so the first sentence starts as soon as it exists.
"""
from __future__ import annotations

import logging
import os
import queue
import random
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Iterable

from app.audio.devices import AudioRouter
from app.config import Settings
from app.utils.lang import LANG_NAMES, normalize_lang
from app.utils.text import clean_for_tts, split_long

log = logging.getLogger("sonku.tts")
_DONE = object()
CUES = {
    "en": ["Hmm, let me think.", "Let me see.", "One moment."],
    "hi": ["हम्म, ज़रा सोचने दीजिए।", "एक पल।"],
}


class TTS:
    def __init__(self, s: Settings, audio: AudioRouter | None = None):
        self.s = s
        self.audio = audio or AudioRouter(s)
        self.ready = False
        self.supported: set[str] = set()
        self.first_audio_ts = 0.0
        self._proc: subprocess.Popen | None = None
        self._counter = 0
        self._warned: dict[str, float] = {}

    def load(self) -> None:
        """Check that gTTS and sox are installed; raises with a fix-it hint otherwise."""
        try:
            from gtts.lang import tts_langs
        except ImportError as exc:
            raise RuntimeError("gTTS is not installed: pip install gTTS") from exc
        if not shutil.which("sox"):
            raise RuntimeError("sox not found: sudo apt install sox libsox-fmt-mp3")
        langs = tts_langs()
        self.supported = {code for code in LANG_NAMES if code in langs}
        missing = [LANG_NAMES[c] for c in LANG_NAMES if c not in self.supported]
        if missing:
            log.warning("gTTS cannot speak %s: replies in those languages stay silent", ", ".join(missing))
        self.ready = True
        log.info("speech ready (gTTS, %d languages, English accent %s)", len(self.supported), self.s.gtts_tld)

    def _warn_once(self, key: str, msg: str, *args, every: float = 60.0) -> None:
        if time.time() - self._warned.get(key, 0.0) > every:
            self._warned[key] = time.time()
            log.error(msg, *args)

    # -- synthesis --------------------------------------------------------------
    def _tmp(self, ext: str) -> str:
        self._counter += 1
        return os.path.join(tempfile.gettempdir(), f"sonku_tts_{os.getpid()}_{self._counter}.{ext}")

    def _synth(self, text: str, lang: str) -> str | None:
        text = clean_for_tts(text)
        if not text:
            return None
        lang = normalize_lang(lang)
        if self.supported and lang not in self.supported:
            self._warn_once(f"lang-{lang}", "gTTS has no %s voice; not speaking this reply", LANG_NAMES.get(lang, lang))
            return None
        try:
            from gtts import gTTS

            mp3, wav = self._tmp("mp3"), self._tmp("wav")
            gTTS(text=text, lang=lang, tld=self.s.gtts_tld if lang == "en" else "com").save(mp3)
            subprocess.run(["sox", mp3, "-r", "48000", "-c", "2", wav], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            os.unlink(mp3)
            return wav
        except Exception as exc:
            self._warn_once("synth", "speech synthesis failed (%s); is the internet reachable?", exc)
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
    def speak_stream(self, sentences: Iterable[str], lang: str = "en", cue: bool = False) -> str:
        """Speak sentences as they arrive; returns the full text that was spoken.

        With cue=True a short "let me think" is spoken if the first sentence takes longer than
        THINKING_CUE_AFTER_SEC, so a slow model never means a long silence."""
        q: queue.Queue = queue.Queue(maxsize=3)
        self.first_audio_ts = 0.0
        cue_lock = threading.Lock()
        real_started = threading.Event()  # set once the first real sentence has arrived

        def player() -> None:
            while True:
                item = q.get()
                if item is _DONE:
                    return
                path, is_cue = item
                if not is_cue and not self.first_audio_ts:
                    self.first_audio_ts = time.time()
                self._play(path)

        def cue_worker() -> None:
            if real_started.wait(self.s.thinking_cue_after_sec):
                return
            with cue_lock:
                if real_started.is_set():
                    return
                cue_lang = lang if lang in CUES else "en"
                path = self._synth(random.choice(CUES[cue_lang]), cue_lang)
                if path and not real_started.is_set():
                    q.put((path, True))

        t = threading.Thread(target=player, daemon=True, name="tts-player")
        t.start()
        if cue and self.s.thinking_cue:
            threading.Thread(target=cue_worker, daemon=True, name="tts-cue").start()
        spoken: list[str] = []
        try:
            for sentence in sentences:
                with cue_lock:  # a cue that is mid-synthesis is queued first, so the order stays right
                    real_started.set()
                spoken.append(sentence)
                for piece in split_long(sentence):  # keep each synthesis call short
                    path = self._synth(piece, lang)
                    if path:
                        q.put((path, False))
        finally:
            real_started.set()
            q.put(_DONE)
            t.join(timeout=600)
        log.info("[Sonku %s]: %s", lang, " ".join(spoken))
        time.sleep(0.3)  # let the DAC settle before the microphone reopens
        return " ".join(spoken)

    def stop(self) -> None:
        proc = self._proc
        if proc and proc.poll() is None:
            proc.terminate()
