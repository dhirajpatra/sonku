"""Microphone capture with a simple energy-based voice activity detector.

Opens the input stream per call (not kept open) so the robot never records its own voice
while speaking, and so ALSA handles are released between turns.
"""
from __future__ import annotations

import logging
import time
from collections import deque

import numpy as np

from app.config import Settings

log = logging.getLogger("sonku.mic")
TARGET_RATE = 16000


class Recorder:
    def __init__(self, s: Settings):
        self.s = s
        self.device: int | None = None
        self.rate = TARGET_RATE
        self.noise = 0.0
        self._last_err = 0.0
        self.last_peak = 0.0  # loudest block of the last listen() call
        self.last_thr = 0.0   # speech threshold in force during it

    def open(self) -> None:
        import sounddevice as sd

        match = self.s.audio_input_match.lower()
        devices = sd.query_devices()
        for idx, d in enumerate(devices):
            if d["max_input_channels"] > 0 and match and match in d["name"].lower():
                self.device = idx
                break
        else:
            log.warning("no input device matching '%s'; using system default", self.s.audio_input_match)
        for rate in (TARGET_RATE, int(sd.query_devices(self.device, "input")["default_samplerate"])):
            try:
                sd.check_input_settings(device=self.device, samplerate=rate, channels=1, dtype="int16")
                self.rate = rate
                break
            except Exception:
                continue
        name = sd.query_devices(self.device, "input")["name"]
        log.info("microphone: %s @ %d Hz", name, self.rate)

    def _threshold(self) -> float:
        return max(self.s.vad_min_rms, self.noise * 3.0)

    def listen(self, wait_sec: float = 2.0) -> np.ndarray | None:
        """Wait up to `wait_sec` for speech, then record until silence. Returns 16 kHz int16 or None."""
        import sounddevice as sd

        block = int(self.rate * 0.03)
        pre: deque = deque(maxlen=10)
        frames: list[np.ndarray] = []
        started, loud, quiet = False, 0, 0
        self.last_peak = 0.0
        t_start = time.time()
        silence_blocks = int(self.s.vad_silence_sec / 0.03)
        max_blocks = int(self.s.max_utterance_sec / 0.03)
        try:
            with sd.InputStream(samplerate=self.rate, channels=1, dtype="int16", blocksize=block, device=self.device) as st:
                while True:
                    data, _overflow = st.read(block)
                    x = data[:, 0]
                    rms = float(np.sqrt(np.mean(x.astype(np.float32) ** 2)))
                    thr = self._threshold()
                    self.last_peak, self.last_thr = max(self.last_peak, rms), thr
                    if not started:
                        if rms < thr:
                            self.noise = rms if self.noise == 0 else 0.95 * self.noise + 0.05 * rms
                        pre.append(x.copy())
                        loud = loud + 1 if rms > thr else 0
                        if loud >= 3:
                            started, frames, quiet = True, list(pre), 0
                        elif time.time() - t_start > wait_sec:
                            return None
                    else:
                        frames.append(x.copy())
                        quiet = quiet + 1 if rms < thr * 0.8 else 0
                        if quiet >= silence_blocks or len(frames) >= max_blocks:
                            break
        except Exception as exc:
            if time.time() - self._last_err > 60:
                log.error("microphone error: %s", exc)
                self._last_err = time.time()
            time.sleep(2.0)
            return None
        audio = np.concatenate(frames)
        if self.rate != TARGET_RATE:
            from scipy.signal import resample_poly

            audio = resample_poly(audio.astype(np.float32), TARGET_RATE, self.rate).astype(np.int16)
        return audio if len(audio) >= TARGET_RATE * 0.4 else None
