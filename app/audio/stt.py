"""Speech to text with pluggable backends tried in order (STT_BACKEND=indic,google ...)."""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
import wave

import numpy as np

from app.config import Settings
from app.utils.lang import detect_lang, normalize_lang

log = logging.getLogger("sonku.stt")

GOOGLE_CODES = {
    "en": "en-IN", "hi": "hi-IN", "bn": "bn-IN", "gu": "gu-IN", "kn": "kn-IN", "ml": "ml-IN",
    "mr": "mr-IN", "pa": "pa-Guru-IN", "ta": "ta-IN", "te": "te-IN",
}


def write_wav(path: str, audio: np.ndarray, rate: int = 16000) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(audio.astype(np.int16).tobytes())


class STT:
    def __init__(self, s: Settings):
        self.s = s
        self._indic_model = None

    def transcribe(self, audio: np.ndarray, hint_lang: str | None = None) -> tuple[str, str]:
        """Return (text, language_code). Empty text means nothing intelligible."""
        for name in [b.strip() for b in self.s.stt_backend.split(",") if b.strip()]:
            try:
                text, lang = getattr(self, f"_{name}")(audio, hint_lang)
                if text:
                    return text.strip(), normalize_lang(lang or detect_lang(text))
                return "", "en"
            except _NoSpeech:
                return "", "en"
            except Exception as exc:
                log.warning("STT backend '%s' failed: %s", name, exc)
        return "", "en"

    # -- backends ---------------------------------------------------------------
    def _google(self, audio, hint):
        import speech_recognition as sr

        data = sr.AudioData(audio.astype(np.int16).tobytes(), 16000, 2)
        code = GOOGLE_CODES.get(hint or "", self.s.google_stt_lang)
        try:
            text = sr.Recognizer().recognize_google(data, language=code)
        except sr.UnknownValueError:
            raise _NoSpeech()
        return text, detect_lang(text, default=(hint or "en") if code != self.s.google_stt_lang else "en")

    def _whisper_cpp(self, audio, hint):
        binary = shutil.which(self.s.whisper_cpp_bin)
        if not binary or not self.s.whisper_cpp_model:
            raise RuntimeError("whisper.cpp binary or WHISPER_CPP_MODEL not configured")
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            write_wav(path, audio)
            out = subprocess.run([binary, "-m", self.s.whisper_cpp_model, "-f", path, "-nt", "-np", "-l", "en"],
                                 capture_output=True, text=True, timeout=30, check=True).stdout
        finally:
            os.unlink(path)
        text = " ".join(line.strip() for line in out.splitlines() if line.strip())
        if not text:
            raise _NoSpeech()
        return text, "en"

    def _indic_load(self):
        from huggingface_hub import snapshot_download  # gated repo: needs `hf auth login` or HF_TOKEN

        src = self.s.indic_stt_model
        model_dir = src if os.path.isdir(src) else snapshot_download(src)
        sys.path.insert(0, model_dir)  # the model repo ships its own `indic_transcribe` package
        from indic_transcribe import IndicTranscribe  # type: ignore

        log.info("loading Indic STT from %s (first load is slow)", model_dir)
        self._indic_model = IndicTranscribe.from_pretrained(model_dir)

    def _indic(self, audio, hint):
        if self._indic_model is None:
            self._indic_load()
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            write_wav(path, audio)
            kwargs = {"lang": hint} if hint and hint != "en" else {}
            text, lid = self._indic_model.transcribe(path, return_lid=True, **kwargs)
        finally:
            os.unlink(path)
        if not text or not text.strip():
            raise _NoSpeech()
        return text, (lid or {}).get("lang") or detect_lang(text)


class _NoSpeech(Exception):
    pass
