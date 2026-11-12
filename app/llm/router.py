"""Meta API first, local Ollama only if Meta fails. A circuit breaker skips Meta for a while after
repeated failures so a dead network does not add its timeout to every single turn."""
from __future__ import annotations

import logging
import threading
import time
from typing import Iterator

from app.config import Settings
from app.llm.meta_client import MetaClient, MetaError
from app.llm.ollama_client import OllamaClient, OllamaError
from app.utils.text import clean_for_tts, split_sentences

log = logging.getLogger("sonku.llm")


class LLMUnavailable(RuntimeError):
    pass


class LLMRouter:
    def __init__(self, s: Settings):
        self.s = s
        self.meta = MetaClient(s)
        self.ollama = OllamaClient(s)
        self.last_backend = "none"
        self.last_error = ""
        self._fails = 0
        self._skip_until = 0.0

    # -- circuit breaker --------------------------------------------------------
    def _meta_allowed(self) -> bool:
        return self.s.llm_backend == "meta_api" and time.time() >= self._skip_until

    def _meta_failed(self, exc: Exception) -> None:
        self._fails += 1
        self.last_error = f"meta: {exc}"
        log.warning("Meta API failed (%s) - falling back to Ollama", exc)
        if self._fails >= 2:
            self._skip_until = time.time() + self.s.meta_cooldown_sec
            log.warning("Meta API skipped for %ds after %d failures", self.s.meta_cooldown_sec, self._fails)

    # -- public -----------------------------------------------------------------
    def reply_stream(self, system: str, history, user_text: str, image_b64: str | None = None) -> Iterator[str]:
        """Yield the reply one sentence at a time (so TTS can start on the first one)."""
        if self._meta_allowed():
            try:
                text = clean_for_tts(self.meta.complete(system, history, user_text, image_b64))
                self._fails = 0
                self.last_backend = "meta_api"
                yield from split_sentences(text)
                return
            except MetaError as exc:
                self._meta_failed(exc)
        self.last_backend = "ollama"
        try:
            got = False
            for sent in self.ollama.stream(system, history, user_text, image_b64):
                got = True
                yield sent
            if not got:
                raise OllamaError("empty reply")
        except OllamaError as exc:
            self.last_error = f"ollama: {exc}"
            log.error("Ollama failed: %s", exc)
            raise LLMUnavailable(str(exc)) from exc

    def complete_local(self, prompt: str, max_tokens: int = 200) -> str:
        try:
            return self.ollama.complete(prompt, max_tokens=max_tokens)
        except OllamaError as exc:
            raise LLMUnavailable(str(exc)) from exc

    def warm_async(self) -> None:
        threading.Thread(target=self.ollama.warm, daemon=True, name="ollama-warm").start()

    def status(self) -> dict:
        return {
            "configured_backend": self.s.llm_backend,
            "last_backend": self.last_backend,
            "meta_model": self.s.meta_model,
            "meta_key_set": bool(self.s.meta_api_key) and not self.s.meta_api_key.startswith("your_"),
            "meta_skipped_for_sec": max(0, round(self._skip_until - time.time())),
            "ollama_model": self.s.ollama_model,
            "ollama_up": self.ollama.healthy(),
            "last_error": self.last_error,
            "meta_usage": dict(self.meta.totals),
        }
