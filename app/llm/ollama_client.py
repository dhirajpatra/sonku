"""Local Ollama client: streaming /api/chat with thinking off and the model kept resident."""
from __future__ import annotations

import json
import logging
from typing import Iterator

import requests

from app.config import Settings
from app.llm.prompts import time_note
from app.utils.text import SentenceChunker, clean_for_tts

log = logging.getLogger("sonku.ollama")


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, s: Settings):
        self.s = s
        self.http = requests.Session()

    def _messages(self, system: str, history: list[tuple[str, str]], user_text: str, image_b64: str | None) -> list[dict]:
        msgs = [{"role": "system", "content": system}]
        msgs += [{"role": r, "content": t} for r, t in history]
        note = time_note(user_text)
        last: dict = {"role": "user", "content": f"{user_text}\n{note}" if note else user_text}
        if image_b64 and self.s.ollama_vision:
            last["images"] = [image_b64]
        return msgs + [last]

    def _payload(self, model: str, messages: list[dict], stream: bool, num_predict: int) -> dict:
        return {
            "model": model, "messages": messages, "think": False, "stream": stream,
            "keep_alive": self.s.ollama_keep_alive,
            "options": {"num_ctx": self.s.ollama_num_ctx, "num_predict": num_predict, "temperature": 0.6},
        }

    def _http_error(self, r) -> str:
        body = r.text[:200]
        hint = f" - is the model installed? run: ollama pull {self.s.ollama_model}" if r.status_code == 404 else ""
        return f"HTTP {r.status_code}: {body}{hint}"

    def stream(self, system: str, history, user_text: str, image_b64: str | None = None) -> Iterator[str]:
        payload = self._payload(self.s.ollama_model, self._messages(system, history, user_text, image_b64), True, self.s.ollama_num_predict)
        chunker = SentenceChunker()
        try:
            with self.http.post(f"{self.s.ollama_base_url}/api/chat", json=payload, stream=True, timeout=(3, 60)) as r:
                if r.status_code != 200:
                    raise OllamaError(self._http_error(r))
                for line in r.iter_lines():
                    if not line:
                        continue
                    obj = json.loads(line)
                    if obj.get("error"):
                        raise OllamaError(obj["error"])
                    for sent in chunker.feed(obj.get("message", {}).get("content", "")):
                        yield clean_for_tts(sent)
                    if obj.get("done"):
                        break
        except (requests.RequestException, json.JSONDecodeError) as exc:
            raise OllamaError(str(exc)) from exc
        for sent in chunker.flush():
            yield clean_for_tts(sent)

    def complete(self, prompt: str, model: str | None = None, max_tokens: int = 200, timeout: float = 60.0) -> str:
        """One-shot, non-streaming completion (decisions, summaries)."""
        payload = self._payload(model or self.s.ollama_model, [{"role": "user", "content": prompt}], False, max_tokens)
        payload["options"]["temperature"] = 0.1
        try:
            r = self.http.post(f"{self.s.ollama_base_url}/api/chat", json=payload, timeout=(3, timeout))
            if r.status_code != 200:
                raise OllamaError(self._http_error(r))
            return clean_for_tts(r.json().get("message", {}).get("content", ""))
        except (requests.RequestException, ValueError) as exc:
            raise OllamaError(str(exc)) from exc

    def warm(self, model: str | None = None) -> None:
        """Load the model into RAM so the first real question is fast."""
        try:
            name = model or self.s.ollama_model
            r = self.http.post(f"{self.s.ollama_base_url}/api/chat",
                               json={"model": name, "keep_alive": self.s.ollama_keep_alive}, timeout=(3, 120))
            if r.status_code == 200:
                log.info("Ollama model %s warmed", name)
            else:
                log.error("Ollama warm-up for %s failed: %s", name, self._http_error(r))
        except requests.RequestException as exc:
            log.warning("Ollama warm-up failed: %s", exc)

    def healthy(self) -> bool:
        try:
            return self.http.get(f"{self.s.ollama_base_url}/api/tags", timeout=2).ok
        except requests.RequestException:
            return False
