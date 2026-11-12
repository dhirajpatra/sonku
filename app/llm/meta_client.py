"""Meta API client (Responses-style endpoint), non-streaming, same payload shape as the original robot."""
from __future__ import annotations

import logging

import requests

from app.config import Settings
from app.llm.prompts import time_note

log = logging.getLogger("sonku.meta")


class MetaError(RuntimeError):
    pass


class MetaClient:
    def __init__(self, s: Settings):
        self.s = s
        self.http = requests.Session()  # keep-alive saves a TLS handshake per turn
        self.totals = {"calls": 0, "empty": 0, "input": 0, "cached": 0, "output": 0, "reasoning": 0}

    @staticmethod
    def flat_prompt(system: str, history: list[tuple[str, str]], user_text: str) -> str:
        """Most stable text first (rules, person notes), then the append-only history, then the
        volatile bits (clock, new question) last, so repeated prefixes can be cached."""
        lines = [system, "", "Conversation so far:"]
        for role, text in history:
            lines.append(f"{'User' if role == 'user' else 'Sonku'}: {text}")
        note = time_note(user_text)
        if note:
            lines.append(note)
        lines += [f"User: {user_text}", "Sonku:"]
        return "\n".join(lines)

    def _account(self, data: dict) -> None:
        """Count tokens for every call, including ones that return no text (those are billed too)."""
        u = data.get("usage") or {}
        inp, out = int(u.get("input_tokens") or 0), int(u.get("output_tokens") or 0)
        cached = int((u.get("input_tokens_details") or {}).get("cached_tokens") or 0)
        reasoning = int((u.get("output_tokens_details") or {}).get("reasoning_tokens") or 0)
        t = self.totals
        t["calls"] += 1
        t["input"] += inp
        t["cached"] += cached
        t["output"] += out
        t["reasoning"] += reasoning
        log.info("meta usage: in=%d (cached %d) out=%d (reasoning %d)", inp, cached, out, reasoning)

    def complete(self, system: str, history, user_text: str, image_b64: str | None = None) -> str:
        if not self.s.meta_api_key or self.s.meta_api_key.startswith("your_"):
            raise MetaError("MODEL_API_KEY is not set")
        content = [{"type": "input_text", "text": self.flat_prompt(system, history, user_text)}]
        if image_b64:
            content.append({"type": "input_image", "image_url": f"data:image/jpeg;base64,{image_b64}"})
        payload = {
            "model": self.s.meta_model,
            "input": [{"role": "user", "content": content}],
            "max_output_tokens": self.s.max_output_tokens,
            "stream": False,
        }
        if self.s.meta_reasoning_effort not in ("", "none"):
            payload["reasoning"] = {"effort": self.s.meta_reasoning_effort}
        headers = {"Authorization": f"Bearer {self.s.meta_api_key}", "Content-Type": "application/json"}
        try:
            r = self.http.post(self.s.meta_api_url, headers=headers, json=payload,
                               timeout=(self.s.meta_connect_timeout, self.s.meta_read_timeout))
        except requests.RequestException as exc:
            raise MetaError(f"{type(exc).__name__}: {exc}") from exc
        if r.status_code != 200:
            raise MetaError(f"HTTP {r.status_code}: {r.text[:200]}")
        data = r.json()
        self._account(data)
        for item in data.get("output", []):
            if item.get("type") == "message":
                for part in item.get("content", []):
                    if part.get("type") == "output_text" and part.get("text"):
                        return part["text"]
        if data.get("output_text"):
            return data["output_text"]
        self.totals["empty"] += 1
        kinds = [i.get("type") for i in data.get("output", [])]
        raise MetaError(f"no text in response (status={data.get('status')}, "
                        f"incomplete={data.get('incomplete_details')}, output_types={kinds}); "
                        f"try a larger MAX_OUTPUT_TOKENS")
