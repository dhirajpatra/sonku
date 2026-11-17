"""Quick decisions with rules first and the small local model only when the rules cannot tell."""
from __future__ import annotations

import logging
import re

from app.config import Settings
from app.core import intents
from app.llm.ollama_client import OllamaClient, OllamaError

log = logging.getLogger("sonku.decision")
_WEAK = re.compile(r"\b(this|that|these|those|here|there|it)\b", re.I)
_HARD = re.compile(r"\b(explain|why|how (to|do|does|did|can|could|many|much|long|far)|compare|difference|syllabus|plan|steps?|solve|calculate|code|write|essay|summar\w+|translate|recommend|advice|should i|prove|derive)\b", re.I)
_ASCII_WORD = re.compile(r"^[A-Za-z][A-Za-z'\- ]{0,28}$")


class Decision:
    def __init__(self, s: Settings, ollama: OllamaClient):
        self.s, self.ollama = s, ollama

    def needs_vision(self, text: str) -> bool:
        """Only attach a camera frame when the question is about what Sonku can see (saves seconds and bandwidth)."""
        if intents.vision_hint(text):
            return True
        words = len(text.split())
        # The small model costs seconds on a Pi: ask it only when the sentence points at something ("is that ok?")
        if not self.s.decision_use_llm or not 3 <= words <= 25 or not _WEAK.search(text):
            return False
        prompt = ("Does answering this request require looking through a camera at the room or an object? "
                  f'Answer only YES or NO.\nRequest: "{text}"')
        try:
            return self.ollama.complete(prompt, model=self.s.decision_model, max_tokens=4, timeout=4).strip().upper().startswith("YES")
        except OllamaError as exc:
            log.debug("decision model unavailable: %s", exc)
            return False

    def reasoning_effort(self, text: str, has_image: bool) -> str | None:
        """Short casual turns ("how are you") get the lightest thinking so the answer starts sooner; anything that
        looks like a real question, or a photo, keeps the configured effort. None = use the configured effort."""
        if not self.s.adaptive_effort or has_image:
            return None
        if len(text.split()) <= 8 and not _HARD.search(text):
            return "minimal"
        return None

    def extract_name(self, text: str) -> str | None:
        name = intents.extract_name(text)
        if name or not self.s.decision_use_llm:
            return name
        prompt = ("Extract the speaker's first name from this sentence. "
                  f'Reply with only the name, or NONE if there is no name.\nSentence: "{text}"')
        try:
            out = self.ollama.complete(prompt, model=self.s.decision_model, max_tokens=8, timeout=6).strip().strip(".\"'")
        except OllamaError:
            return None
        return out.title() if out and out.upper() != "NONE" and _ASCII_WORD.match(out) else None
