"""Text clean-up for speech and sentence chunking for streamed replies."""
from __future__ import annotations

import re

_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_URL = re.compile(r"https?://\S+")
_MD = re.compile(r"[*_#>`~|]+")
_EMOJI = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u200d]")
_SPACE = re.compile(r"\s+")
_END = re.compile(r"""([.!?\u0964\u0965]+["')\]]*)(\s+|$)""")


def clean_for_tts(text: str) -> str:
    text = _THINK.sub(" ", text)
    text = text.replace("</think>", " ")
    text = _URL.sub(" ", text)
    text = _EMOJI.sub("", text)
    text = _MD.sub(" ", text)
    return _SPACE.sub(" ", text).strip()


class SentenceChunker:
    """Collect streamed text and emit whole sentences (also splits on the Devanagari danda)."""

    def __init__(self, min_chars: int = 8):
        self.min_chars = min_chars
        self.buf = ""

    def feed(self, piece: str) -> list[str]:
        self.buf += piece
        out: list[str] = []
        start = 0
        for m in _END.finditer(self.buf):
            if m.group(2) == "":  # terminator at the very end: more text may follow ("3.5")
                break
            cand = self.buf[start:m.end()].strip()
            if len(cand) >= self.min_chars:
                out.append(cand)
                start = m.end()
        self.buf = self.buf[start:]
        return out

    def flush(self) -> list[str]:
        rest = self.buf.strip()
        self.buf = ""
        return [rest] if rest else []


def split_sentences(text: str) -> list[str]:
    c = SentenceChunker()
    return c.feed(text + " ") + c.flush()


_FINAL = re.compile(r"""[.!?\u0964\u0965]["')\]]*$""")


def drop_unfinished(text: str) -> str:
    """For a reply that was cut off by a token limit: remove the trailing fragment that never reached
    sentence-ending punctuation (keep it only if it is all there is)."""
    text = text.strip()
    if not text or _FINAL.search(text):
        return text
    sents = split_sentences(text)
    return " ".join(sents[:-1]) if len(sents) > 1 else text


def split_long(text: str, max_chars: int = 180) -> list[str]:
    """Split an over-long sentence at commas/spaces so a speech model never has to swallow it whole."""
    out: list[str] = []
    text = text.strip()
    while len(text) > max_chars:
        window = text[:max_chars]
        cut = max(window.rfind(", "), window.rfind("; "), window.rfind(": "))
        if cut < max_chars * 0.4:
            cut = window.rfind(" ")
        if cut <= 0:
            piece = text[:max_chars]
        else:
            piece = text[:cut + 1] if text[cut] in ",;:" else text[:cut]
        out.append(piece.strip())
        text = text[len(piece):].strip()
    if text:
        out.append(text)
    return out
