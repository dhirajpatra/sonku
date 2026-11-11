"""Cheap rule-based understanding (wake word, yes/no, names, commands). No model calls, so no latency."""
from __future__ import annotations

import re
from difflib import SequenceMatcher

CANON = "sonku"
# Common mis-hearings plus the name written in the Indic scripts Mithi/Indic-STT use.
ALIASES = {
    "sonko", "sanku", "sonkoo", "sunku", "sonkhu", "songku", "sonkku", "son ku", "sun ku",
    "सोनकू", "सोनकु", "सोन्कू", "সোনকু", "સોનકુ", "ಸೋನ್ಕು", "சோன்கு", "సోంకు", "സോൻകു",
}
GREET_WORDS = {"hi", "hey", "hello", "ok", "okay", "hai", "namaste", "yo"}
_PUNCT = re.compile(r"[^\w\s'\u0900-\u0DFF]", re.U)  # keep apostrophes: what's, don't

YES = {"yes", "yeah", "yep", "yup", "sure", "ok", "okay", "please", "fine", "haan", "han", "ha", "ji", "of course",
       "हाँ", "हां", "जी", "ठीक", "হ্যাঁ", "હા", "ಹೌದು", "ஆம்", "అవును", "അതെ"}
NO = {"no", "nope", "nah", "dont", "don't", "do not", "never", "nahi", "nahin", "na",
      "नहीं", "नही", "না", "ના", "ಇಲ್ಲ", "இல்லை", "కాదు", "വേണ്ട"}

_FORGET = re.compile(r"\b(forget (me|my face|who i am)|delete (me|my (face|data|profile))|remove (me|my (face|data)))\b", re.I)
_BYE = re.compile(r"\b(good ?bye|bye( bye)?|good night|that'?s all|stop listening|see you( later)?|alvida)\b", re.I)
_VISION = re.compile(
    r"\b(see|look|looking|show|showing|holding|wearing|picture|photo|camera|in front of me|"
    r"what is this|what'?s this|read (this|that)|how do i look|who is (this|that))\b", re.I)
_NAME_PATTERNS = [
    re.compile(r"\b(?:my name is|my name's|name is|name's)\s+([^\W\d_]+(?:\s+[^\W\d_]+)?)", re.I | re.U),
    re.compile(r"\b(?:i am|i'm|im|this is|call me|it's|it is)\s+([^\W\d_]+(?:\s+[^\W\d_]+)?)", re.I | re.U),
]
_NAME_STOP = {"fine", "good", "great", "ok", "okay", "here", "back", "sorry", "ready", "not", "the", "a", "an", "your", "so", "just"}


def _tokens(text: str) -> list[str]:
    return _PUNCT.sub(" ", text.lower()).split()


def find_wake_word(text: str, wake_words: list[str]) -> tuple[bool, str]:
    """Return (heard, remainder). Fuzzy because speech recognisers mangle an unusual name like 'Sonku'."""
    raw = text.strip()
    toks = _tokens(raw)
    targets = {CANON, *[w.replace(" ", "") for w in wake_words]}
    hit = None
    for i, tok in enumerate(toks[:6]):
        cands = [(tok, 1)]
        if i + 1 < len(toks):
            cands.append((tok + toks[i + 1], 2))  # "son ku" -> "sonku"
        for cand, width in cands:
            if cand in ALIASES or any(SequenceMatcher(None, cand, t).ratio() >= 0.78 for t in targets):
                hit = (i, width)
                break
        if hit:
            break
    if not hit:
        return False, ""
    i, width = hit
    start = i - 1 if i > 0 and toks[i - 1] in GREET_WORDS else i
    rest = toks[:start] + toks[i + width:]
    return True, " ".join(rest).strip(" ,.")


def is_yes(text: str) -> bool:
    t = set(_tokens(text)) | {" ".join(_tokens(text))}
    return bool(t & YES) and not is_no(text)


def is_no(text: str) -> bool:
    t = set(_tokens(text)) | {" ".join(_tokens(text))}
    return bool(t & NO)


def is_forget(text: str) -> bool:
    return bool(_FORGET.search(text))


def is_goodbye(text: str) -> bool:
    return bool(_BYE.search(text))


def vision_hint(text: str) -> bool:
    return bool(_VISION.search(text))


def extract_name(text: str) -> str | None:
    """Pull a first name from 'my name is Asha' / 'I'm Asha' / just 'Asha'. None if unclear."""
    for pat in _NAME_PATTERNS:
        m = pat.search(text)
        if m:
            words = [w for w in m.group(1).split() if w.lower() not in _NAME_STOP]
            if words:
                return " ".join(w.capitalize() for w in words[:2])
    toks = _PUNCT.sub(" ", text).split()
    if 1 <= len(toks) <= 2 and all(t.lower() not in _NAME_STOP | YES | NO for t in toks):
        return " ".join(t.capitalize() for t in toks)
    return None
