"""Language helpers. Script-based detection needs no model and works on the Pi."""
from __future__ import annotations

LANG_NAMES = {
    "en": "English", "hi": "Hindi", "bn": "Bengali", "gu": "Gujarati", "kn": "Kannada",
    "ml": "Malayalam", "mr": "Marathi", "or": "Odia", "pa": "Punjabi", "ta": "Tamil", "te": "Telugu",
}
# Languages Mithi v1 Flash can speak (same 11 as LANG_NAMES).
MITHI_LANGS = set(LANG_NAMES)

_SCRIPTS = [
    (0x0900, 0x097F, "hi"), (0x0980, 0x09FF, "bn"), (0x0A00, 0x0A7F, "pa"), (0x0A80, 0x0AFF, "gu"),
    (0x0B00, 0x0B7F, "or"), (0x0B80, 0x0BFF, "ta"), (0x0C00, 0x0C7F, "te"), (0x0C80, 0x0CFF, "kn"),
    (0x0D00, 0x0D7F, "ml"),
]


def detect_lang(text: str, default: str = "en") -> str:
    """Pick the dominant Indic script in `text`; Latin-only text returns `default`.

    Devanagari maps to Hindi (it cannot tell Marathi apart); the Indic STT backend
    reports a real language id when it is enabled.
    """
    counts: dict[str, int] = {}
    for ch in text:
        o = ord(ch)
        for lo, hi, code in _SCRIPTS:
            if lo <= o <= hi:
                counts[code] = counts.get(code, 0) + 1
                break
    return max(counts, key=counts.get) if counts else default


def normalize_lang(code: str | None) -> str:
    """Map any code to one Sonku can answer and speak; unknown languages become English."""
    code = (code or "en").lower().split("-")[0].split("_")[0]
    return code if code in LANG_NAMES else "en"
