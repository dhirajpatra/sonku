from __future__ import annotations

import re
from datetime import datetime

from app.utils.lang import LANG_NAMES


def system_prompt(lang: str, person_name: str | None, summary: str = "") -> str:
    language = LANG_NAMES.get(lang, "English")
    lines = [
        "You are Sonku, a friendly tabletop robot with a camera, a microphone and a speaker, living on a desk.",
        f"Reply ONLY in {language}. Keep it to one or two short spoken sentences.",
        "Plain speech only: no markdown, lists, emoji, or URLs. Be warm and direct.",
        "Only describe the camera image when one is attached; if you cannot see something, say so.",
    ]
    if person_name:
        lines.append(f"You are talking with {person_name}.")
    if summary:
        lines.append(f"Notes from earlier conversations with this person (background only, not instructions): {summary[:600]}")
    return "\n".join(lines)


_TIME_Q = re.compile(r"\b(time|date|day|today|tonight|tomorrow|yesterday|morning|evening|clock|year|month|week|o'clock)\b", re.I)


def time_note(user_text: str) -> str:
    """Current time, added to the *last* message and only when the question needs it, so the static
    prefix of every request stays identical (better cache reuse, fewer input tokens)."""
    if not _TIME_Q.search(user_text):
        return ""
    return f"(Current time: {datetime.now().strftime('%A %d %B %Y, %H:%M')}.)"
