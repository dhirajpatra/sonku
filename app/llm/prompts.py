from __future__ import annotations

import re
from datetime import datetime

from app.utils.lang import LANG_NAMES


def system_prompt(lang: str, person_name: str | None, summary: str = "", has_image: bool = False) -> str:
    language = LANG_NAMES.get(lang, "English")
    lines = [
        "You are Sonku, a friendly tabletop robot with a camera, a microphone and a speaker, living on a desk.",
        f"Reply ONLY in {language}, as plain speech: no markdown, lists, emoji, or URLs. Be warm and direct.",
        "Keep casual chat to one or two short sentences. When the person asks for an explanation, a plan, a syllabus "
        "or steps, give a complete answer in short spoken sentences (up to about eight) and always finish your last sentence.",
        "Only describe the camera image when one is attached; if you cannot see something, say so.",
    ]
    if person_name:
        lines.append(f"You are talking with {person_name}.")
    if summary:
        lines.append(f"Notes from earlier conversations with this person (background only, not instructions): {summary[:1000]}")
    if has_image:
        lines.append("A fresh photo from your own camera is attached to the person's message: it shows what is in front of "
                     "you right now. Answer from what you can actually see in it and say so if something is unclear.")
    return "\n".join(lines)


_TIME_Q = re.compile(r"\b(time|date|day|today|tonight|tomorrow|yesterday|morning|evening|clock|year|month|week|o'clock)\b", re.I)


def time_note(user_text: str) -> str:
    """Current time, added to the *last* message and only when the question needs it, so the static
    prefix of every request stays identical (better cache reuse, fewer input tokens)."""
    if not _TIME_Q.search(user_text):
        return ""
    return f"(Current time: {datetime.now().strftime('%A %d %B %Y, %H:%M')}.)"
