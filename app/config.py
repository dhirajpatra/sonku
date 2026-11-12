"""All configuration comes from environment variables / .env.

A small whitelist (EDITABLE) can be changed at runtime from the admin panel; those
overrides are persisted in the database and applied on top of .env at startup.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _s(key: str, default: str = "") -> str:
    v = os.getenv(key)
    return default if v is None or not v.strip() else v.strip()


def _i(key: str, default: int) -> int:
    try:
        return int(float(_s(key, str(default))))
    except ValueError:
        return default


def _f(key: str, default: float) -> float:
    try:
        return float(_s(key, str(default)))
    except ValueError:
        return default


def _b(key: str, default: bool) -> bool:
    return _s(key, "true" if default else "false").lower() in ("1", "true", "yes", "on")


def _p(key: str, default: str) -> Path:
    p = Path(os.path.expanduser(_s(key, default)))
    return p if p.is_absolute() else BASE_DIR / p


# key -> (type, label, choices)
EDITABLE: dict[str, tuple[str, str, list[str] | None]] = {
    "llm_backend": ("str", "LLM backend", ["meta_api", "ollama"]),
    "ollama_model": ("str", "Local model (Ollama tag)", None),
    "default_voice": ("str", "Voice", ["charvi", "satvik"]),
    "greeting_enabled": ("bool", "Greet known people", None),
    "greet_cooldown_min": ("float", "Greeting cooldown (minutes)", None),
    "session_timeout_sec": ("int", "Conversation timeout (seconds)", None),
    "require_face_consent": ("bool", "Ask before remembering a face", None),
    "face_match_threshold": ("float", "Face match threshold (cosine, higher is stricter)", None),
    "face_refresh_days": ("int", "Overwrite a stored face after (days)", None),
    "face_max_age_days": ("int", "Delete faces not refreshed for (days, 0 = never)", None),
    "max_memory_gb": ("float", "Memory limit (GB)", None),
    "meta_read_timeout": ("float", "Meta API timeout before fallback (seconds)", None),
    "wake_words": ("str", "Wake words (comma separated)", None),
}


@dataclass
class Settings:
    # LLM
    llm_backend: str
    meta_api_key: str
    meta_api_url: str
    meta_model: str
    meta_connect_timeout: float
    meta_read_timeout: float
    meta_cooldown_sec: int
    max_output_tokens: int
    ollama_base_url: str
    ollama_model: str
    decision_model: str
    decision_use_llm: bool
    ollama_keep_alive: str
    ollama_num_ctx: int
    ollama_num_predict: int
    ollama_vision: bool
    history_turns: int
    # speech
    stt_backend: str
    google_stt_lang: str
    whisper_cpp_bin: str
    whisper_cpp_model: str
    indic_stt_model: str
    mithi_model_path: str
    default_voice: str
    tts_fallback_gtts: bool
    audio_input_match: str
    audio_output_device: str
    vad_min_rms: float
    vad_silence_sec: float
    max_utterance_sec: float
    # vision
    camera_backend: str
    camera_width: int
    camera_height: int
    camera_rotate_180: bool
    presence_interval_sec: float
    absence_sec: float
    face_min_width_px: int
    face_match_threshold: float
    face_refresh_days: int
    face_max_age_days: int
    faces_per_person: int
    models_dir: Path
    # behaviour
    wake_words: str
    greeting_enabled: bool
    greet_cooldown_min: float
    session_timeout_sec: int
    require_face_consent: bool
    # memory
    data_dir: Path
    db_path: Path
    max_memory_gb: float
    compression_threshold_mb: int
    check_interval_sec: int
    summary_keep_recent: int
    # web / misc
    web_host: str
    web_port: int
    admin_user: str
    admin_password: str
    headless: bool
    log_level: str

    @classmethod
    def from_env(cls) -> "Settings":
        base = _s("OLLAMA_BASE_URL")
        if not base:  # accept the legacy OLLAMA_URL=http://host:11434/api/generate
            legacy = _s("OLLAMA_URL")
            if legacy:
                u = urlsplit(legacy)
                base = f"{u.scheme}://{u.netloc}"
            else:
                base = "http://localhost:11434"
        data_dir = _p("DATA_DIR", "data")
        return cls(
            llm_backend=_s("LLM_BACKEND", "meta_api").lower(),
            meta_api_key=_s("MODEL_API_KEY"),
            meta_api_url=_s("META_API_URL", "https://api.meta.ai/v1/responses"),
            meta_model=_s("META_MODEL", "muse-spark-1.3-contributor"),
            meta_connect_timeout=_f("META_CONNECT_TIMEOUT", 4.0),
            meta_read_timeout=_f("FALLBACK_TIMEOUT", 10.0),
            meta_cooldown_sec=_i("META_COOLDOWN_SEC", 60),
            max_output_tokens=_i("MAX_OUTPUT_TOKENS", 500),
            ollama_base_url=base.rstrip("/"),
            ollama_model=_s("OLLAMA_MODEL", "qwen3.5:2b-q4_K_M"),
            decision_model=_s("DECISION_MODEL", "qwen3.5:0.8b"),
            decision_use_llm=_b("DECISION_USE_LLM", True),
            ollama_keep_alive=_s("OLLAMA_KEEP_ALIVE", "30m"),
            ollama_num_ctx=_i("OLLAMA_NUM_CTX", 2048),
            ollama_num_predict=_i("OLLAMA_NUM_PREDICT", 128),
            ollama_vision=_b("OLLAMA_VISION", True),
            history_turns=_i("HISTORY_TURNS", 6),
            stt_backend=_s("STT_BACKEND", "google").lower(),
            google_stt_lang=_s("GOOGLE_STT_LANG", "en-IN"),
            whisper_cpp_bin=_s("WHISPER_CPP_BIN", "whisper-cli"),
            whisper_cpp_model=_s("WHISPER_CPP_MODEL"),
            indic_stt_model=_s("INDIC_STT_MODEL", "bodhan-ai/indic-transcribe-core"),
            mithi_model_path=_s("MITHI_MODEL_PATH", "/home/dhirajpatra/mithi-v1-flash"),
            default_voice=_s("DEFAULT_VOICE", "charvi"),
            tts_fallback_gtts=_b("TTS_FALLBACK_GTTS", True),
            audio_input_match=_s("AUDIO_INPUT_MATCH", "Jabra"),
            audio_output_device=_s("AUDIO_OUTPUT_DEVICE", "default"),
            vad_min_rms=_f("VAD_MIN_RMS", 350.0),
            vad_silence_sec=_f("VAD_SILENCE_SEC", 0.9),
            max_utterance_sec=_f("MAX_UTTERANCE_SEC", 12.0),
            camera_backend=_s("CAMERA_BACKEND", "auto").lower(),
            camera_width=_i("CAMERA_WIDTH", 640),
            camera_height=_i("CAMERA_HEIGHT", 480),
            camera_rotate_180=_b("CAMERA_ROTATE_180", True),
            presence_interval_sec=_f("PRESENCE_INTERVAL_SEC", 0.7),
            absence_sec=_f("ABSENCE_SEC", 8.0),
            face_min_width_px=_i("FACE_MIN_WIDTH_PX", 70),
            face_match_threshold=_f("FACE_MATCH_THRESHOLD", 0.363),
            face_refresh_days=_i("FACE_REFRESH_DAYS", 30),
            face_max_age_days=_i("FACE_MAX_AGE_DAYS", 90),
            faces_per_person=_i("FACES_PER_PERSON", 3),
            models_dir=_p("MODELS_DIR", "models"),
            wake_words=_s("WAKE_WORDS", "sonku"),
            greeting_enabled=_b("GREETING_ENABLED", True),
            greet_cooldown_min=_f("GREET_COOLDOWN_MIN", 10.0),
            session_timeout_sec=_i("SESSION_TIMEOUT_SEC", 45),
            require_face_consent=_b("REQUIRE_FACE_CONSENT", True),
            data_dir=data_dir,
            db_path=_p("DB_PATH", str(data_dir / "sonku.db")),
            max_memory_gb=_f("MAX_MEMORY_GB", 5.0),
            compression_threshold_mb=_i("COMPRESSION_THRESHOLD_MB", 50),
            check_interval_sec=_i("CHECK_INTERVAL_SEC", 600),
            summary_keep_recent=_i("SUMMARY_KEEP_RECENT", 10),
            web_host=_s("WEB_HOST", "0.0.0.0"),
            web_port=_i("WEB_PORT", 8000),
            admin_user=_s("ADMIN_USER", "admin"),
            admin_password=_s("ADMIN_PASSWORD"),
            headless=_b("SONKU_HEADLESS", False),
            log_level=_s("LOG_LEVEL", "INFO").upper(),
        )

    # -- runtime-editable subset ------------------------------------------------
    @property
    def wake_word_list(self) -> list[str]:
        return [w.strip().lower() for w in self.wake_words.split(",") if w.strip()]

    def set_value(self, key: str, raw: Any) -> Any:
        if key not in EDITABLE:
            raise ValueError(f"'{key}' cannot be changed at runtime")
        kind, _label, choices = EDITABLE[key]
        if kind == "bool":
            val: Any = raw if isinstance(raw, bool) else str(raw).strip().lower() in ("1", "true", "yes", "on")
        elif kind == "int":
            val = int(float(raw))
        elif kind == "float":
            val = float(raw)
        else:
            val = str(raw).strip()
        if choices and val not in choices:
            raise ValueError(f"{key} must be one of {choices}")
        if kind in ("int", "float") and val < 0:
            raise ValueError(f"{key} must not be negative")
        if key == "face_match_threshold" and not (0.1 <= val <= 0.9):
            raise ValueError("face_match_threshold must be between 0.1 and 0.9")
        if key == "max_memory_gb" and val <= 0:
            raise ValueError("max_memory_gb must be positive")
        if key == "wake_words" and not val:
            raise ValueError("at least one wake word is required")
        setattr(self, key, val)
        return val

    def editable_view(self) -> list[dict]:
        return [
            {"key": k, "type": t, "label": label, "choices": ch, "value": getattr(self, k)}
            for k, (t, label, ch) in EDITABLE.items()
        ]


assert all(f.name for f in fields(Settings))
