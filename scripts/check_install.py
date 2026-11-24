#!/usr/bin/env python3
"""One-shot health check of the installation: code version, .env, libraries, models, service, Ollama.

Run on the Pi:  ~/robot_env/bin/python scripts/check_install.py
Prints PASS / WARN / FAIL per line (never prints secrets) - paste the whole output when asking for help.
"""
import collections
import importlib
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
counts = collections.Counter()


def out(level: str, msg: str) -> None:
    counts[level] += 1
    print(f"{level:5} {msg}")


def section(name: str) -> None:
    print(f"\n== {name}")


# ---------------------------------------------------------------- libraries and tools
section("python libraries and tools")
out("PASS" if sys.version_info >= (3, 10) else "FAIL", f"python {sys.version.split()[0]}")
for mod, needed in [("fastapi", True), ("uvicorn", True), ("numpy", True), ("scipy", True), ("cv2", True), ("requests", True),
                    ("dotenv", True), ("sounddevice", True), ("speech_recognition", True), ("gtts", True), ("picamera2", False)]:
    try:
        importlib.import_module(mod)
        out("PASS", f"import {mod}")
    except Exception as exc:
        out("FAIL" if needed else "WARN", f"import {mod}: {exc}" + ("" if needed else " (needed for the Pi camera; apt install python3-picamera2, venv with --system-site-packages)"))
try:
    import cv2
    out("PASS" if hasattr(cv2, "FaceDetectorYN") and hasattr(cv2, "FaceRecognizerSF") else "FAIL", f"OpenCV {cv2.__version__} has the face detector and recogniser (needs >= 4.8)")
except Exception:
    pass
for tool, needed, hint in [("sox", True, "sudo apt install sox libsox-fmt-mp3"), ("paplay", False, "sound server playback"),
                           ("pactl", False, "sound server (PipeWire) control"), ("aplay", False, "ALSA playback"), ("rpicam-still", False, "Pi camera tools")]:
    out("PASS" if shutil.which(tool) else ("FAIL" if needed else "WARN"), f"{tool} {'found' if shutil.which(tool) else 'missing - ' + hint}")

# ---------------------------------------------------------------- code version
section("code version (features of the latest release)")
checks = [
    ("app.vision.presence", "Track", "per-face tracking (several people at once)"),
    ("app.memory.cache", "PersonCache", "person cache (>= 3 people)"),
    ("app.audio.devices", "AudioRouter", "Bluetooth / USB / built-in audio detection"),
    ("app.core.orchestrator", "Orchestrator", "conversation loop"),
]
for mod, attr, what in checks:
    try:
        m = importlib.import_module(mod)
        out("PASS" if hasattr(m, attr) else "FAIL", f"{what}: {mod}.{attr}")
    except Exception as exc:
        out("FAIL", f"{what}: cannot import {mod} ({exc})")
try:
    from app.config import Settings
    s = Settings.from_env()
    need = ["face_max_people", "enroll_templates", "face_unknown_min_width_px", "unknown_face_mode", "proactive_enabled",
            "person_cache_size", "gtts_tld", "meta_reasoning_effort", "patience_sec", "audio_priority"]
    missing = [n for n in need if not hasattr(s, n)]
    out("PASS" if not missing else "FAIL", "config.py has the latest settings" if not missing else f"config.py is OLD, missing: {missing}")
    from app.core.orchestrator import Orchestrator
    for fn in ("_maybe_greet", "_maybe_meet_stranger", "_maybe_proactive"):
        out("PASS" if hasattr(Orchestrator, fn) else "FAIL", f"orchestrator.{fn}")
    tts_src = (ROOT / "app/audio/tts.py").read_text()
    out("PASS" if "mithi" not in tts_src.lower() else "WARN", "tts.py is gTTS only" if "mithi" not in tts_src.lower() else "tts.py still mentions Mithi (old version)")
except Exception as exc:
    out("FAIL", f"cannot load settings/code: {type(exc).__name__}: {exc}")
    s = None

# ---------------------------------------------------------------- .env
section(".env")
env_path = ROOT / ".env"
if not env_path.exists():
    out("FAIL", f"{env_path} missing (cp .env.example .env)")
else:
    lines = env_path.read_text().splitlines()
    assigns = [(l.split("=", 1)[0], l.split("=", 1)[1]) for l in lines if "=" in l and not l.lstrip().startswith("#")]
    dup = [k for k, n in collections.Counter(k for k, _ in assigns).items() if n > 1]
    out("WARN" if dup else "PASS", f"duplicate keys: {dup}" if dup else "no duplicate keys (the last one would win anyway)")
    inline = [k for k, v in assigns if re.search(r"\s#", v)]
    out("FAIL" if inline else "PASS", f"inline comments (python-dotenv strips them, systemd does not): {inline}" if inline else "no inline comments")
    vals = dict(assigns)
    mode = oct(env_path.stat().st_mode & 0o777)
    out("PASS" if mode == "0o600" else "WARN", f".env permissions {mode}" + ("" if mode == "0o600" else " - run: chmod 600 .env"))
    out("WARN" if vals.get("ADMIN_PASSWORD", "") in ("", "admin", "password", "1234", "test") else "PASS",
        "ADMIN_PASSWORD is empty or a weak default - the panel shows faces and conversations" if vals.get("ADMIN_PASSWORD", "") in ("", "admin", "password", "1234", "test") else "ADMIN_PASSWORD is set")
    key = vals.get("MODEL_API_KEY", "")
    out("PASS" if key and not key.startswith("your_") else "FAIL", "MODEL_API_KEY is set" if key and not key.startswith("your_") else "MODEL_API_KEY missing or placeholder")
    for stale in ("MITHI_MODEL_PATH", "DEFAULT_VOICE", "TTS_FALLBACK_GTTS"):
        if stale in vals:
            out("WARN", f"{stale} is no longer used (Mithi was removed)")
    if "USERNAME" in vals or "PASSWORD" in vals:
        out("WARN", "USERNAME/PASSWORD are not used by Sonku; a plain-text account password does not belong in this file")
    if s:
        notes = []
        if s.vad_min_rms > 600:
            notes.append(f"VAD_MIN_RMS={s.vad_min_rms:g} is high: quiet speech may never start a recording (try 400-500)")
        if s.faces_per_person < s.enroll_templates:
            notes.append("FACES_PER_PERSON is below ENROLL_TEMPLATES: new looks can never be learned")
        if s.ollama_num_predict < 200:
            notes.append(f"OLLAMA_NUM_PREDICT={s.ollama_num_predict} cuts local answers short")
        if s.max_output_tokens < 800:
            notes.append(f"MAX_OUTPUT_TOKENS={s.max_output_tokens} may cut Meta answers (reasoning shares this budget)")
        if s.meta_reasoning_effort in ("high", "xhigh"):
            notes.append(f"META_REASONING_EFFORT={s.meta_reasoning_effort}: best quality, slowest first sentence (5-10 s)")
        for n in notes:
            out("WARN", n)
        if not notes:
            out("PASS", "recognition, speech and token settings look sensible")
        print(f"      effective: model={s.meta_model} effort={s.meta_reasoning_effort} stranger_mode={s.unknown_face_mode} "
              f"faces(min {s.face_min_width_px}px, max {s.face_max_people}, per person {s.faces_per_person}) accent={s.gtts_tld}")

# ---------------------------------------------------------------- models
section("face models")
for name in ("face_detection_yunet_2023mar.onnx", "face_recognition_sface_2021dec.onnx"):
    p = ROOT / "models" / name
    out("PASS" if p.exists() and p.stat().st_size > 50_000 else "FAIL", f"models/{name}" + ("" if p.exists() else " missing - run scripts/download_models.py"))

# ---------------------------------------------------------------- service and ollama
section("service")
unit = Path("/etc/systemd/system/sonku.service")
if unit.exists():
    txt = unit.read_text()
    out("FAIL" if "@" in txt else "PASS", "unit file still has @PLACEHOLDERS@ (copy the rendered file, not the template)" if "@" in txt else "unit file has no placeholders")
    out("PASS" if "-m app" in txt else "FAIL", "ExecStart runs 'python -m app'" if "-m app" in txt else "ExecStart does not run 'python -m app' (old robot_main.py?)")
    out("PASS" if "XDG_RUNTIME_DIR" in txt else "WARN", "XDG_RUNTIME_DIR set (needed for Bluetooth/PipeWire audio)" if "XDG_RUNTIME_DIR" in txt else "XDG_RUNTIME_DIR missing: the service cannot see Bluetooth audio")
    out("WARN" if "EnvironmentFile" in txt else "PASS", "unit uses EnvironmentFile: systemd keeps inline comments in values" if "EnvironmentFile" in txt else "unit does not use EnvironmentFile")
    if shutil.which("systemctl"):
        r = subprocess.run(["systemctl", "is-active", "sonku.service"], capture_output=True, text=True)
        out("PASS" if r.stdout.strip() == "active" else "WARN", f"sonku.service is {r.stdout.strip() or 'unknown'}")
else:
    out("WARN", "no /etc/systemd/system/sonku.service (fine when running in the foreground)")
section("ollama")
if s:
    try:
        import requests
        tags = requests.get(f"{s.ollama_base_url}/api/tags", timeout=3).json().get("models", [])
        names = {m.get("name") for m in tags}
        out("PASS", f"ollama reachable, {len(names)} model(s) installed")
        for label, tag in (("main model", s.ollama_model), ("decision model", s.decision_model)):
            out("PASS" if tag in names else "FAIL", f"{label} '{tag}' " + ("installed" if tag in names else f"NOT installed - run: ollama pull {tag}   (installed: {sorted(names)})"))
    except Exception as exc:
        out("WARN", f"ollama not reachable at {s.ollama_base_url}: {exc}")

print(f"\nsummary: {counts['PASS']} pass, {counts['WARN']} warn, {counts['FAIL']} fail")
sys.exit(1 if counts["FAIL"] else 0)
