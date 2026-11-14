#!/usr/bin/env python3
"""Hardware and service checks. Usage: python scripts/diagnose.py [audio|devices|camera|faces|mic|speaker|llm|meta|meta-tokens|stt-bench|all]"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings  # noqa: E402

s = Settings.from_env()


def audio():
    from app.audio.devices import AudioRouter
    r = AudioRouter(s)
    r.scan()
    st = r.status()
    print(f"mode={st['mode']} (AUDIO_BACKEND={st['backend']}, priority={st['priority']})")
    print("microphones:", st["microphones"] or "none")
    print("speakers:   ", st["speakers"] or "none")
    print(f"chosen -> microphone={st['microphone']}  speaker={st['speaker']}")
    if st["mode"] == "alsa":
        print("No sound server found: using AUDIO_INPUT_MATCH / AUDIO_OUTPUT_DEVICE. For Bluetooth install pipewire-pulse (see README).")


def devices():
    import sounddevice as sd
    print(sd.query_devices())
    print("\nSet AUDIO_INPUT_MATCH to part of the microphone name; set AUDIO_OUTPUT_DEVICE from `aplay -l` (e.g. plughw:2,0).")


def camera():
    from app.vision.camera import Camera
    cam = Camera(s)
    cam.start()
    t0 = time.time()
    frame = cam.read()
    print(f"backend={cam.backend} first frame in {time.time() - t0:.2f}s shape={None if frame is None else frame.shape}")
    if frame is not None:
        Path("/tmp/sonku_diag.jpg").write_bytes(cam.jpeg(frame))
        print("saved /tmp/sonku_diag.jpg (check it is the right way up)")
    cam.stop()


def faces():
    from app.vision.camera import Camera
    from app.vision.faces import FaceEngine
    cam, eng = Camera(s), FaceEngine(s)
    cam.start()
    eng.load()
    for i in range(5):
        frame = cam.read()
        t0 = time.time()
        rows = eng.detect(frame)
        ms = (time.time() - t0) * 1000
        emb_ms = 0.0
        if rows:
            t1 = time.time()
            eng.embed(frame, rows[0])
            emb_ms = (time.time() - t1) * 1000
        print(f"frame {i}: {len(rows)} face(s)  detect {ms:.0f} ms  embed {emb_ms:.0f} ms")
        time.sleep(0.5)
    cam.stop()


def mic():
    from app.audio.recorder import Recorder
    from app.audio.stt import STT
    r = Recorder(s)
    r.open()
    stt = STT(s)
    for phrase in ("yes", "yes, remember me", "hello Sonku"):
        print(f'\nSay "{phrase}" now...')
        audio = r.listen(wait_sec=8)
        if audio is None:
            print(f"  nothing recorded: loudest sound {r.last_peak:.0f}, needs more than {r.last_thr:.0f}. Speak louder/closer or lower VAD_MIN_RMS.")
            continue
        t0 = time.time()
        text, lang = stt.transcribe(audio)
        print(f"  recorded {len(audio) / 16000:.1f}s (loudest {r.last_peak:.0f}, threshold {r.last_thr:.0f}) -> [{lang}] {text!r} in {time.time() - t0:.1f}s")


def speaker():
    from app.audio.tts import TTS
    tts = TTS(s)
    tts.load()
    t0 = time.time()
    tts.speak_stream(["Hello, I am Sonku. Can you hear me clearly?"], "en")
    print(f"done in {time.time() - t0:.1f}s, first audio after {tts.first_audio_ts - t0:.1f}s")
    tts.speak_stream(["नमस्ते! मैं सोनकू हूँ।"], "hi")


def llm():
    from app.llm.router import LLMRouter
    router = LLMRouter(s)
    for backend in ("meta_api", "ollama"):
        s.llm_backend = backend
        router._skip_until = 0
        t0 = time.time()
        try:
            out = []
            for i, sent in enumerate(router.reply_stream("Answer in one short sentence.", [], "Suggest a Raspberry Pi project.")):
                if i == 0:
                    print(f"[{backend}] first sentence after {time.time() - t0:.1f}s")
                out.append(sent)
            print(f"[{backend}] answered by {router.last_backend}: {' '.join(out)}")
        except Exception as exc:
            print(f"[{backend}] FAILED: {exc}")


def meta():
    import requests
    base = s.meta_api_url.rsplit("/", 1)[0]  # https://api.meta.ai/v1
    r = requests.get(f"{base}/models", headers={"Authorization": f"Bearer {s.meta_api_key}"}, timeout=10)
    print(f"GET {base}/models -> HTTP {r.status_code}")
    try:
        ids = [m.get("id") for m in r.json().get("data", [])]
        print("models:", ids or r.text[:300])
        print(f"configured META_MODEL={s.meta_model!r} is {'listed' if s.meta_model in ids else 'NOT in that list'}")
    except Exception:
        print(r.text[:300])


def meta_tokens():
    """Same short question at each reasoning effort: shows what a turn really costs and how fast it is."""
    from app.llm.meta_client import MetaClient
    c = MetaClient(s)
    for effort in ("minimal", "low", "medium"):
        s.meta_reasoning_effort = effort
        before = dict(c.totals)
        t0 = time.time()
        try:
            text = c.complete("Reply in one short spoken sentence.", [], "Suggest a Raspberry Pi project.")
        except Exception as exc:
            text = f"FAILED: {exc}"
        d = {k: c.totals[k] - before[k] for k in ("input", "cached", "output", "reasoning")}
        print(f"effort={effort:8} {time.time() - t0:.1f}s tokens={d}  -> {text[:90]!r}")


def stt_bench():
    import numpy as np
    from app.audio.recorder import Recorder
    from app.audio.stt import STT
    r = Recorder(s)
    r.open()
    print("Say a sentence to record a test clip...")
    audio = r.listen(wait_sec=10)
    if audio is None:
        return print("no speech captured")
    secs = len(audio) / 16000
    for backend in [b.strip() for b in sys.argv[2:]] or ["google", "indic"]:
        s.stt_backend = backend
        stt = STT(s)
        t0 = time.time()
        text, lang = stt.transcribe(audio)
        dt = time.time() - t0
        print(f"{backend}: {dt:.1f}s for {secs:.1f}s of audio (real-time factor {dt / secs:.2f}) -> [{lang}] {text}")


CHECKS = {"audio": audio, "devices": devices, "camera": camera, "faces": faces, "mic": mic, "speaker": speaker, "llm": llm, "meta": meta, "meta-tokens": meta_tokens, "stt-bench": stt_bench}
if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in CHECKS.items():
        if which in ("all", name) and not (which == "all" and name in ("stt-bench", "mic", "meta-tokens")):
            print(f"\n=== {name} ===")
            try:
                fn()
            except Exception as exc:
                print(f"FAILED: {type(exc).__name__}: {exc}")
