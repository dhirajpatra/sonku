#!/usr/bin/env bash
# Installs everything Mithi TTS needs into the robot venv, then speaks a test sentence to a wav file.
# Mithi's own requirements.txt (torch, safetensors, misaki, ...) is the source of truth; this just runs it.
set -euo pipefail
MITHI="${MITHI_MODEL_PATH:-$HOME/mithi-v1-flash}"
VENV="${VENV:-$HOME/robot_env}"
[ -f "$MITHI/requirements.txt" ] || { echo "No $MITHI/requirements.txt - download the model first: hf download Oomni/mithi-v1-flash --local-dir $MITHI"; exit 1; }
"$VENV/bin/pip" install -r "$MITHI/requirements.txt"
"$VENV/bin/python" -m spacy download en_core_web_sm
cd "$MITHI"
"$VENV/bin/python" - << 'PY'
import time
from mithi_tts import Mithi
t0 = time.time()
tts = Mithi(".", device="cpu")
print(f"model loaded in {time.time() - t0:.1f}s")
for lang, text in (("en", "Hello, I am Sonku."), ("hi", "नमस्ते! मैं सोनकू हूँ।")):
    t0 = time.time()
    audio = tts.synthesize(text, lang=lang, voice="charvi")
    tts.save(audio, f"/tmp/mithi_{lang}.wav")
    print(f"{lang}: synthesised in {time.time() - t0:.1f}s -> /tmp/mithi_{lang}.wav")
PY
echo "OK. Play it with: paplay /tmp/mithi_en.wav"
