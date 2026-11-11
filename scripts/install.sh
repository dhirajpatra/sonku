#!/usr/bin/env bash
# One-time setup on Raspberry Pi OS (Bookworm, 64-bit). Safe to re-run.
set -euo pipefail
APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${VENV:-$HOME/robot_env}"
cd "$APP_DIR"

echo "[1/5] System packages"
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip python3-picamera2 libportaudio2 alsa-utils sox libsox-fmt-mp3 libcamera-apps

echo "[2/5] Python environment at $VENV"
# --system-site-packages lets the venv see apt's picamera2 / libcamera bindings
[ -d "$VENV" ] || python3 -m venv --system-site-packages "$VENV"
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install -r requirements.txt

echo "[3/5] Face models"
"$VENV/bin/python" scripts/download_models.py || echo "!! download the face models manually (see README)"

echo "[4/5] Configuration"
if [ ! -f .env ]; then cp .env.example .env; chmod 600 .env; echo "Created .env - edit it now (ADMIN_PASSWORD, MODEL_API_KEY)."; fi

echo "[5/5] Local models (Ollama must already be installed)"
if command -v ollama >/dev/null; then
  ollama pull "$(grep -E '^OLLAMA_MODEL=' .env | cut -d= -f2-)" || true
  ollama pull "$(grep -E '^DECISION_MODEL=' .env | cut -d= -f2-)" || true
else
  echo "ollama not found: install it from https://ollama.com, then run: ollama pull qwen3.5:2b-q4_K_M"
fi
echo "Done. Next: $VENV/bin/python scripts/diagnose.py all   then   scripts/install_service.sh"
