#!/bin/bash
set -e

echo "[1/4] Installing System Packages..."
# sudo apt update && sudo apt install -y portaudio19-dev espeak-ng libcap-dev libatlas-base-dev
sudo apt update && sudo apt install -y portaudio19-dev espeak-ng libcap-dev libopenblas-dev

# 1. Install the Python venv module (if not already installed)
sudo apt update
sudo apt install -y python3-venv python3-full

# 2. Create the virtual environment with access to system packages (recommended for OpenCV/ALSA bindings on Pi)
python3 -m venv --system-site-packages /home/dhirajpatra/robot_env

# 3. Activate the virtual environment
source /home/dhirajpatra/robot_env/bin/activate

echo "[2/4] Installing Python Requirements..."
pip install -r requirements.txt
python3 -m spacy download en_core_web_sm

echo "[3/4] Downloading Mithi-v1-flash TTS Model..."
if [ ! -d "/home/dhirajpatra/mithi-v1-flash" ]; then
    huggingface-cli download Oomni/mithi-v1-flash --local-dir /home/dhirajpatra/mithi-v1-flash
fi

echo "[4/4] Setup Completed Successfully!"
