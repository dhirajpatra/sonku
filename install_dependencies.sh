#!/bin/bash
set -e

echo "[1/4] Installing System Packages..."
# sudo apt update && sudo apt install -y portaudio19-dev espeak-ng libcap-dev libatlas-base-dev
sudo apt update && sudo apt install -y portaudio19-dev espeak-ng libcap-dev libopenblas-dev

echo "[2/4] Installing Python Requirements..."
pip install -r requirements.txt
python3 -m spacy download en_core_web_sm

echo "[3/4] Downloading Mithi-v1-flash TTS Model..."
if [ ! -d "/home/dhirajpatra/mithi-v1-flash" ]; then
    huggingface-cli download Oomni/mithi-v1-flash --local-dir /home/dhirajpatra/mithi-v1-flash
fi

echo "[4/4] Setup Completed Successfully!"
