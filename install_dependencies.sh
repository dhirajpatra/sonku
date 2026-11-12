#!/bin/bash
set -e

echo "[1/4] Installing System Packages..."
# sudo apt update && sudo apt install -y portaudio19-dev espeak-ng libcap-dev libatlas-base-dev
sudo apt update && sudo apt install -y portaudio19-dev espeak-ng libcap-dev libopenblas-dev pipewire-pulse libspa-0.2-bluetooth pipewire-audio-client-libraries pipewire-alsa bluetooth bluez blueman libasound2-plugins

# 1. Install the Python venv module (if not already installed)
sudo apt update
sudo apt install -y python3-venv python3-full

# 2. Create the virtual environment with access to system packages (recommended for OpenCV/ALSA bindings on Pi)
python3 -m venv --system-site-packages /home/dhirajpatra/robot_env

# 3. Activate the virtual environment
source ~/robot_env/bin/activate

echo "[2/4] Installing Python Requirements..."
pip install -r requirements.txt
python3 -m spacy download en_core_web_sm

echo "[3/4] Downloading Mithi-v1-flash TTS Model..."
if [ ! -d "/home/dhirajpatra/mithi-v1-flash" ]; then
    huggingface-cli download Oomni/mithi-v1-flash --local-dir /home/dhirajpatra/mithi-v1-flash
fi

echo "[3/4] Pulling Ollama Models..."
if ! command -v ollama &> /dev/null; then
    echo "Ollama CLI not found. Please install Ollama CLI first."
    exit 1
fi

ollama list                          # see which tags are installed
ollama pull qwen3.5:2b-q4_K_M        # or set OLLAMA_MODEL in .env to a tag from the list
ollama pull qwen3.5:0.8b             # decision model; set DECISION_MODEL to whatever "tev1:0.8b" really is

# 4. Set ALSA/PortAudio environment flags and run in foreground
export JACK_NO_START_SERVER=1
export PA_ALSA_PLUGHOST=1
export PA_ALSA_INITIAL_BUFFER_SIZE=4096

wpctl set-volume @DEFAULT_SINK@ 1.0
pactl set-sink-volume @DEFAULT_SINK@ 100%
pactl set-source-volume @DEFAULT_SOURCE@ 100%

# 1. Install your edited unit (use your edited file's path)
sudo cp systemd/sonku.service /etc/systemd/system/sonku.service

# 2. Retire the old service so it doesn't hold the mic and camera
sudo systemctl disable --now tabletop-robot.service

# 3. Reload systemd, enable on boot, start
sudo loginctl enable-linger dhirajpatra
bluetoothctl trust 30:63:71:04:38:BB # for BT mic speaker
pactl list sinks short
pactl list sources short
sudo systemctl daemon-reload && sudo systemctl restart sonku.service
sudo systemctl enable --now sonku.service

# 4. Check it
systemctl status sonku.service --no-pager
journalctl -u sonku.service -n 40 --no-pager

# 5. Test Mithi-v1-flash TTS model
~/robot_env/bin/pip install torch
cat /home/dhirajpatra/mithi-v1-flash/requirements.txt   # install anything else the model repo lists
~/robot_env/bin/python test_mithi.py                    # your existing script; confirms Mithi speaks on its own

# 1. Kill the old program by hand (systemctl can't stop a broken unit)
pgrep -af robot_main            # should show PID 1797
sudo kill 1797
sleep 2; pgrep -af robot_main || echo "old program is gone"

# 3. Check the file has no placeholders left
grep -n "@" /etc/systemd/system/sonku.service || echo "no placeholders left"
systemd-analyze verify /etc/systemd/system/sonku.service

# 4. Load and start it
sudo systemctl daemon-reload
sudo systemctl enable --now sonku.service
systemctl status sonku.service --no-pager
journalctl -u sonku.service -n 30 --no-pager

echo "[4/4] Setup Completed Successfully!"
