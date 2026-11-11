# Sonku Tabletop Conversational AI

Multimodal stationary conversational AI application built for Raspberry Pi 5 (16GB).

## Features
- **Visual Reasoning**: Direct capture via `rpicam-still` hardware tools (bypassing V4L2 lock issues).
- **Multilingual Support**: Automatic input language detection (`ftlangdetect`) and offline 11-language Indian speech synthesis via `Oomni/mithi-v1-flash`.
- **Hybrid LLM Routing**: Primary execution via Meta API (`muse-spark-1.3-contributor`) with zero-latency automated fallback to local Ollama (`qwen3.5:4b`).
- **Systemd Integration**: Configured as an auto-restarting background daemon.

## Quick Setup

1. Extract application files to `/home/dhirajpatra/`.
2. Make shell script executable and run installation:
   ```bash
   chmod +x install_dependencies.sh
   ./install_dependencies.sh
   ```
3. Copy `.env.example` to `.env` and fill in your Meta API key if active:
   ```bash
   cp .env.example .env
   ```
4. Test hardware components:
   ```bash
   python3 test_cam.py
   python3 test_mithi.py
   ```
5. Install systemd service:
   ```bash
   sudo cp tabletop-robot.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable --now tabletop-robot.service
   sudo cp tabletop-robot.service /etc/systemd/system/
   sudo journalctl -u tabletop-robot.service -f
   ```
