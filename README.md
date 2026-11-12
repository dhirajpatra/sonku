# Sonku — tabletop conversational robot (Raspberry Pi 5)

A voice robot that recognises faces, greets people it knows without a wake word, waits for
"sonku" / "hi sonku" for everyone else, and talks in 11 Indian languages. Includes a web
admin panel to see and delete what it remembers.

```
camera ─► presence watcher ─► face match (YuNet + SFace) ─┐
mic ─► VAD ─► speech-to-text ─► orchestrator ◄────────────┘
                                   │  decision rules/0.8B model (needs camera? name?)
                                   ▼
          Meta API ──(fail/timeout)──► Ollama qwen3.5:2b ─► sentence chunks ─► Mithi TTS ─► speaker
                                   │
                          SQLite memory (people, face templates, chats)  ◄── FastAPI admin panel :8000
```

## Layout
| Path | Purpose |
|---|---|
| `app/core/` | `orchestrator.py` (conversation logic), `intents.py` (wake word, yes/no, names), `maintenance.py` (memory limit, compression) |
| `app/vision/` | camera, face engine, presence watcher |
| `app/audio/` | microphone + VAD, speech-to-text backends, Mithi TTS |
| `app/llm/` | Meta client, Ollama client, router with fallback, decision helper |
| `app/memory/store.py` | SQLite storage |
| `app/web/` | admin API (`server.py`) and page (`static/index.html`) |
| `scripts/` | `install.sh`, `install_service.sh`, `download_models.py`, `diagnose.py` |
| `systemd/sonku.service` | service template |
| `tests/` | `pytest` suite (no hardware needed) |

## Install (on the Pi)
```bash
cd ~/sonku                       # copy this folder to the Pi first
./scripts/install.sh             # apt packages, venv (~/robot_env), pip, face models, .env, ollama pull
nano .env                        # set ADMIN_PASSWORD and MODEL_API_KEY at minimum
~/robot_env/bin/python scripts/diagnose.py all    # check camera, faces, speaker, LLMs one by one
./scripts/install_service.sh     # installs sonku.service, retires the old tabletop-robot.service
```
`diagnose.py devices` lists audio devices; put part of the mic name in `AUDIO_INPUT_MATCH` and the
speaker (e.g. `plughw:2,0` for the Jabra) in `AUDIO_OUTPUT_DEVICE`.

## Run / stop
```bash
./start.sh          # restart service and follow the log (Ctrl+C leaves it running)
./stop.sh           # stop
./logs.sh           # follow the log
sudo systemctl enable sonku    # start on boot (install_service.sh already does this)
~/robot_env/bin/python -m app  # run in the foreground for debugging (stop the service first)
```
Admin panel: `http://<pi-ip>:8000` (user `ADMIN_USER`, password `ADMIN_PASSWORD`). It is HTTP on your LAN, so do not expose the port to the internet.

## How it behaves
1. **Known face** in view for ~2 s: greets by name in that person's last language and keeps listening, no wake word. Not greeted again within `GREET_COOLDOWN_MIN`.
2. **Unknown face**: silent until the wake word. Then it asks the name, asks permission ("may I remember your face?"), and stores 3 face templates. Say "forget me" to delete your face and conversations.
3. **Conversation** ends after `SESSION_TIMEOUT_SEC` of silence, on "goodbye", or when the person walks away.
4. **Camera frame** goes to the LLM only when the question is about what Sonku can see.
5. **Faces are overwritten monthly**: when a person is recognised and their newest template is older than `FACE_REFRESH_DAYS` (30), a fresh one is stored and the oldest is dropped (max `FACES_PER_PERSON`). Templates not refreshed for `FACE_MAX_AGE_DAYS` (90) are deleted.
6. **Memory limit** (`MAX_MEMORY_GB`, 5): the whole `data/` folder is measured. At 80% the dashboard and log warn; at 100% the oldest conversations are deleted down to 90%. Old conversations are also compressed into a short per-person summary.

Only 128-number face templates are stored — no photos, no audio.

## Speech models
- **STT** (`STT_BACKEND`, comma-separated, tried in order): `google` (online, default), `whisper_cpp` (offline, build whisper.cpp and set `WHISPER_CPP_MODEL`), `indic` (`bodhan-ai/indic-transcribe-core`; gated model, `pip install -r requirements-indic.txt`, `hf auth login`). It is a 1.2B-parameter model: run `diagnose.py stt-bench google indic` and check the real-time factor before switching.
- **TTS**: Mithi v1 Flash locally; English gTTS (online) only if Mithi fails to load.
- The reply language follows the language you spoke. Language is detected from script, so Latin-script speech counts as English unless the Indic STT reports otherwise.

## Tuning for speed
`diagnose.py llm` shows the time to the first sentence per backend, and every turn logs `stt_ms`, `llm_first_sentence_ms`, `first_audio_ms`, `total_ms` (also on the dashboard). Lower `FALLBACK_TIMEOUT` so a slow Meta call falls back sooner; after two failures Meta is skipped for 60 s.

## Audio devices (Bluetooth, USB headset, built-in)
Sonku asks the sound server (PipeWire) which microphones and speakers exist, ranks them by `AUDIO_PRIORITY`
(default `bluetooth,usb,other`), makes the best ones the system defaults, and re-checks every 10 s, so a Bluetooth
headset that connects later is picked up and a disconnected one is dropped. The dashboard's **Audio devices** card
shows the choice and has a *Rescan* button. `diagnose.py audio` prints the same from the shell.
- Set `AUDIO_PRIORITY=usb,bluetooth,other` to prefer the Jabra when both are connected.
- Bluetooth microphones need the headset (HFP) profile; in music (A2DP) mode only the speaker exists and the
  microphone is taken from the next device in the list.
- Pair once and trust the device so it reconnects at boot: `bluetoothctl trust <MAC>`.
- The service must reach your PipeWire session: `sudo loginctl enable-linger $USER` and keep
  `Environment=XDG_RUNTIME_DIR=/run/user/<uid>` (`id -u`) in `sonku.service` (the template does this).
- No sound server? Set `AUDIO_BACKEND=alsa` and use `AUDIO_INPUT_MATCH` / `AUDIO_OUTPUT_DEVICE`.

## Troubleshooting
| Symptom | Check |
|---|---|
| "camera unavailable" | `diagnose.py camera`; the old service may still hold the camera: `sudo systemctl status tabletop-robot` |
| face models missing | `python scripts/download_models.py`, or download the two `.onnx` files into `models/` |
| never hears you | `diagnose.py devices`, then `mic`; lower `VAD_MIN_RMS` |
| no sound | `AUDIO_OUTPUT_DEVICE`, `aplay -l`; test with `diagnose.py speaker` |
| wrong person recognised | raise `FACE_MATCH_THRESHOLD` (stricter) in Settings |
| forgot the admin password | read the generated one in `./logs.sh`, or set `ADMIN_PASSWORD` |

## Development (no hardware)
```bash
pip install -r requirements-dev.txt
SONKU_HEADLESS=true ADMIN_PASSWORD=test python -m app    # panel + LLM only
python -m pytest -q
```
