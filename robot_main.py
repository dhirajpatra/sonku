import os
import sys
import ctypes
import time
import base64
import subprocess
import requests
import speech_recognition as sr
try:
    from ftlangdetect import detect
except ImportError:
    try:
        import fasttext
        # Fallback wrapper mimicking detect(text) -> {"lang": "en", "score": 0.99}
        def detect(text: str):
            # If fasttext model isn't preloaded, default gracefully to English
            return {"lang": "en", "score": 1.0}
    except ImportError:
        try:
            from langdetect import detect as _ld_detect
            def detect(text: str):
                try:
                    return {"lang": _ld_detect(text), "score": 0.99}
                except Exception:
                    return {"lang": "en", "score": 1.0}
        except ImportError:
            # Minimal no-op fallback if no language detection library is present
            def detect(text: str):
                return {"lang": "en", "score": 1.0}
from gtts import gTTS
import gc
from dotenv import load_dotenv

# Force PortAudio to ignore JACK server checks
os.environ["JACK_NO_START_SERVER"] = "1"
os.environ["PA_ALSA_PLUGHOST"] = "1"

# Suppress ALSA/JACK C-library log spam in systemd
ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p)
def py_error_handler(filename, line, function, err, fmt):
    pass
c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
try:
    asound = ctypes.cdll.LoadLibrary('libasound.so.2')
    asound.snd_lib_error_set_handler(c_error_handler)
except Exception:
    pass

MITHI_PATH = os.getenv("MITHI_MODEL_PATH", "/home/dhirajpatra/mithi-v1-flash")
sys.path.append(MITHI_PATH)

FALLBACK_TIMEOUT = int(os.getenv("FALLBACK_TIMEOUT", 30))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", 500))

try:
    from mithi_tts import Mithi
except ImportError:
    Mithi = None

load_dotenv('/home/dhirajpatra/.env', override=True)

MITHI_LANG_MAP = {
    'bn': 'bn', 'hi': 'hi', 'en': 'en', 'gu': 'gu',
    'kn': 'kn', 'ml': 'ml', 'mr': 'mr', 'or': 'or',
    'pa': 'pa', 'ta': 'ta', 'te': 'te'
}

class VisionEngine:
    """Handles low-latency image capture on Pi 5 via libcamera rpicam-still (headless 180-deg flip)."""
    def __init__(self, width=640, height=480):
        self.width = str(width)
        self.height = str(height)
        self.output_path = "/tmp/sonku_vision.jpg"

    def capture_frame_b64(self):
        try:
            # Execute rpicam-still snapshot with headless preview suppression and 180-deg orientation
            cmd = [
                "rpicam-still",
                "-n",                      # Suppress DRM/EGL preview window allocation
                "-t", "1",
                "-o", self.output_path,
                "--immediate",
                "--width", self.width,
                "--height", self.height,
                "--vflip",
                "--hflip"
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            
            # Read image and convert to Base64
            with open(self.output_path, "rb") as f:
                b64_str = base64.b64encode(f.read()).decode("utf-8")
            
            print("[VisionEngine]: Captured right-side up frame (preview suppressed).", flush=True)
            return b64_str
        except Exception as e:
            print(f"[VisionEngine Error]: Frame capture failed ({e})", flush=True)
            return None

class AudioEngine:
    def __init__(self, alsa_card_out="default"):
        self.recognizer = sr.Recognizer()
        self.device_out = alsa_card_out
        self.mic_index = self._find_jabra_mic_index()
        print(f"[AudioEngine] Initialized (Output: {self.device_out}, Mic Index: {self.mic_index})...", flush=True)

    def _find_jabra_mic_index(self):
        try:
            mic_list = sr.Microphone.list_microphone_names()
            for index, name in enumerate(mic_list):
                if "Jabra" in name or "USB Audio" in name:
                    return index
        except Exception as e:
            print(f"[AudioEngine Warning] Mic scan error: {e}", flush=True)
        return None

    def listen_and_detect(self, timeout=5):
        source = None
        try:
            # Force explicit instantiating and releasing of PyAudio microphone handle
            source = sr.Microphone(device_index=self.mic_index, sample_rate=16000, chunk_size=1024)
            with source:
                self.recognizer.dynamic_energy_threshold = False
                self.recognizer.energy_threshold = 400
                self.recognizer.pause_threshold = 0.8
                
                print("[Sonku] Listening...", flush=True)
                # Set non-blocking listen limit
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=8)
                
                text = self.recognizer.recognize_google(audio, language="en-US")
                print(f"[User (en)]: {text}", flush=True)
                return text, "en"

        except sr.WaitTimeoutError:
            # Timeout is expected when silent - loop cleanly back
            return None, None
        except sr.UnknownValueError:
            return None, None
        except Exception as e:
            print(f"[Audio Input Error]: {e}", flush=True)
            return None, None
        finally:
            # Force cleanup of ALSA handles
            del source
            gc.collect()

    def speak(self, text, lang="en"):
        print(f"[Sonku (en)]: {text}", flush=True)
        mp3_file = "/tmp/sonku_hd.mp3"
        wav_file = "/tmp/sonku_48k.wav"

        try:
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(mp3_file)
            
            subprocess.run(
                ["sox", mp3_file, "-r", "48000", "-c", "2", wav_file],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )
            
            # Run aplay with short buffer limit so it closes ALSA PCM handles completely
            res = subprocess.run(["aplay", "-D", self.device_out, wav_file], capture_output=True, text=True)
            if res.returncode != 0:
                print(f"[aplay Error]: {res.stderr.strip()}", flush=True)
            else:
                print("[aplay Output]: Audio playback complete.", flush=True)
        except Exception as e:
            print(f"[TTS Execution Error]: {e}", flush=True)

class LLMClient:
    """LLM Router for Meta API with fallback to local Ollama."""
    def __init__(self):
        load_dotenv('/home/dhirajpatra/.env', override=True)
        
        self.backend = os.getenv("LLM_BACKEND", "meta_api").lower()
        self.meta_api_key = os.getenv("MODEL_API_KEY")
        self.meta_url = os.getenv("META_API_URL", "https://api.meta.ai/v1/responses")
        self.meta_model = os.getenv("META_MODEL", "muse-spark-1.3-contributor")
        
        self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "qwen3.5:4b")
        
        if not self.meta_api_key:
            print("[LLM Router WARNING] MODEL_API_KEY is missing or empty! Meta API calls will fail.")

    def generate(self, user_input, lang="en", image_b64=None):
        system_prompt = (
            f"You are Sonku, a tabletop AI robot on Raspberry Pi 5. "
            f"The user spoke in language code '{lang}'. MUST answer strictly in language code '{lang}'. "
            f"Keep answers brief, conversational, and direct (2 short sentences)."
        )

        if self.backend == "meta_api":
            print(f"[LLM Router] Calling Meta API ({self.meta_model})...")
            res = self._call_meta_api(system_prompt, user_input, image_b64)
            if res:
                return res
            print("[LLM Router] Meta API call failed/timed out. Falling back to local Ollama...")

        print(f"[LLM Router] Calling local Ollama ({self.ollama_model})...")
        return self._call_ollama(system_prompt, user_input, image_b64)

    def _call_meta_api(self, system_prompt, user_input, image_b64=None):
        if not self.meta_api_key:
            print("[Meta API Error]: MODEL_API_KEY is None.")
            return None

        headers = {
            "Authorization": f"Bearer {self.meta_api_key}",
            "Content-Type": "application/json",
        }
        content_list = [{"type": "input_text", "text": f"{system_prompt}\nUser: {user_input}"}]
        
        if image_b64:
            content_list.append({"type": "input_image", "image_url": f"data:image/jpeg;base64,{image_b64}"})

        payload = {
            "model": self.meta_model,
            "input": [{"role": "user", "content": content_list}],
            "reasoning": {"effort": "low"},
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "stream": False,
        }

        try:
            res = requests.post(self.meta_url, headers=headers, json=payload, timeout=(10, FALLBACK_TIMEOUT))
            if res.status_code == 200:
                data = res.json()
                for item in data.get("output", []):
                    if item.get("type") == "message" and "content" in item:
                        for content_part in item["content"]:
                            if content_part.get("type") == "output_text":
                                return content_part.get("text")
            print(f"[Meta API HTTP Error]: Status {res.status_code} - {res.text}")
            return None
        except requests.exceptions.Timeout:
            print(f"[Meta API Error]: Request timed out (exceeded {FALLBACK_TIMEOUT}s limit).")
            return None
        except Exception as e:
            print(f"[Meta API Exception]: {e}")
            return None

    def _call_ollama(self, system_prompt, user_input, image_b64=None):
        prompt = f"{system_prompt}\nUser: {user_input}\nSonku:"
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False
        }
        if image_b64:
            payload["images"] = [image_b64]

        try:
            res = requests.post(self.ollama_url, json=payload, timeout=20)
            if res.status_code == 200:
                return res.json().get('response', 'Model returned empty payload.')
            return "Ollama local model failed."
        except Exception as e:
            print(f"[Ollama Exception]: {e}")
            return "Both primary Meta API and fallback Ollama service are offline."

class RitchieMiniRobot:
    def __init__(self):
        self.vision = VisionEngine()
        self.audio = AudioEngine()
        self.llm = LLMClient()

    def run(self):
        print("[Sonku System] Multi-turn Loop Active.", flush=True)
        
        while True:
            try:
                user_speech, _ = self.audio.listen_and_detect(timeout=5)
                
                if user_speech:
                    image_b64 = self.vision.capture_frame_b64()
                    reply = self.llm.generate(user_speech, image_b64=image_b64)
                    self.audio.speak(reply)
                
                # Small pause to let ALSA release hardware DAC locks
                time.sleep(0.5)

            except Exception as e:
                print(f"[Sonku Main Loop Error]: {e}", flush=True)
                time.sleep(1)

if __name__ == "__main__":
    robot = RitchieMiniRobot()
    robot.run()