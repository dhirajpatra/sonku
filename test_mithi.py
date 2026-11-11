import sys
import subprocess

sys.path.insert(0, '/home/dhirajpatra/mithi-v1-flash')

try:
    from mithi_tts import Mithi
    print("[Mithi TTS] Loading model onto CPU...")
    tts = Mithi("/home/dhirajpatra/mithi-v1-flash", device="cpu")
    
    # Synthesize Bengali text
    audio = tts.synthesize("নমস্কার! আমি সোনকু, আপনার টেবিলটপ রোবট।", lang="bn", voice="charvi")
    out_file = "/tmp/sonku_bn_test.wav"
    tts.save(audio, out_file)
    print(f"[Mithi TTS] Saved audio to {out_file}")

    # Output directly to Jabra EVOLVE 20 SE MS (Card 2, Device 0)
    subprocess.run(["aplay", "-D", "plughw:2,0", out_file])

except Exception as e:
    print(f"[Mithi TTS Error]: {e}")