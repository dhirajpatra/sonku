import subprocess
import cv2
import os

def test_camera():
    frame_path = '/tmp/test_capture.jpg'
    print("[Testing Camera] Capturing frame via rpicam-still...")
    cmd = f"rpicam-still -o {frame_path} --timeout 1000 --width 640 --height 480 --nopreview"
    res = subprocess.run(cmd, shell=True)
    
    if res.returncode == 0 and os.path.exists(frame_path):
        img = cv2.imread(frame_path)
        if img is not None:
            print(f"Success! Frame captured saved to {frame_path} (Shape: {img.shape})")
        else:
            print("Error reading captured image frame.")
    else:
        print("Camera capture failed. Check hardware connection or camera lock.")

if __name__ == "__main__":
    test_camera()
