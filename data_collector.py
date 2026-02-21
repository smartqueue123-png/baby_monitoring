import cv2
import sounddevice as sd
import numpy as np
import pandas as pd
import time

## Look for the number next to your "Built-in Microphone" or "Realtek Audio."
print(sd.query_devices())

# Settings
DURATION = 120  # How many seconds to record
fs = 44100      # Sample rate for audio

data_log = []
cap = cv2.VideoCapture(0)
_, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

print("Recording normal behavior... keep it quiet and still!")

for i in range(DURATION):
    # 1. Capture Movement (Camera)
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(prev_gray, gray)
    move_score = np.mean(diff)  # Average change in pixels
    prev_gray = gray

    # 2. Capture Sound (Mic)
    recording = sd.rec(int(1 * fs), samplerate=fs, channels=1)
    sd.wait()
    sound_score = np.linalg.norm(recording) * 100  # Loudness

    # 3. Log Data [Hour, Temp(Simulated), Sound, Movement]
    data_log.append([time.localtime().tm_hour, 22.0, sound_score, move_score])
    
    if i % 10 == 0: print(f"{i}/{DURATION} seconds logged...")

# Save to CSV
df = pd.DataFrame(data_log, columns=['hour', 'temp', 'sound', 'movement'])
df.to_csv('training_data.csv', index=False)
cap.release()
print("Training data saved to training_data.csv")