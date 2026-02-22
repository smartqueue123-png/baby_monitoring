import sounddevice as sd
import numpy as np
import time

def test_mic():
    fs = 44100
    duration = 2
    
    print("Recording for 2 seconds...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    
    audio = recording.flatten()
    rms = np.sqrt(np.mean(audio**2))
    print(f"RMS level: {rms}")
    print(f"Max amplitude: {np.max(np.abs(audio))}")
    
    return audio

if __name__ == "__main__":
    test_mic()