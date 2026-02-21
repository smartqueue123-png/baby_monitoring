import cv2
import sounddevice as sd

print("--- Testing Camera ---")
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if ret:
    print("✅ Camera is working!")
else:
    print("❌ Camera NOT found at index 0.")
cap.release()

print("\n--- Testing Microphone ---")
try:
    print(sd.query_devices())
    print("✅ Audio devices found!")
except Exception as e:
    print(f"❌ Audio error: {e}")