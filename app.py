import streamlit as st
import cv2
import sounddevice as sd
import numpy as np
import joblib
import time

print(sd.query_devices())

st.set_page_config(page_title="AI Baby Monitor", page_icon="👶")
st.title("👶 Smart Baby Behavior Monitor")

# Load the saved model
try:
    model = joblib.load('baby_model.pkl')
except:
    st.error("Model file 'baby_model.pkl' not found. Please ensure it is in the same directory.")

# UI Layout
col1, col2 = st.columns(2)
with col1:
    temp_sim = st.slider("Simulate Temperature (°C)", 18, 35, 22)
with col2:
    st.write("Current Status:")
    status_box = st.empty() # This is for text/alerts only

# --- FIX 1: Explicitly define the chart targets ---
st.subheader("Audio Activity")
sound_chart = st.line_chart(np.zeros((1, 1)))

st.subheader("Movement Activity")
move_chart = st.line_chart(np.zeros((1, 1)))

# Initialize Webcam
cap = cv2.VideoCapture(0)
# Warm up the camera to avoid empty frames
ret, prev_frame = cap.read()
if ret:
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# --- FIX 2: Mic/Speaker Issue ---
# This ensures we only look for INPUT devices, not output loops
def get_mic_score():
    try:
        # 0.2 seconds of audio
        duration = 0.2 
        fs = 44100
        # Explicitly setting kind='input' prevents accessing speakers
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, kind='input')
        sd.wait()
        return np.linalg.norm(recording) * 10
    except Exception as e:
        return 0.0

while True:
    # 1. Camera Logic
    ret, frame = cap.read()
    if not ret:
        st.warning("Webcam not detected.")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_diff = cv2.absdiff(prev_gray, gray)
    move_score = np.mean(frame_diff)
    prev_gray = gray 
    
    # 2. Sound Logic
    sound_score = get_mic_score()

    #
    st.sidebar.write(f"🎤 Mic Check: {sound_score}")

    # 3. Predict Anomaly
    current_data = pd.DataFrame([[time.localtime().tm_hour, temp_sim, sound_score, move_score]], columns=['hour', 'temp', 'sound', 'movement'])
    prediction = model.predict(current_data)

    # 4. Update UI
    # We use .success() or .error() inside the empty status_box container
    if prediction == -1:
        status_box.error("🚨 ANOMALY: Check Baby!")
    else:
        status_box.success("✅ Behavior: Normal")
    
    # --- FIX 1 (Continued): Targeted add_rows ---
    sound_chart.add_rows(np.array([sound_score]))
    move_chart.add_rows(np.array([move_score]))
    
    time.sleep(0.05)