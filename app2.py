import streamlit as st
import cv2
import sounddevice as sd
import numpy as np
import pandas as pd
import joblib
import time
from cry_infer import audio_to_features

print(sd.query_devices())

st.set_page_config(page_title="AI Baby Monitor", page_icon="👶")
st.title("👶 Smart Baby Behavior Monitor")

# -----------------------------
# Load Models
# -----------------------------
try:
    model = joblib.load('baby_model.pkl')  # anomaly model
except:
    st.error("Model file 'baby_model.pkl' not found.")

try:
    cry_model = joblib.load('cry_model.pkl')  # cry classifier
except:
    st.warning("Cry model not found. Cry detection disabled.")
    cry_model = None

# -----------------------------
# UI Layout
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    temp_sim = st.slider("Simulate Temperature (°C)", 18, 35, 22)

with col2:
    st.write("Current Status:")
    status_box = st.empty()

st.subheader("Audio Activity")
sound_chart = st.line_chart(np.zeros((1, 1)))

st.subheader("Movement Activity")
move_chart = st.line_chart(np.zeros((1, 1)))

# -----------------------------
# Initialize Webcam
# -----------------------------
cap = cv2.VideoCapture(0)
ret, prev_frame = cap.read()

if ret:
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
else:
    st.error("Webcam not detected.")
    st.stop()

# -----------------------------
# Mic Function
# -----------------------------
def get_mic_audio_and_score():
    try:
        duration = 1.0
        fs = 44100
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, kind='input')
        sd.wait()
        audio = recording[:, 0]
        loudness = np.linalg.norm(audio) * 10
        return audio, fs, loudness
    except:
        return None, 44100, 0.0

# -----------------------------
# MAIN LOOP (UNCHANGED STRUCTURE)
# -----------------------------
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
    audio, fs, sound_score = get_mic_audio_and_score()

    # -----------------------------
    # NEW: Cry AI Detection
    # -----------------------------
    cry_prob = 0.0
    is_cry = False

    if cry_model is not None and audio is not None:
        features = audio_to_features(audio, fs)
        prob = cry_model.predict_proba(features)[0]
        cry_prob = float(prob[1])
        is_cry = cry_prob >= 0.80

    st.sidebar.write(f"🧠 Cry Probability: {cry_prob:.2f}")

    # -----------------------------
    # 3. Predict Anomaly (ORIGINAL)
    # -----------------------------
    current_data = pd.DataFrame(
        [[time.localtime().tm_hour, temp_sim, sound_score, move_score]],
        columns=['hour', 'temp', 'sound', 'movement']
    )

    prediction = model.predict(current_data)

    # -----------------------------
    # 4. Update UI (Enhanced Logic)
    # -----------------------------
    if prediction == -1 and is_cry:
        status_box.error("🚨 HIGH RISK: Cry + Anomaly!")
    elif is_cry:
        status_box.warning("👶 Cry Detected")
    elif prediction == -1:
        status_box.error("🚨 ANOMALY: Check Baby!")
    else:
        status_box.success("✅ Behavior: Normal")

    # Charts (UNCHANGED)
    sound_chart.add_rows(np.array([sound_score]))
    move_chart.add_rows(np.array([move_score]))

    time.sleep(0.05)