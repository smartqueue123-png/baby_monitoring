import streamlit as st
import cv2
import sounddevice as sd
import numpy as np
import pandas as pd
import joblib
import time
import mysql.connector

from cry_infer import audio_to_features

# ---------------------------------
# PAGE CONFIG
# ---------------------------------
st.set_page_config(page_title="AI Baby Monitor", page_icon="👶")
st.title("👶 Smart Baby Behavior Monitor")

# ---------------------------------
# MYSQL CONNECTION
# ---------------------------------
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="your_password_here",  # 🔴 CHANGE THIS
        database="baby_monitor"
    )

def log_to_mysql(hour, temp, sound, movement, cry_prob, anomaly_score, is_cry, is_anomaly):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        query = """
        INSERT INTO monitor_activity
        (hour, temperature, sound_level, movement_level,
         cry_probability, anomaly_score, is_cry, is_anomaly)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
        """

        values = (
            hour,
            temp,
            sound,
            movement,
            cry_prob,
            anomaly_score,
            int(is_cry),
            int(is_anomaly)
        )

        cursor.execute(query, values)
        conn.commit()

        cursor.close()
        conn.close()

    except Exception as e:
        st.error(f"MySQL Error: {e}")

# ---------------------------------
# LOAD MODELS
# ---------------------------------
try:
    anomaly_model = joblib.load("baby_model.pkl")
    cry_model = joblib.load("cry_model.pkl")
except:
    st.error("❌ Model files not found.")
    st.stop()

# ---------------------------------
# UI CONTROLS
# ---------------------------------
col1, col2 = st.columns(2)

with col1:
    temp_sim = st.slider("Simulated Temperature (°C)", 18, 35, 22)

with col2:
    status_box = st.empty()

if "run" not in st.session_state:
    st.session_state.run = False

if st.button("▶ Start Monitoring"):
    st.session_state.run = True

if st.button("⏹ Stop"):
    st.session_state.run = False

sound_chart = st.line_chart(np.zeros((1, 1), dtype=np.float64))
move_chart = st.line_chart(np.zeros((1, 1), dtype=np.float64))

# ---------------------------------
# SAFE CAMERA INITIALIZATION
# ---------------------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    st.error("❌ Webcam not detected.")
    st.stop()

time.sleep(1)

ret, prev_frame = cap.read()

if not ret or prev_frame is None:
    st.error("❌ Failed to read from webcam.")
    st.stop()

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# ---------------------------------
# LIVE MICROPHONE AUDIO FUNCTION
# ---------------------------------
def get_mic_audio_and_score():
    try:
        fs = 44100
        chunk = int(0.05 * fs)  # 50 ms for near real-time

        # Record tiny chunk
        recording = sd.rec(chunk, samplerate=fs, channels=1, dtype='float32')
        sd.wait()

        audio = recording[:, 0]

        # RMS → scale 0–100, reflects silence immediately
        rms = np.sqrt(np.mean(audio**2))
        loudness = float(rms * 100)
        if loudness > 100:
            loudness = 100.0

        return audio, fs, loudness

    except Exception as e:
        print("Microphone error:", e)
        return None, 44100, 0.0

# ---------------------------------
# MAIN LOOP
# ---------------------------------
while st.session_state.run:

    # -------- CAMERA --------
    ret, frame = cap.read()
    if not ret or frame is None:
        st.warning("⚠ Webcam frame not received.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_diff = cv2.absdiff(prev_gray, gray)
    move_score = float(np.mean(frame_diff))
    prev_gray = gray

    # -------- MICROPHONE --------
    audio, audio_fs, sound_score = get_mic_audio_and_score()

    # -------- CRY DETECTION --------
    cry_prob = 0.0
    is_cry = False

    if audio is not None:
        feats = audio_to_features(audio, audio_fs)
        prob = cry_model.predict_proba(feats)[0]
        cry_prob = float(prob[1])
        is_cry = cry_prob >= 0.80

    # -------- ANOMALY DETECTION --------
    hour_now = time.localtime().tm_hour

    current_data = pd.DataFrame(
        [[hour_now, temp_sim, sound_score, move_score]],
        columns=["hour", "temp", "sound", "movement"]
    )

    prediction = anomaly_model.predict(current_data)[0]
    anomaly_score = anomaly_model.decision_function(current_data)[0]
    is_anomaly = prediction == -1

    # -------- STORE TO MYSQL --------
    log_to_mysql(
        hour_now,
        temp_sim,
        sound_score,
        move_score,
        cry_prob,
        anomaly_score,
        is_cry,
        is_anomaly
    )

    # -------- DECISION DISPLAY --------
    if is_anomaly and is_cry:
        status_box.error("🚨 HIGH RISK: Cry + Abnormal")
    elif is_cry:
        status_box.warning("👶 Cry Detected")
    elif is_anomaly:
        status_box.error("🚨 Abnormal Behavior")
    else:
        status_box.success("✅ Normal")

    # -------- UPDATE CHARTS --------
    sound_chart.add_rows(np.array([float(sound_score)], dtype=np.float64))
    move_chart.add_rows(np.array([float(move_score)], dtype=np.float64))

    # very short sleep to make it almost real-time
    time.sleep(0.05)

cap.release()