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
    st.session_state.sound_buffer = []
    st.session_state.move_buffer = []
    st.session_state.counter = 0
    st.session_state.cap = None
    st.session_state.silence_level = None
    st.session_state.prev_sound = 0

if st.button("▶ Start Monitoring"):
    st.session_state.run = True
    st.session_state.sound_buffer = []
    st.session_state.move_buffer = []
    st.session_state.silence_level = None
    st.session_state.prev_sound = 0
    if st.session_state.cap is None:
        st.session_state.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if st.button("⏹ Stop"):
    st.session_state.run = False
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None

# Create placeholders for charts
sound_chart = st.line_chart(np.zeros((1, 1), dtype=np.float64))
move_chart = st.line_chart(np.zeros((1, 1), dtype=np.float64))

# ---------------------------------
# MICROPHONE FUNCTION - HIGH SENSITIVITY
# ---------------------------------
def get_mic_audio_and_score():
    try:
        fs = 44100
        chunk = int(0.2 * fs)  # 200ms
        
        # Record chunk
        recording = sd.rec(chunk, samplerate=fs, channels=1, dtype='float32')
        sd.wait()
        
        audio = recording[:, 0]
        
        # Calculate RMS
        rms = np.sqrt(np.mean(audio**2))
        
        # Convert to dB
        epsilon = 1e-10
        db = 20 * np.log10(rms + epsilon)
        
        # Establish silence level on first run
        if st.session_state.silence_level is None:
            st.session_state.silence_level = db
            print(f"🔇 Silence level set to: {db:.2f} dB")
            return audio, fs, 0.0
        
        # Calculate how much louder than silence
        db_above_silence = db - st.session_state.silence_level
        
        # HIGH SENSITIVITY: Each 0.1 dB above silence = 10% volume
        # This will make small changes show big movements
        volume = db_above_silence * 100  # 100% per dB (very sensitive!)
        
        # Ensure it doesn't go below 0
        if volume < 0:
            volume = 0
            
        # Cap at 100
        if volume > 100:
            volume = 100
        
        # Apply smoothing (less smoothing for more responsiveness)
        smoothed_volume = st.session_state.prev_sound * 0.3 + volume * 0.7
        st.session_state.prev_sound = smoothed_volume
        
        print(f"📊 dB: {db:.2f} | Silence: {st.session_state.silence_level:.2f} | Above: {db_above_silence:.2f} | Volume: {smoothed_volume:.2f}")
        
        return audio, fs, smoothed_volume

    except Exception as e:
        print("Microphone error:", e)
        return None, 44100, 0.0

# ---------------------------------
# INITIALIZE CAMERA
# ---------------------------------
if st.session_state.cap is None and st.session_state.run:
    st.session_state.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    time.sleep(1)

# ---------------------------------
# MAIN LOOP
# ---------------------------------
try:
    while st.session_state.run:
        # Check if camera is available
        if st.session_state.cap is None or not st.session_state.cap.isOpened():
            st.error("❌ Camera not available")
            break

        # -------- CAMERA --------
        ret, frame = st.session_state.cap.read()
        if not ret or frame is None:
            st.warning("⚠ Webcam frame not received.")
            time.sleep(0.5)
            continue

        # Initialize previous frame if needed
        if 'prev_gray' not in st.session_state:
            st.session_state.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_diff = cv2.absdiff(st.session_state.prev_gray, gray)
        
        # Calculate movement score
        move_raw = float(np.mean(frame_diff))
        
        # Establish movement silence level
        if 'move_silence' not in st.session_state:
            st.session_state.move_silence = move_raw
        
        # Movement above silence with high sensitivity
        move_above = move_raw - st.session_state.move_silence
        move_score = min(100, move_above * 2000)  # Even more sensitive
        if move_score < 0:
            move_score = 0
            
        st.session_state.prev_gray = gray

        # -------- MICROPHONE - HIGH SENSITIVITY --------
        audio, audio_fs, sound_score = get_mic_audio_and_score()

        # -------- CRY DETECTION --------
        cry_prob = 0.0
        is_cry = False

        if audio is not None:
            try:
                feats = audio_to_features(audio, audio_fs)
                prob = cry_model.predict_proba(feats)[0]
                cry_prob = float(prob[1])
                is_cry = cry_prob >= 0.80
            except Exception as e:
                print(f"Cry detection error: {e}")

        # -------- ANOMALY DETECTION --------
        hour_now = time.localtime().tm_hour

        current_data = pd.DataFrame(
            [[hour_now, temp_sim, sound_score, move_score]],
            columns=["hour", "temp", "sound", "movement"]
        )

        try:
            prediction = anomaly_model.predict(current_data)[0]
            anomaly_score = anomaly_model.decision_function(current_data)[0]
            is_anomaly = prediction == -1
        except Exception as e:
            print(f"Anomaly detection error: {e}")
            is_anomaly = False
            anomaly_score = 0

        # -------- DECISION DISPLAY --------
        if is_anomaly and is_cry:
            status_box.error("🚨 HIGH RISK: Cry + Abnormal")
        elif is_cry:
            status_box.warning("👶 Cry Detected")
        elif is_anomaly:
            status_box.error("🚨 Abnormal Behavior")
        else:
            status_box.success("✅ Normal")

        # -------- CHART UPDATE --------
        # Store values in buffer
        st.session_state.sound_buffer.append(float(sound_score))
        st.session_state.move_buffer.append(float(move_score))
        
        # Keep only last 50 points
        if len(st.session_state.sound_buffer) > 50:
            st.session_state.sound_buffer.pop(0)
        if len(st.session_state.move_buffer) > 50:
            st.session_state.move_buffer.pop(0)
        
        # Update charts
        st.session_state.counter += 1
        
        if len(st.session_state.sound_buffer) > 0:
            sound_df = pd.DataFrame({
                'sound': st.session_state.sound_buffer
            })
            move_df = pd.DataFrame({
                'movement': st.session_state.move_buffer
            })
            
            # Update charts
            sound_chart.line_chart(sound_df)
            move_chart.line_chart(move_df)

        # Print values
        print(f"🎤 FINAL SOUND: {sound_score:.2f} | 📹 MOVEMENT: {move_score:.2f}")
        print("-" * 50)

        time.sleep(0.2)

except Exception as e:
    st.error(f"Error in main loop: {e}")
    print(f"Main loop error: {e}")

finally:
    # Cleanup
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None
    st.session_state.run = False