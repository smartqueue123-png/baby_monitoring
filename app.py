# import streamlit as st
# import cv2
# import sounddevice as sd
# import numpy as np
# import pandas as pd
# import joblib
# import time
# import mysql.connector
# from cry_infer import audio_to_features

# # ---------------------------------
# # PAGE CONFIG
# # ---------------------------------
# st.set_page_config(page_title="AI Baby Monitor", page_icon="👶")
# st.title("👶 Smart Baby Behavior Monitor")

# # ---------------------------------
# # MYSQL CONNECTION
# # ---------------------------------
# def get_db_connection():
#     return mysql.connector.connect(
#         host="localhost",
#         user="root",
#         password="root1234",  # 🔴 CHANGE THIS
#         database="baby_monitor"
#     )

# def log_to_mysql(hour, temp, sound, movement, cry_prob, anomaly_score, is_cry, is_anomaly):
#     try:
#         conn = get_db_connection()
#         cursor = conn.cursor()
        
#         query = """
#         INSERT INTO monitor_activity
#         (hour, temperature, sound_level, movement_level,
#          cry_probability, anomaly_score, is_cry, is_anomaly)
#         VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
#         """
        
#         values = (
#             hour,
#             temp,
#             sound,
#             movement,
#             cry_prob,
#             anomaly_score,
#             int(is_cry),
#             int(is_anomaly)
#         )
        
#         cursor.execute(query, values)
#         conn.commit()
        
#         cursor.close()
#         conn.close()
        
#     except Exception as e:
#         st.error(f"MySQL Error: {e}")

# # ---------------------------------
# # LOAD MODELS
# # ---------------------------------
# try:
#     anomaly_model = joblib.load("baby_model.pkl")
#     cry_model = joblib.load("cry_model.pkl")
#     st.success("✅ Models loaded successfully")
# except Exception as e:
#     st.error(f"❌ Model files not found: {e}")
#     st.stop()

# # ---------------------------------
# # IMPROVED AUDIO FUNCTION
# # ---------------------------------
# def get_mic_audio_and_score():
#     try:
#         fs = 44100
#         # Record 2 seconds for better feature extraction
#         duration = 2.0
#         chunk = int(duration * fs)
        
#         # Use device 1 (Microphone) from your test
#         recording = sd.rec(chunk, samplerate=fs, channels=1, dtype='float32', device=1)
#         sd.wait()
        
#         # Flatten the audio array
#         audio = recording.flatten()
        
#         # Calculate RMS and scale to 0-100
#         rms = np.sqrt(np.mean(audio**2))
        
#         # Scale factor adjusted based on your test (rms=0.037 gives ~74)
#         loudness = float(min(rms * 2000, 100.0))
        
#         # For debugging - you can remove this later
#         print(f"Debug Audio - RMS: {rms:.4f}, Loudness: {loudness:.2f}")
        
#         return audio, fs, loudness
        
#     except Exception as e:
#         print(f"Microphone error: {e}")
#         # Return silent audio
#         return np.zeros(int(2.0 * 44100)), 44100, 0.0

# # ---------------------------------
# # CRY DETECTION FUNCTION
# # ---------------------------------
# def detect_cry(audio, fs):
#     try:
#         if audio is None or len(audio) == 0 or np.all(audio == 0):
#             return 0.0, False
        
#         # Extract features
#         feats = audio_to_features(audio, fs)
        
#         # Ensure features are in correct shape
#         if len(feats.shape) == 1:
#             feats = feats.reshape(1, -1)
        
#         print(f"Debug Features - Shape: {feats.shape}")
        
#         # Get cry probability
#         if hasattr(cry_model, 'predict_proba'):
#             prob = cry_model.predict_proba(feats)[0]
#             # Handle both binary and multi-class
#             if len(prob) > 1:
#                 cry_prob = float(prob[1])  # Probability of positive class
#             else:
#                 cry_prob = float(prob[0])
#         else:
#             # Fallback for models without predict_proba
#             cry_prob = float(cry_model.decision_function(feats)[0])
#             # Normalize to 0-1 using sigmoid
#             cry_prob = 1.0 / (1.0 + np.exp(-cry_prob))
        
#         is_cry = cry_prob >= 0.80
        
#         print(f"Debug Cry - Probability: {cry_prob:.3f}, Is Cry: {is_cry}")
        
#         return cry_prob, is_cry
        
#     except Exception as e:
#         print(f"Cry detection error: {e}")
#         return 0.0, False

# # ---------------------------------
# # UI CONTROLS
# # ---------------------------------
# col1, col2 = st.columns(2)

# with col1:
#     temp_sim = st.slider("Simulated Temperature (°C)", 18, 35, 22)

# with col2:
#     status_box = st.empty()

# # Add audio test section
# with st.expander("Audio Test"):
#     if st.button("Test Microphone"):
#         with st.spinner("Recording for 2 seconds..."):
#             test_audio, test_fs, test_level = get_mic_audio_and_score()
#             st.write(f"Audio level: {test_level:.2f}/100")
#             if test_level > 10:
#                 st.success("✅ Microphone working!")
#             else:
#                 st.warning("⚠ Low audio level - try speaking louder")

# if "run" not in st.session_state:
#     st.session_state.run = False

# col_start, col_stop = st.columns(2)
# with col_start:
#     if st.button("▶ Start Monitoring"):
#         st.session_state.run = True

# with col_stop:
#     if st.button("⏹ Stop"):
#         st.session_state.run = False

# # Create charts
# sound_chart = st.line_chart(np.zeros((1, 1)), use_container_width=True)
# move_chart = st.line_chart(np.zeros((1, 1)), use_container_width=True)

# # Add metrics display
# col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
# with col_metrics1:
#     sound_metric = st.empty()
# with col_metrics2:
#     cry_metric = st.empty()
# with col_metrics3:
#     anomaly_metric = st.empty()

# # ---------------------------------
# # SAFE CAMERA INITIALIZATION
# # ---------------------------------
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# if not cap.isOpened():
#     st.error("❌ Webcam not detected.")
#     st.stop()

# time.sleep(1)

# ret, prev_frame = cap.read()

# if not ret or prev_frame is None:
#     st.error("❌ Failed to read from webcam.")
#     st.stop()

# prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# # ---------------------------------
# # MAIN LOOP
# # ---------------------------------
# if st.session_state.run:
#     st.info("🟢 Monitoring active...")
    
#     # Initialize charts with empty data
#     sound_history = []
#     move_history = []
    
#     while st.session_state.run:
#         # -------- CAMERA MOTION DETECTION --------
#         ret, frame = cap.read()
#         if not ret or frame is None:
#             st.warning("⚠ Webcam frame not received.")
#             break
        
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         frame_diff = cv2.absdiff(prev_gray, gray)
#         move_score = float(np.mean(frame_diff))
#         prev_gray = gray
        
#         # -------- MICROPHONE AUDIO CAPTURE --------
#         audio, audio_fs, sound_score = get_mic_audio_and_score()
        
#         # -------- CRY DETECTION --------
#         cry_prob, is_cry = detect_cry(audio, audio_fs)
        
#         # -------- ANOMALY DETECTION --------
#         hour_now = time.localtime().tm_hour
        
#         current_data = pd.DataFrame(
#             [[hour_now, temp_sim, sound_score, move_score]],
#             columns=["hour", "temp", "sound", "movement"]
#         )
        
#         try:
#             prediction = anomaly_model.predict(current_data)[0]
#             # Handle different anomaly detection model types
#             if hasattr(anomaly_model, 'decision_function'):
#                 anomaly_score = anomaly_model.decision_function(current_data)[0]
#             else:
#                 anomaly_score = 0.0
#             is_anomaly = prediction == -1
#         except Exception as e:
#             print(f"Anomaly detection error: {e}")
#             anomaly_score = 0.0
#             is_anomaly = False
        
#         # -------- UPDATE METRICS --------
#         sound_metric.metric("Sound Level", f"{sound_score:.1f}/100")
#         cry_metric.metric("Cry Probability", f"{cry_prob*100:.1f}%")
#         anomaly_metric.metric("Anomaly Score", f"{anomaly_score:.2f}")
        
#         # -------- DECISION DISPLAY --------
#         if is_anomaly and is_cry:
#             status_box.error("🚨 HIGH RISK: Cry + Abnormal Behavior")
#         elif is_cry:
#             status_box.warning("👶 Cry Detected")
#         elif is_anomaly:
#             status_box.error("🚨 Abnormal Behavior Detected")
#         else:
#             status_box.success("✅ Normal")
        
#         # -------- UPDATE CHARTS --------
#         # Keep last 50 values for charts
#         sound_history.append(sound_score)
#         move_history.append(move_score)
        
#         if len(sound_history) > 50:
#             sound_history.pop(0)
#         if len(move_history) > 50:
#             move_history.pop(0)
        
#         # Update line charts
#         sound_df = pd.DataFrame(sound_history, columns=["Sound Level"])
#         move_df = pd.DataFrame(move_history, columns=["Movement"])
        
#         sound_chart.line_chart(sound_df)
#         move_chart.line_chart(move_df)
        
#         # -------- STORE TO MYSQL --------
#         log_to_mysql(
#             hour_now,
#             temp_sim,
#             sound_score,
#             move_score,
#             cry_prob,
#             anomaly_score,
#             is_cry,
#             is_anomaly
#         )
        
#         # Small sleep to prevent overwhelming
#         time.sleep(0.1)

# # ---------------------------------
# # CLEANUP
# # ---------------------------------
# cap.release()
# st.success("🛑 Monitoring stopped")



















import streamlit as st
import cv2
import sounddevice as sd
import numpy as np
import pandas as pd
import joblib
import time
import mysql.connector
from cry_infer import audio_to_features

# ============================================
# NEW: Import Baby State Detector
# ============================================
class BabyStateDetector:
    """Real-time baby state detection using unsupervised learning"""
    
    def __init__(self):
        # Load your trained model
        try:
            model_data = joblib.load('baby_state_model.pkl')
            self.kmeans = model_data['kmeans']
            self.state_names = model_data['state_names']
            print("✅ Unsupervised baby state model loaded")
        except Exception as e:
            print(f"⚠️ Using fallback rules: {e}")
            self.kmeans = None
            self.state_names = {
                0: "😴 Deep Sleep",
                1: "😊 Active & Awake", 
                2: "😢 Intense Crying"
            }
    
    def extract_movement_features(self, frame, prev_frame):
        """Extract movement features from video frames"""
        if prev_frame is None:
            return [0, 0, 0, 0]
        
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
            prev_gray = prev_frame
        
        # Calculate frame difference
        diff = cv2.absdiff(prev_gray, gray)
        
        # Extract features (same as training)
        mean_movement = float(np.mean(diff))
        std_movement = float(np.std(diff))
        max_movement = float(np.max(diff))
        movement_range = float(np.ptp(diff))
        
        return [mean_movement, std_movement, max_movement, movement_range]
    
    def detect_state(self, frame, prev_frame):
        """Detect current baby state from video frame"""
        # Extract features
        features = self.extract_movement_features(frame, prev_frame)
        
        if self.kmeans:
            # Use ML model
            cluster = self.kmeans.predict([features])[0]
            state = self.state_names.get(cluster, f"State {cluster}")
            
            # Get confidence
            distances = np.linalg.norm(
                self.kmeans.cluster_centers_ - features, axis=1
            )
            confidence = 1 - (distances[cluster] / (np.sum(distances) + 0.001))
            
            return state, confidence, features
        else:
            # Fallback rules
            if features[0] < 2:
                return "😴 Deep Sleep", 0.7, features
            elif features[0] < 5:
                return "😊 Active & Awake", 0.6, features
            else:
                return "😢 Intense Crying", 0.8, features

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
        password="root1234",  # 🔴 CHANGE THIS
        database="baby_monitor"
    )

def log_to_mysql(hour, temp, sound, movement, cry_prob, anomaly_score, is_cry, is_anomaly, baby_state):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Add baby_state to your table if you have it, otherwise comment out
        query = """
        INSERT INTO monitor_activity
        (hour, temperature, sound_level, movement_level,
         cry_probability, anomaly_score, is_cry, is_anomaly, baby_state)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """
        
        values = (
            hour,
            temp,
            sound,
            movement,
            cry_prob,
            anomaly_score,
            int(is_cry),
            int(is_anomaly),
            baby_state
        )
        
        cursor.execute(query, values)
        conn.commit()
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        # If baby_state column doesn't exist, log without it
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            query = """
            INSERT INTO monitor_activity
            (hour, temperature, sound_level, movement_level,
             cry_probability, anomaly_score, is_cry, is_anomaly)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
            """
            values = (hour, temp, sound, movement, cry_prob, anomaly_score, int(is_cry), int(is_anomaly))
            cursor.execute(query, values)
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e2:
            st.error(f"MySQL Error: {e2}")

# ---------------------------------
# LOAD MODELS
# ---------------------------------
try:
    anomaly_model = joblib.load("baby_model.pkl")
    cry_model = joblib.load("cry_model.pkl")
    st.success("✅ Models loaded successfully")
except Exception as e:
    st.error(f"❌ Model files not found: {e}")
    st.stop()

# ---------------------------------
# NEW: Initialize Baby State Detector
# ---------------------------------
state_detector = BabyStateDetector()
if state_detector.kmeans:
    st.sidebar.success("✅ Baby state detection ready")
else:
    st.sidebar.warning("⚠️ Using fallback state detection")

# ---------------------------------
# AUDIO FUNCTION
# ---------------------------------
def get_mic_audio_and_score():
    try:
        fs = 44100
        duration = 2.0
        chunk = int(duration * fs)
        
        recording = sd.rec(chunk, samplerate=fs, channels=1, dtype='float32', device=1)
        sd.wait()
        
        audio = recording.flatten()
        rms = np.sqrt(np.mean(audio**2))
        loudness = float(min(rms * 2000, 100.0))
        
        return audio, fs, loudness
        
    except Exception as e:
        print(f"Microphone error: {e}")
        return np.zeros(int(2.0 * 44100)), 44100, 0.0

# ---------------------------------
# CRY DETECTION FUNCTION
# ---------------------------------
def detect_cry(audio, fs):
    try:
        if audio is None or len(audio) == 0 or np.all(audio == 0):
            return 0.0, False
        
        feats = audio_to_features(audio, fs)
        
        if len(feats.shape) == 1:
            feats = feats.reshape(1, -1)
        
        if hasattr(cry_model, 'predict_proba'):
            prob = cry_model.predict_proba(feats)[0]
            if len(prob) > 1:
                cry_prob = float(prob[1])
            else:
                cry_prob = float(prob[0])
        else:
            cry_prob = float(cry_model.decision_function(feats)[0])
            cry_prob = 1.0 / (1.0 + np.exp(-cry_prob))
        
        is_cry = cry_prob >= 0.80
        
        return cry_prob, is_cry
        
    except Exception as e:
        print(f"Cry detection error: {e}")
        return 0.0, False

# ---------------------------------
# UI CONTROLS
# ---------------------------------
col1, col2 = st.columns(2)

with col1:
    temp_sim = st.slider("Simulated Temperature (°C)", 18, 35, 22)

with col2:
    status_box = st.empty()

# Add audio test section
with st.expander("Audio Test"):
    if st.button("Test Microphone"):
        with st.spinner("Recording for 2 seconds..."):
            test_audio, test_fs, test_level = get_mic_audio_and_score()
            st.write(f"Audio level: {test_level:.2f}/100")
            if test_level > 10:
                st.success("✅ Microphone working!")
            else:
                st.warning("⚠ Low audio level - try speaking louder")

if "run" not in st.session_state:
    st.session_state.run = False

col_start, col_stop = st.columns(2)
with col_start:
    if st.button("▶ Start Monitoring"):
        st.session_state.run = True

with col_stop:
    if st.button("⏹ Stop"):
        st.session_state.run = False

# Create charts
sound_chart = st.line_chart(np.zeros((1, 1)), use_container_width=True)
move_chart = st.line_chart(np.zeros((1, 1)), use_container_width=True)

# Add metrics display
col_metrics1, col_metrics2, col_metrics3, col_metrics4 = st.columns(4)
with col_metrics1:
    sound_metric = st.empty()
with col_metrics2:
    cry_metric = st.empty()
with col_metrics3:
    anomaly_metric = st.empty()
with col_metrics4:
    state_metric = st.empty()

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

# NEW: Store previous frame for state detection
prev_frame_state = prev_frame.copy()

# ---------------------------------
# MAIN LOOP
# ---------------------------------
if st.session_state.run:
    st.info("🟢 Monitoring active...")
    
    # Initialize charts with empty data
    sound_history = []
    move_history = []
    state_history = []
    
    while st.session_state.run:
        # -------- CAMERA MOTION DETECTION --------
        ret, frame = cap.read()
        if not ret or frame is None:
            st.warning("⚠ Webcam frame not received.")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_diff = cv2.absdiff(prev_gray, gray)
        move_score = float(np.mean(frame_diff))
        prev_gray = gray
        
        # -------- NEW: BABY STATE DETECTION --------
        baby_state, state_confidence, movement_features = state_detector.detect_state(
            frame, prev_frame_state
        )
        prev_frame_state = frame.copy()
        
        # -------- MICROPHONE AUDIO CAPTURE --------
        audio, audio_fs, sound_score = get_mic_audio_and_score()
        
        # -------- CRY DETECTION --------
        cry_prob, is_cry = detect_cry(audio, audio_fs)
        
        # -------- ANOMALY DETECTION --------
        hour_now = time.localtime().tm_hour
        
        current_data = pd.DataFrame(
            [[hour_now, temp_sim, sound_score, move_score]],
            columns=["hour", "temp", "sound", "movement"]
        )
        
        try:
            prediction = anomaly_model.predict(current_data)[0]
            if hasattr(anomaly_model, 'decision_function'):
                anomaly_score = anomaly_model.decision_function(current_data)[0]
            else:
                anomaly_score = 0.0
            is_anomaly = prediction == -1
        except Exception as e:
            print(f"Anomaly detection error: {e}")
            anomaly_score = 0.0
            is_anomaly = False
        
        # -------- UPDATE METRICS --------
        sound_metric.metric("Sound Level", f"{sound_score:.1f}/100")
        cry_metric.metric("Cry Probability", f"{cry_prob*100:.1f}%")
        anomaly_metric.metric("Anomaly Score", f"{anomaly_score:.2f}")
        state_metric.metric("Baby State", baby_state, delta=None)
        
        # -------- DECISION DISPLAY --------
        if is_anomaly and is_cry:
            status_box.error(f"🚨 HIGH RISK: {baby_state} + Abnormal")
        elif is_cry:
            status_box.warning(f"👶 {baby_state} - Cry Detected")
        elif is_anomaly:
            status_box.error(f"🚨 Abnormal Behavior - {baby_state}")
        else:
            status_box.success(f"✅ {baby_state}")
        
        # Show state confidence in expander
        with st.sidebar:
            st.markdown("---")
            st.subheader("👶 State Details")
            st.markdown(f"### {baby_state}")
            st.progress(state_confidence)
            st.caption(f"Confidence: {state_confidence:.1%}")
            
            with st.expander("Movement Features"):
                st.write(f"Mean: {movement_features[0]:.2f}")
                st.write(f"Std: {movement_features[1]:.2f}")
                st.write(f"Max: {movement_features[2]:.2f}")
                st.write(f"Range: {movement_features[3]:.2f}")
        
        # -------- UPDATE CHARTS --------
        sound_history.append(sound_score)
        move_history.append(move_score)
        state_history.append(baby_state)
        
        if len(sound_history) > 50:
            sound_history.pop(0)
            move_history.pop(0)
            state_history.pop(0)
        
        # Update line charts
        sound_df = pd.DataFrame(sound_history, columns=["Sound Level"])
        move_df = pd.DataFrame(move_history, columns=["Movement"])
        
        sound_chart.line_chart(sound_df)
        move_chart.line_chart(move_df)
        
        # -------- STORE TO MYSQL --------
        log_to_mysql(
            hour_now,
            temp_sim,
            sound_score,
            move_score,
            cry_prob,
            anomaly_score,
            is_cry,
            is_anomaly,
            baby_state
        )
        
        # Small sleep to prevent overwhelming
        time.sleep(0.1)

# ---------------------------------
# CLEANUP
# ---------------------------------
cap.release()
st.success("🛑 Monitoring stopped")