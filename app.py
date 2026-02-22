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
# UNSUPERVISED LEARNING: Baby State Detector
# ============================================
class BabyStateDetector:
    """UNSUPERVISED LEARNING: Discovers baby states without labels"""
    
    def __init__(self):
        # Load your trained unsupervised model
        try:
            model_data = joblib.load('baby_state_model.pkl')
            self.kmeans = model_data['kmeans']
            self.state_names = model_data['state_names']
            print("✅ UNSUPERVISED MODEL LOADED: K-Means clustering")
            print(f"   Discovered states: {list(self.state_names.values())}")
        except Exception as e:
            print(f"⚠️ Unsupervised model not found, using fallback: {e}")
            self.kmeans = None
            self.state_names = {
                0: "😴 Deep Sleep",
                1: "😊 Active & Awake", 
                2: "😢 Intense Crying"
            }
    
    def extract_movement_features(self, frame, prev_frame):
        """Extract movement features for unsupervised clustering"""
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
        """UNSUPERVISED: Predict baby state using K-Means clustering"""
        # Extract features
        features = self.extract_movement_features(frame, prev_frame)
        
        if self.kmeans:
            # Use unsupervised ML model
            cluster = self.kmeans.predict([features])[0]
            state = self.state_names.get(cluster, f"State {cluster}")
            
            # Get confidence (distance to cluster center)
            distances = np.linalg.norm(
                self.kmeans.cluster_centers_ - features, axis=1
            )
            confidence = 1 - (distances[cluster] / (np.sum(distances) + 0.001))
            
            return state, confidence, features, cluster
        else:
            # Fallback rules
            if features[0] < 2:
                return "😴 Deep Sleep", 0.7, features, 0
            elif features[0] < 5:
                return "😊 Active & Awake", 0.6, features, 1
            else:
                return "😢 Intense Crying", 0.8, features, 2

# ============================================
# SUPERVISED LEARNING: Cry Detection Classifier
# ============================================
class CryDetector:
    """SUPERVISED LEARNING: Binary classification (cry vs not-cry)"""
    
    def __init__(self):
        try:
            self.model = joblib.load("cry_model.pkl")
            print("✅ SUPERVISED MODEL LOADED: Random Forest Classifier")
            print("   Trained on 77 cry + 60 not-cry samples")
        except Exception as e:
            print(f"❌ Supervised model not found: {e}")
            self.model = None
    
    def detect_cry(self, audio, fs):
        """SUPERVISED: Predict if audio contains crying"""
        try:
            if audio is None or len(audio) == 0 or np.all(audio == 0):
                return 0.0, False
            
            # Extract features
            feats = audio_to_features(audio, fs)
            
            if len(feats.shape) == 1:
                feats = feats.reshape(1, -1)
            
            if hasattr(self.model, 'predict_proba'):
                prob = self.model.predict_proba(feats)[0]
                if len(prob) > 1:
                    cry_prob = float(prob[1])  # Probability of positive class
                else:
                    cry_prob = float(prob[0])
            else:
                cry_prob = float(self.model.decision_function(feats)[0])
                cry_prob = 1.0 / (1.0 + np.exp(-cry_prob))
            
            is_cry = cry_prob >= 0.80
            
            return cry_prob, is_cry
            
        except Exception as e:
            print(f"Cry detection error: {e}")
            return 0.0, False

# ============================================
# UNSUPERVISED LEARNING: Anomaly Detection
# ============================================
class AnomalyDetector:
    """UNSUPERVISED LEARNING: IsolationForest for anomaly detection"""
    
    def __init__(self):
        try:
            self.model = joblib.load("baby_model.pkl")
            print("✅ UNSUPERVISED ANOMALY MODEL LOADED: IsolationForest")
        except Exception as e:
            print(f"❌ Anomaly model not found: {e}")
            self.model = None
    
    def detect_anomaly(self, hour, temp, sound, movement):
        """UNSUPERVISED: Detect unusual patterns"""
        try:
            current_data = pd.DataFrame(
                [[hour, temp, sound, movement]],
                columns=["hour", "temp", "sound", "movement"]
            )
            
            prediction = self.model.predict(current_data)[0]
            
            if hasattr(self.model, 'decision_function'):
                anomaly_score = self.model.decision_function(current_data)[0]
            else:
                anomaly_score = 0.0
            
            is_anomaly = prediction == -1
            
            return anomaly_score, is_anomaly
            
        except Exception as e:
            print(f"Anomaly detection error: {e}")
            return 0.0, False

# ---------------------------------
# PAGE CONFIG
# ---------------------------------
st.set_page_config(page_title="AI Baby Monitor", page_icon="👶")
st.title("👶 Smart Baby Behavior Monitor")

# ---------------------------------
# AI TECHNIQUES OVERVIEW
# ---------------------------------
with st.sidebar:
    st.header("🤖 AI Techniques Used")
    
    st.markdown("""
    ### ✅ SUPERVISED LEARNING
    - **Algorithm:** Random Forest Classifier
    - **Purpose:** Cry detection
    - **Training:** 137 labeled samples (77 cry, 60 not-cry)
    - **Output:** Cry probability (0-100%)
    
    ### ✅ UNSUPERVISED LEARNING (Clustering)
    - **Algorithm:** K-Means Clustering
    - **Purpose:** Baby state discovery
    - **Training:** 6 real YouTube videos
    - **Discovered States:** Deep Sleep, Active & Awake, Intense Crying
    
    ### ✅ UNSUPERVISED LEARNING (Anomaly Detection)
    - **Algorithm:** IsolationForest
    - **Purpose:** Detect unusual behavior
    - **Training:** Normal baby patterns
    - **Output:** Anomaly score + flag
    """)

# ---------------------------------
# MYSQL CONNECTION
# ---------------------------------
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="root1234",
        database="baby_monitor"
    )

def log_to_mysql(hour, temp, sound, movement, cry_prob, anomaly_score, is_cry, is_anomaly, baby_state, cluster_id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        query = """
        INSERT INTO monitor_activity
        (hour, temperature, sound_level, movement_level,
         cry_probability, anomaly_score, is_cry, is_anomaly, 
         baby_state, cluster_id)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
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
            baby_state,
            int(cluster_id)
        )
        
        cursor.execute(query, values)
        conn.commit()
        cursor.close()
        conn.close()
        
    except Exception as e:
        # Fallback if columns don't exist
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
# INITIALIZE ALL AI MODELS
# ---------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Model Status")

# Initialize all detectors
state_detector = BabyStateDetector()
cry_detector = CryDetector()
anomaly_detector = AnomalyDetector()

# Show status in sidebar
if state_detector.kmeans:
    st.sidebar.success("✅ Unsupervised (Clustering): Ready")
else:
    st.sidebar.warning("⚠️ Unsupervised (Clustering): Fallback")

if cry_detector.model:
    st.sidebar.success("✅ Supervised (Classification): Ready")
else:
    st.sidebar.error("❌ Supervised (Classification): Missing")

if anomaly_detector.model:
    st.sidebar.success("✅ Unsupervised (Anomaly): Ready")
else:
    st.sidebar.warning("⚠️ Unsupervised (Anomaly): Fallback")

# ---------------------------------
# UI CONTROLS
# ---------------------------------
col1, col2 = st.columns(2)

with col1:
    temp_sim = st.slider("Simulated Temperature (°C)", 18, 35, 22)

with col2:
    status_box = st.empty()

# Audio test section
with st.expander("Audio Test"):
    if st.button("Test Microphone"):
        with st.spinner("Recording for 2 seconds..."):
            fs = 44100
            recording = sd.rec(int(2.0 * fs), samplerate=fs, channels=1, dtype='float32', device=1)
            sd.wait()
            audio = recording.flatten()
            rms = np.sqrt(np.mean(audio**2))
            test_level = float(min(rms * 2000, 100.0))
            st.write(f"Audio level: {test_level:.2f}/100")
            if test_level > 10:
                st.success("✅ Microphone working!")
            else:
                st.warning("⚠ Low audio level - try speaking louder")

# Start/Stop buttons
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

# Metrics display
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
# CAMERA INITIALIZATION
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
prev_frame_state = prev_frame.copy()

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
# MAIN LOOP
# ---------------------------------
if st.session_state.run:
    st.info("🟢 Monitoring active...")
    
    # Initialize history
    sound_history = []
    move_history = []
    
    while st.session_state.run:
        # -------- VIDEO PROCESSING --------
        ret, frame = cap.read()
        if not ret or frame is None:
            st.warning("⚠ Webcam frame not received.")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_diff = cv2.absdiff(prev_gray, gray)
        move_score = float(np.mean(frame_diff))
        prev_gray = gray
        
        # -------- UNSUPERVISED: BABY STATE DETECTION (Clustering) --------
        baby_state, state_confidence, movement_features, cluster_id = state_detector.detect_state(
            frame, prev_frame_state
        )
        prev_frame_state = frame.copy()
        
        # -------- AUDIO PROCESSING --------
        audio, audio_fs, sound_score = get_mic_audio_and_score()
        
        # -------- SUPERVISED: CRY DETECTION (Classification) --------
        cry_prob, is_cry = cry_detector.detect_cry(audio, audio_fs)
        
        # -------- UNSUPERVISED: ANOMALY DETECTION --------
        hour_now = time.localtime().tm_hour
        anomaly_score, is_anomaly = anomaly_detector.detect_anomaly(
            hour_now, temp_sim, sound_score, move_score
        )
        
        # -------- UPDATE METRICS --------
        sound_metric.metric("Sound Level", f"{sound_score:.1f}/100")
        cry_metric.metric("Cry Probability", f"{cry_prob*100:.1f}%")
        anomaly_metric.metric("Anomaly Score", f"{anomaly_score:.2f}")
        state_metric.metric("Baby State", baby_state, delta=None)
        
        # -------- AI TECHNIQUE SUMMARY --------
        with st.sidebar:
            st.markdown("---")
            st.subheader("🎯 Current Predictions")
            
            # Supervised
            st.markdown("**🔵 SUPERVISED:**")
            if is_cry:
                st.warning(f"👶 Cry Detected ({cry_prob*100:.1f}%)")
            else:
                st.success(f"✅ No Cry ({cry_prob*100:.1f}%)")
            
            # Unsupervised - Clustering
            st.markdown("**🟢 UNSUPERVISED (Clustering):**")
            st.info(f"{baby_state}")
            st.progress(state_confidence)
            st.caption(f"Confidence: {state_confidence:.1%} | Cluster: {cluster_id}")
            
            # Unsupervised - Anomaly
            st.markdown("**🟠 UNSUPERVISED (Anomaly):**")
            if is_anomaly:
                st.error(f"⚠️ Anomaly Detected (Score: {anomaly_score:.2f})")
            else:
                st.success(f"✅ Normal Pattern (Score: {anomaly_score:.2f})")
        
        # -------- DECISION DISPLAY --------
        if is_anomaly and is_cry:
            status_box.error(f"🚨 HIGH RISK: {baby_state} + Abnormal Pattern")
        elif is_cry:
            status_box.warning(f"👶 {baby_state} - Cry Detected")
        elif is_anomaly:
            status_box.error(f"🚨 Abnormal Pattern - {baby_state}")
        else:
            status_box.success(f"✅ {baby_state}")
        
        # -------- UPDATE CHARTS --------
        sound_history.append(sound_score)
        move_history.append(move_score)
        
        if len(sound_history) > 50:
            sound_history.pop(0)
            move_history.pop(0)
        
        sound_chart.line_chart(pd.DataFrame(sound_history, columns=["Sound Level"]))
        move_chart.line_chart(pd.DataFrame(move_history, columns=["Movement"]))
        
        # -------- LOG TO DATABASE --------
        log_to_mysql(
            hour_now,
            temp_sim,
            sound_score,
            move_score,
            cry_prob,
            anomaly_score,
            is_cry,
            is_anomaly,
            baby_state,
            cluster_id
        )
        
        time.sleep(0.1)

# ---------------------------------
# CLEANUP
# ---------------------------------
cap.release()
st.success("🛑 Monitoring stopped")















# import streamlit as st
# import cv2
# import sounddevice as sd
# import numpy as np
# import pandas as pd
# import joblib
# import time
# import mysql.connector
# from cry_infer import audio_to_features
# from datetime import datetime, timedelta
# import plotly.graph_objects as go
# import plotly.express as px
# from collections import deque
# import threading
# import queue
# import atexit

# # ---------------------------------
# # PAGE CONFIG
# # ---------------------------------
# st.set_page_config(
#     page_title="AI Baby Monitor", 
#     page_icon="👶",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS for better styling
# st.markdown("""
# <style>
#     .risk-high {
#         background-color: #ff4b4b;
#         padding: 20px;
#         border-radius: 10px;
#         color: white;
#         text-align: center;
#         font-size: 24px;
#         font-weight: bold;
#     }
#     .risk-moderate {
#         background-color: #ffa64b;
#         padding: 20px;
#         border-radius: 10px;
#         color: white;
#         text-align: center;
#         font-size: 24px;
#         font-weight: bold;
#     }
#     .risk-low {
#         background-color: #4CAF50;
#         padding: 20px;
#         border-radius: 10px;
#         color: white;
#         text-align: center;
#         font-size: 24px;
#         font-weight: bold;
#     }
#     .explanation-box {
#         background-color: #f0f2f6;
#         padding: 15px;
#         border-radius: 10px;
#         border-left: 5px solid #0066cc;
#     }
#     .metric-card {
#         background-color: white;
#         padding: 15px;
#         border-radius: 10px;
#         box-shadow: 0 2px 4px rgba(0,0,0,0.1);
#     }
# </style>
# """, unsafe_allow_html=True)

# # ---------------------------------
# # TITLE WITH DESCRIPTION
# # ---------------------------------
# st.title("👶 Smart Baby Behavior Monitor")
# st.markdown("""
# *An intelligent monitoring system that combines supervised cry detection and unsupervised behavioral anomaly detection 
# into a hybrid risk scoring model for enhanced baby safety.*
# """)

# # ---------------------------------
# # SESSION STATE INITIALIZATION
# # ---------------------------------
# if 'history' not in st.session_state:
#     st.session_state.history = {
#         'timestamps': deque(maxlen=50),
#         'sound_levels': deque(maxlen=50),
#         'movement_levels': deque(maxlen=50),
#         'cry_probs': deque(maxlen=50),
#         'risk_scores': deque(maxlen=50),
#         'anomaly_scores': deque(maxlen=50),
#         'is_cry': deque(maxlen=50),
#         'is_anomaly': deque(maxlen=50)
#     }

# if 'avg_baseline' not in st.session_state:
#     st.session_state.avg_baseline = {
#         'movement': 5.0,
#         'sound': 10.0,
#         'cry_prob': 0.1
#     }

# if 'monitoring_active' not in st.session_state:
#     st.session_state.monitoring_active = False

# if 'data_queue' not in st.session_state:
#     st.session_state.data_queue = queue.Queue()

# if 'monitoring_thread' not in st.session_state:
#     st.session_state.monitoring_thread = None
    
# if 'stop_event' not in st.session_state:
#     st.session_state.stop_event = threading.Event()
    
# if 'last_update' not in st.session_state:
#     st.session_state.last_update = time.time()

# # ---------------------------------
# # SIDEBAR - SYSTEM CONFIGURATION
# # ---------------------------------
# with st.sidebar:
#     st.header("⚙️ System Configuration")
    
#     # Model status
#     st.subheader("Model Status")
#     try:
#         anomaly_model = joblib.load("baby_model.pkl")
#         cry_model = joblib.load("cry_model.pkl")
#         st.success("✅ All models loaded")
#     except Exception as e:
#         st.error(f"❌ Model loading failed: {e}")
#         st.stop()
    
#     # Sensor simulation
#     st.subheader("Environment Settings")
#     temp_sim = st.slider("Temperature (°C)", 18, 35, 22, 
#                         help="Simulated room temperature")
    
#     # Threshold settings
#     st.subheader("Detection Thresholds")
#     cry_threshold = st.slider("Cry Detection Threshold", 0.5, 0.95, 0.80, 0.05,
#                              help="Probability above this triggers cry detection")
    
#     # Audio test
#     st.subheader("Audio Test")
#     if st.button("🎤 Test Microphone"):
#         with st.spinner("Recording for 2 seconds..."):
#             try:
#                 fs = 44100
#                 recording = sd.rec(int(2.0 * fs), samplerate=fs, 
#                                  channels=1, dtype='float32', device=1)
#                 sd.wait()
#                 rms = np.sqrt(np.mean(recording**2))
#                 level = float(min(rms * 2000, 100.0))
                
#                 if level > 10:
#                     st.success(f"✅ Microphone working! Level: {level:.1f}/100")
#                 else:
#                     st.warning(f"⚠ Low audio level: {level:.1f}/100")
#             except Exception as e:
#                 st.error(f"❌ Microphone error: {e}")

# # ---------------------------------
# # MYSQL CONNECTION
# # ---------------------------------
# @st.cache_resource
# def get_db_connection():
#     return mysql.connector.connect(
#         host="localhost",
#         user="root", 
#         password="root1234",
#         database="baby_monitor"
#     )

# def log_to_mysql(hour, temp, sound, movement, cry_prob, anomaly_score, 
#                  is_cry, is_anomaly, risk_score):
#     try:
#         conn = get_db_connection()
#         cursor = conn.cursor()
        
#         query = """
#         INSERT INTO monitor_activity
#         (hour, temperature, sound_level, movement_level,
#          cry_probability, anomaly_score, is_cry, is_anomaly, risk_score)
#         VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
#         """
        
#         cursor.execute(query, (
#             hour, temp, sound, movement, cry_prob, 
#             anomaly_score, int(is_cry), int(is_anomaly), risk_score
#         ))
#         conn.commit()
#         cursor.close()
#         conn.close()
#     except Exception as e:
#         st.error(f"MySQL Error: {e}")

# # ---------------------------------
# # AUDIO CAPTURE FUNCTION
# # ---------------------------------
# def capture_audio():
#     try:
#         fs = 44100
#         duration = 2.0
#         recording = sd.rec(int(duration * fs), samplerate=fs, 
#                          channels=1, dtype='float32', device=1)
#         sd.wait()
        
#         audio = recording.flatten()
#         rms = np.sqrt(np.mean(audio**2))
#         sound_level = float(min(rms * 2000, 100.0))
        
#         return audio, fs, sound_level
#     except Exception as e:
#         print(f"Audio error: {e}")
#         return np.zeros(int(2.0 * 44100)), 44100, 0.0

# # ---------------------------------
# # CRY DETECTION FUNCTION
# # ---------------------------------
# def detect_cry(audio, fs, model, threshold):
#     try:
#         if audio is None or len(audio) == 0 or np.all(audio == 0):
#             return 0.0, False
        
#         feats = audio_to_features(audio, fs)
#         if len(feats.shape) == 1:
#             feats = feats.reshape(1, -1)
        
#         if hasattr(model, 'predict_proba'):
#             prob = model.predict_proba(feats)[0]
#             cry_prob = float(prob[1]) if len(prob) > 1 else float(prob[0])
#         else:
#             cry_prob = float(model.decision_function(feats)[0])
#             cry_prob = 1.0 / (1.0 + np.exp(-cry_prob))
        
#         is_cry = cry_prob >= threshold
        
#         return cry_prob, is_cry
        
#     except Exception as e:
#         print(f"Cry detection error: {e}")
#         return 0.0, False

# # ---------------------------------
# # RISK SCORE CALCULATION
# # ---------------------------------
# def calculate_risk_score(cry_prob, anomaly_score, sound_level, movement_level):
#     # Normalize anomaly score (assuming it can be negative/positive)
#     norm_anomaly = 1 / (1 + np.exp(-anomaly_score))  # Sigmoid normalization
    
#     # Weighted combination
#     risk_score = (
#         cry_prob * 0.5 +                    # Cry is most important
#         norm_anomaly * 0.3 +                 # Anomaly detection
#         (sound_level / 100) * 0.1 +          # Sound level
#         min(movement_level / 50, 1.0) * 0.1   # Movement (cap at 1.0)
#     )
    
#     return min(risk_score, 1.0)  # Ensure between 0-1

# # ---------------------------------
# # GET RISK LEVEL AND COLOR
# # ---------------------------------
# def get_risk_level(risk_score):
#     if risk_score < 0.3:
#         return "LOW", "🟢", "risk-low"
#     elif risk_score < 0.6:
#         return "MODERATE", "🟡", "risk-moderate"
#     else:
#         return "HIGH", "🔴", "risk-high"

# # ---------------------------------
# # GENERATE EXPLANATION
# # ---------------------------------
# def generate_explanation(cry_prob, is_cry, is_anomaly, anomaly_score, 
#                         sound_level, movement_level, hour, baseline):
#     explanations = []
    
#     if is_cry:
#         explanations.append(f"• Baby crying detected with {cry_prob*100:.1f}% confidence")
    
#     if is_anomaly:
#         explanations.append(f"• Unusual behavior pattern detected (anomaly score: {anomaly_score:.2f})")
    
#     # Compare with baselines
#     if movement_level > baseline['movement'] * 2:
#         explanations.append(f"• Movement is {movement_level/baseline['movement']:.1f}x higher than normal")
    
#     if sound_level > baseline['sound'] * 3 and not is_cry:
#         explanations.append(f"• Sound level significantly above baseline")
    
#     # Time-based patterns
#     if hour < 6 or hour > 22:  # Night hours
#         if movement_level > 10:
#             explanations.append(f"• Unusual movement during nighttime hours ({hour}:00)")
    
#     if not explanations:
#         return "All parameters are within normal ranges."
    
#     return "\n".join(explanations)

# # ---------------------------------
# # GET ACTION RECOMMENDATION
# # ---------------------------------
# def get_action_recommendation(risk_score, is_cry, is_anomaly):
#     if risk_score >= 0.6:
#         if is_cry and is_anomaly:
#             return "🚨 URGENT: Check baby immediately - unusual cry pattern detected"
#         elif is_cry:
#             return "👶 Immediate attention needed - baby is crying"
#         elif is_anomaly:
#             return "⚠️ Check baby - unusual movement pattern detected"
#         else:
#             return "⚠️ Monitor closely - risk indicators elevated"
#     elif risk_score >= 0.3:
#         if is_cry:
#             return "👂 Monitor audio - baby may need attention soon"
#         else:
#             return "👀 Keep observing - subtle changes detected"
#     else:
#         return "✅ No action needed - everything normal"

# # ---------------------------------
# # MONITORING FUNCTION (runs in thread)
# # ---------------------------------
# def monitoring_loop(anomaly_model, cry_model, temp_sim, cry_threshold, data_queue, stop_event):
#     # Camera initialization
#     cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
#     if not cap.isOpened():
#         data_queue.put({"error": "Camera not detected"})
#         return
    
#     time.sleep(1)
#     ret, prev_frame = cap.read()
#     if not ret:
#         data_queue.put({"error": "Failed to read from camera"})
#         cap.release()
#         return
    
#     prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
#     # Baseline collection
#     movement_baseline = 5.0
#     sound_baseline = 10.0
    
#     # Collect baseline
#     for i in range(5):
#         if stop_event.is_set():
#             cap.release()
#             return
            
#         ret, frame = cap.read()
#         if ret:
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             move = float(np.mean(cv2.absdiff(prev_gray, gray)))
#             movement_baseline = movement_baseline * 0.7 + move * 0.3
#             prev_gray = gray
#         time.sleep(0.1)
    
#     while not stop_event.is_set():
#         try:
#             # Camera capture
#             ret, frame = cap.read()
#             if not ret:
#                 break
            
#             # Motion detection
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             frame_diff = cv2.absdiff(prev_gray, gray)
#             movement_score = float(np.mean(frame_diff))
#             prev_gray = gray
            
#             # Audio capture
#             audio, audio_fs, sound_level = capture_audio()
            
#             # Cry detection
#             cry_prob, is_cry = detect_cry(audio, audio_fs, cry_model, cry_threshold)
            
#             # Anomaly detection
#             hour_now = datetime.now().hour
#             current_data = pd.DataFrame(
#                 [[hour_now, temp_sim, sound_level, movement_score]],
#                 columns=["hour", "temp", "sound", "movement"]
#             )
            
#             try:
#                 prediction = anomaly_model.predict(current_data)[0]
#                 if hasattr(anomaly_model, 'decision_function'):
#                     anomaly_score = anomaly_model.decision_function(current_data)[0]
#                 else:
#                     anomaly_score = 0.0
#                 is_anomaly = prediction == -1
#             except Exception as e:
#                 anomaly_score = 0.0
#                 is_anomaly = False
            
#             # Calculate risk score
#             risk_score = calculate_risk_score(
#                 cry_prob, anomaly_score, sound_level, movement_score
#             )
            
#             # Generate explanation and action
#             baseline = {'movement': movement_baseline, 'sound': sound_baseline}
#             explanation = generate_explanation(
#                 cry_prob, is_cry, is_anomaly, anomaly_score,
#                 sound_level, movement_score, hour_now, baseline
#             )
            
#             action = get_action_recommendation(risk_score, is_cry, is_anomaly)
            
#             # Update baselines
#             movement_baseline = movement_baseline * 0.95 + movement_score * 0.05
#             sound_baseline = sound_baseline * 0.95 + sound_level * 0.05
            
#             # Prepare data for UI
#             data = {
#                 'timestamp': datetime.now(),
#                 'sound_level': sound_level,
#                 'movement_score': movement_score,
#                 'cry_prob': cry_prob,
#                 'risk_score': risk_score,
#                 'anomaly_score': anomaly_score,
#                 'is_cry': is_cry,
#                 'is_anomaly': is_anomaly,
#                 'explanation': explanation,
#                 'action': action,
#                 'movement_baseline': movement_baseline,
#                 'sound_baseline': sound_baseline
#             }
            
#             # Send to main thread (clear old data if queue is full)
#             if data_queue.qsize() > 5:
#                 try:
#                     data_queue.get_nowait()
#                 except queue.Empty:
#                     pass
#             data_queue.put(data)
            
#             # Log to MySQL
#             try:
#                 log_to_mysql(
#                     hour_now, temp_sim, sound_level, movement_score,
#                     cry_prob, anomaly_score, is_cry, is_anomaly, risk_score
#                 )
#             except:
#                 pass  # Don't let MySQL errors stop monitoring
            
#             time.sleep(2)  # Update every 2 seconds
            
#         except Exception as e:
#             data_queue.put({"error": str(e)})
#             time.sleep(1)
    
#     cap.release()

# # ---------------------------------
# # START/STOP MONITORING
# # ---------------------------------
# col1, col2, col3 = st.columns([1, 1, 4])

# with col1:
#     if st.button("▶ Start Monitoring", type="primary", use_container_width=True):
#         if not st.session_state.monitoring_active:
#             # Reset stop event
#             st.session_state.stop_event = threading.Event()
            
#             # Clear old data
#             while not st.session_state.data_queue.empty():
#                 try:
#                     st.session_state.data_queue.get_nowait()
#                 except queue.Empty:
#                     break
            
#             # Start monitoring thread
#             st.session_state.monitoring_thread = threading.Thread(
#                 target=monitoring_loop,
#                 args=(anomaly_model, cry_model, temp_sim, cry_threshold, 
#                       st.session_state.data_queue, st.session_state.stop_event),
#                 daemon=True
#             )
#             st.session_state.monitoring_thread.start()
#             st.session_state.monitoring_active = True
#             st.session_state.last_update = time.time()
#             st.rerun()

# with col2:
#     if st.button("⏹ Stop", use_container_width=True):
#         if st.session_state.monitoring_active:
#             st.session_state.stop_event.set()
#             st.session_state.monitoring_active = False
#             st.rerun()

# with col3:
#     if st.session_state.monitoring_active:
#         if st.session_state.monitoring_thread and st.session_state.monitoring_thread.is_alive():
#             st.info("🟢 Monitoring active...")
#         else:
#             st.session_state.monitoring_active = False
#             st.warning("⚠ Monitoring thread died. Click Start again.")
#     else:
#         st.info("⏸ System ready for monitoring")

# # ---------------------------------
# # MAIN UI LAYOUT
# # ---------------------------------

# # Create tabs for different views
# tab1, tab2, tab3 = st.tabs(["📊 Live Monitor", "📈 Historical Analytics", "ℹ️ System Info"])

# with tab1:
#     # Process data from queue
#     data_processed = False
#     while not st.session_state.data_queue.empty():
#         try:
#             data = st.session_state.data_queue.get_nowait()
#             data_processed = True
            
#             if "error" in data:
#                 st.error(data["error"])
#             else:
#                 # Update session state with new data
#                 st.session_state.current_risk_score = data['risk_score']
#                 st.session_state.current_sound = data['sound_level']
#                 st.session_state.current_movement = data['movement_score']
#                 st.session_state.current_cry_prob = data['cry_prob']
#                 st.session_state.current_anomaly = data['anomaly_score']
#                 st.session_state.current_is_cry = data['is_cry']
#                 st.session_state.current_is_anomaly = data['is_anomaly']
#                 st.session_state.current_explanation = data['explanation']
#                 st.session_state.current_action = data['action']
                
#                 # Update baselines
#                 st.session_state.avg_baseline['movement'] = data['movement_baseline']
#                 st.session_state.avg_baseline['sound'] = data['sound_baseline']
                
#                 # Update history
#                 st.session_state.history['timestamps'].append(data['timestamp'])
#                 st.session_state.history['sound_levels'].append(data['sound_level'])
#                 st.session_state.history['movement_levels'].append(data['movement_score'])
#                 st.session_state.history['cry_probs'].append(data['cry_prob'])
#                 st.session_state.history['risk_scores'].append(data['risk_score'])
#                 st.session_state.history['anomaly_scores'].append(data['anomaly_score'])
#                 st.session_state.history['is_cry'].append(data['is_cry'])
#                 st.session_state.history['is_anomaly'].append(data['is_anomaly'])
                
#         except queue.Empty:
#             break
    
#     # Risk Score Display (Prominent)
#     risk_container = st.container()
#     with risk_container:
#         if 'current_risk_score' in st.session_state:
#             risk_score = st.session_state.current_risk_score
#             risk_text, risk_icon, risk_class = get_risk_level(risk_score)
            
#             st.markdown(f"""
#             <div class="{risk_class}">
#                 {risk_icon} {risk_text} RISK - Score: {risk_score:.2%}
#             </div>
#             """, unsafe_allow_html=True)
    
#     # Main metrics row
#     col_metrics1, col_metrics2, col_metrics3, col_metrics4 = st.columns(4)
    
#     with col_metrics1:
#         sound_val = st.session_state.get('current_sound', 0)
#         sound_delta = sound_val - st.session_state.avg_baseline['sound']
#         st.metric("Sound Level", f"{sound_val:.1f}/100", 
#                  delta=f"{sound_delta:+.1f}", delta_color="inverse")
    
#     with col_metrics2:
#         move_val = st.session_state.get('current_movement', 0)
#         move_delta = move_val - st.session_state.avg_baseline['movement']
#         st.metric("Movement", f"{move_val:.1f}", 
#                  delta=f"{move_delta:+.1f}", delta_color="inverse")
    
#     with col_metrics3:
#         cry_val = st.session_state.get('current_cry_prob', 0) * 100
#         st.metric("Cry Probability", f"{cry_val:.1f}%")
    
#     with col_metrics4:
#         anomaly_val = st.session_state.get('current_anomaly', 0)
#         st.metric("Anomaly Score", f"{anomaly_val:.2f}")
    
#     # Confidence indicator
#     if 'current_cry_prob' in st.session_state:
#         confidence = max(st.session_state.current_cry_prob, 
#                         1 - st.session_state.current_cry_prob)
#         st.progress(confidence, text=f"Model Confidence: {confidence:.1%}")
    
#     # Charts row
#     col_chart1, col_chart2 = st.columns(2)
    
#     with col_chart1:
#         st.subheader("Sound & Cry Probability")
#         if len(st.session_state.history['sound_levels']) > 0:
#             # Create DataFrame with proper alignment
#             sound_list = list(st.session_state.history['sound_levels'])
#             cry_list = [p * 100 for p in list(st.session_state.history['cry_probs'])]
            
#             # Ensure same length
#             min_len = min(len(sound_list), len(cry_list))
#             chart_data = pd.DataFrame({
#                 'Sound': sound_list[-min_len:],
#                 'Cry Prob': cry_list[-min_len:]
#             })
#             st.line_chart(chart_data, use_container_width=True)
#         else:
#             st.info("Waiting for audio data...")
    
#     with col_chart2:
#         st.subheader("Movement & Anomaly")
#         if len(st.session_state.history['movement_levels']) > 0:
#             # Create DataFrame with proper alignment
#             move_list = list(st.session_state.history['movement_levels'])
#             anomaly_list = list(st.session_state.history['anomaly_scores'])
            
#             # Ensure same length
#             min_len = min(len(move_list), len(anomaly_list))
#             chart_data2 = pd.DataFrame({
#                 'Movement': move_list[-min_len:],
#                 'Anomaly': anomaly_list[-min_len:]
#             })
#             st.line_chart(chart_data2, use_container_width=True)
#         else:
#             st.info("Waiting for movement data...")
    
#     # Explainability Panel
#     st.subheader("🔍 Decision Analysis")
    
#     if st.session_state.get('current_explanation'):
#         with st.container():
#             st.markdown(f"""
#             <div class="explanation-box">
#                 <h4>Why this decision?</h4>
#                 {st.session_state.current_explanation}
#             </div>
#             """, unsafe_allow_html=True)
    
#     # Action Recommendation
#     if st.session_state.get('current_action'):
#         st.info(f"**Recommended Action:** {st.session_state.current_action}")
    
#     # Detailed metrics expander
#     with st.expander("View Detailed Metrics"):
#         col_detail1, col_detail2 = st.columns(2)
        
#         with col_detail1:
#             st.write("**Current Values:**")
#             st.write(f"- Temperature: {temp_sim}°C")
#             st.write(f"- Hour: {datetime.now().hour}:00")
#             st.write(f"- Sound Level: {st.session_state.get('current_sound', 0):.1f}/100")
#             st.write(f"- Movement: {st.session_state.get('current_movement', 0):.1f}")
        
#         with col_detail2:
#             st.write("**Model Outputs:**")
#             st.write(f"- Cry Probability: {st.session_state.get('current_cry_prob', 0):.3f}")
#             st.write(f"- Anomaly Score: {st.session_state.get('current_anomaly', 0):.3f}")
#             st.write(f"- Is Cry: {st.session_state.get('current_is_cry', False)}")
#             st.write(f"- Is Anomaly: {st.session_state.get('current_is_anomaly', False)}")

# with tab2:
#     st.subheader("Historical Analytics")
    
#     if len(st.session_state.history['timestamps']) > 0:
#         # Create historical data
#         timestamps = list(st.session_state.history['timestamps'])
#         risk_scores = list(st.session_state.history['risk_scores'])
#         cry_probs = list(st.session_state.history['cry_probs'])
#         sound_levels = list(st.session_state.history['sound_levels'])
#         movements = list(st.session_state.history['movement_levels'])
        
#         # Ensure all lists have same length
#         min_len = min(len(timestamps), len(risk_scores), len(cry_probs))
        
#         hist_df = pd.DataFrame({
#             'Time': timestamps[-min_len:],
#             'Risk Score': risk_scores[-min_len:],
#             'Cry Probability': cry_probs[-min_len:],
#             'Sound Level': sound_levels[-min_len:],
#             'Movement': movements[-min_len:]
#         })
        
#         fig = px.line(hist_df, x='Time', y=['Risk Score', 'Cry Probability'],
#                       title="Risk Score Trend")
#         st.plotly_chart(fig, use_container_width=True)
        
#         # Summary statistics
#         col_stat1, col_stat2, col_stat3 = st.columns(3)
#         with col_stat1:
#             cry_count = sum(list(st.session_state.history['is_cry'])[-50:])  # Last 50
#             st.metric("Cry Events (last 50)", cry_count)
#         with col_stat2:
#             anomaly_count = sum(list(st.session_state.history['is_anomaly'])[-50:])  # Last 50
#             st.metric("Anomaly Events (last 50)", anomaly_count)
#         with col_stat3:
#             avg_risk = np.mean(list(st.session_state.history['risk_scores'])[-50:]) if st.session_state.history['risk_scores'] else 0
#             st.metric("Avg Risk Score (last 50)", f"{avg_risk:.2%}")
#     else:
#         st.info("📊 No historical data available yet. Start monitoring to collect data.")

# with tab3:
#     st.subheader("System Information")
    
#     st.markdown("""
#     ### Model Architecture
#     - **Cry Detection:** Supervised learning model trained on baby cry samples
#     - **Anomaly Detection:** Unsupervised isolation forest for behavioral patterns
#     - **Risk Scoring:** Hybrid weighted model combining multiple sensors
    
#     ### Decision Logic
#     The system combines multiple data streams to assess baby safety:
#     1. **Audio Analysis:** Real-time cry detection with confidence scoring
#     2. **Motion Detection:** Movement pattern analysis
#     3. **Temporal Context:** Time-of-day behavioral baselines
#     4. **Environmental:** Temperature monitoring
    
#     ### Features
#     - Real-time inference (2-second intervals)
#     - Sensor fusion (audio + video + temperature + time)
#     - Explainable AI with decision reasoning
#     - Risk-based alerting system
#     - Historical data logging
#     """)

# # Auto-refresh while monitoring
# if st.session_state.monitoring_active:
#     # Only rerun if enough time has passed
#     current_time = time.time()
#     if current_time - st.session_state.last_update > 2:
#         st.session_state.last_update = current_time
#         time.sleep(0.1)  # Small delay to prevent excessive reruns
#         st.rerun()

# # ---------------------------------
# # CLEANUP on exit
# # ---------------------------------
# def cleanup():
#     if st.session_state.monitoring_active:
#         st.session_state.stop_event.set()

# atexit.register(cleanup)
