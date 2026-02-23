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
















# import streamlit as st
# import cv2
# import sounddevice as sd
# import numpy as np
# import pandas as pd
# import joblib
# import time
# import mysql.connector
# from cry_infer import audio_to_features
# import matplotlib.pyplot as plt
# from collections import deque
# import requests

# # ============================================
# # SECURITY: Input Validation Function
# # ============================================
# def validate_and_sanitize(value, value_type, min_val=None, max_val=None, default=None):
#     """
#     SECURITY: Validate and sanitize all input values
#     - value: the input to validate
#     - value_type: 'int', 'float', or 'str'
#     - min_val: minimum allowed value
#     - max_val: maximum allowed value
#     - default: return this if validation fails
#     """
#     try:
#         # Convert to correct type
#         if value_type == 'int':
#             val = int(float(value))  # Handle strings that look like numbers
#         elif value_type == 'float':
#             val = float(value)
#         else:  # string
#             val = str(value).strip()
#             # Remove any non-printable characters
#             val = ''.join(char for char in val if char.isprintable())
#             # Limit length to prevent buffer overflow
#             return val[:255]
        
#         # Check range for numbers
#         if min_val is not None and val < min_val:
#             print(f"⚠️ Security: Value {val} below minimum {min_val}, using minimum")
#             return min_val
#         if max_val is not None and val > max_val:
#             print(f"⚠️ Security: Value {val} above maximum {max_val}, using maximum")
#             return max_val
            
#         return val
        
#     except (ValueError, TypeError) as e:
#         print(f"⚠️ Security: Validation failed for {value}: {e}")
#         # Return default or safe value based on type
#         if default is not None:
#             return default
#         return 0 if value_type in ['int', 'float'] else ''

# # ============================================
# # SUPERVISED LEARNING: Cry Detection Classifier
# # ============================================
# class CryDetector:
#     """SUPERVISED LEARNING: Binary classification (cry vs not-cry)"""
    
#     def __init__(self):
#         try:
#             self.model = joblib.load("cry_model.pkl")
#             print("✅ SUPERVISED MODEL LOADED: Random Forest Classifier")
#             print("   Trained on 77 cry + 60 not-cry samples")
#         except Exception as e:
#             print(f"❌ Supervised model not found: {e}")
#             self.model = None
    
#     def detect_cry(self, audio, fs):
#         """SUPERVISED: Predict if audio contains crying"""
#         try:
#             if audio is None or len(audio) == 0 or np.all(audio == 0):
#                 return 0.0, False, 0.0
            
#             # Extract features
#             feats = audio_to_features(audio, fs)
            
#             if len(feats.shape) == 1:
#                 feats = feats.reshape(1, -1)
            
#             if hasattr(self.model, 'predict_proba'):
#                 prob = self.model.predict_proba(feats)[0]
#                 if len(prob) > 1:
#                     cry_prob = float(prob[1])  # Probability of positive class
#                 else:
#                     cry_prob = float(prob[0])
                
#                 # Get confidence (margin between top two classes)
#                 if len(prob) > 1:
#                     confidence = abs(prob[1] - prob[0])
#                 else:
#                     confidence = prob[0]
#             else:
#                 cry_prob = float(self.model.decision_function(feats)[0])
#                 cry_prob = 1.0 / (1.0 + np.exp(-cry_prob))
#                 confidence = abs(cry_prob - 0.5) * 2  # Scale to 0-1
            
#             is_cry = cry_prob >= 0.80
            
#             return cry_prob, is_cry, confidence
            
#         except Exception as e:
#             print(f"Cry detection error: {e}")
#             return 0.0, False, 0.0

# # ============================================
# # UNSUPERVISED LEARNING: Anomaly Detection (FIXED VERSION)
# # ============================================
# class AnomalyDetector:
#     """UNSUPERVISED LEARNING: IsolationForest with temporal context"""
    
#     def __init__(self, window_size=30):  # Store last 30 readings (3 seconds at 0.1s intervals)
#         try:
#             self.model = joblib.load("baby_model.pkl")
#             print("✅ UNSUPERVISED ANOMALY MODEL LOADED: IsolationForest")
#         except Exception as e:
#             print(f"⚠️ Anomaly model not found, using statistical method: {e}")
#             self.model = None
        
#         self.window_size = window_size
#         self.reading_history = deque(maxlen=window_size)
#         self.baseline_stats = self.load_baseline_stats()
    
#     def load_baseline_stats(self):
#         """Load or create baseline statistics from database"""
#         baseline = {}
#         try:
#             # SECURITY: Use parameterized query (even though no user input here)
#             conn = mysql.connector.connect(
#                 host="localhost",
#                 user="root",
#                 password="root1234",
#                 database="baby_monitor"
#             )
            
#             # SECURITY: Using parameterized query with pandas read_sql
#             query = """
#             SELECT hour, sound_level, movement_level, temperature 
#             FROM monitor_activity 
#             WHERE (is_anomaly = 0 OR is_anomaly IS NULL)
#             """
#             # SECURITY: Parameters tuple (empty in this case)
#             df = pd.read_sql(query, conn)
#             conn.close()
            
#             if len(df) > 100:  # Enough data
#                 for hour in range(24):
#                     hour_data = df[df['hour'] == hour]
#                     if len(hour_data) > 5:
#                         baseline[hour] = {
#                             'sound_mean': hour_data['sound_level'].mean(),
#                             'sound_std': hour_data['sound_level'].std(),
#                             'sound_p95': hour_data['sound_level'].quantile(0.95),
#                             'sound_p5': hour_data['sound_level'].quantile(0.05),
#                             'movement_mean': hour_data['movement_level'].mean(),
#                             'movement_std': hour_data['movement_level'].std(),
#                             'movement_p95': hour_data['movement_level'].quantile(0.95),
#                         }
#                     else:
#                         # Fallback for hours with no data
#                         baseline[hour] = self.get_default_stats(hour)
#             else:
#                 # Not enough data, use defaults
#                 for hour in range(24):
#                     baseline[hour] = self.get_default_stats(hour)
                    
#         except Exception as e:
#             print(f"Could not load baseline from DB: {e}")
#             # Use reasonable defaults
#             for hour in range(24):
#                 baseline[hour] = self.get_default_stats(hour)
        
#         return baseline
    
#     def get_default_stats(self, hour):
#         """Provide reasonable default stats based on time of day"""
#         # Night time (11 PM - 6 AM) should be quieter
#         if 23 <= hour or hour <= 6:
#             return {
#                 'sound_mean': 10.0,
#                 'sound_std': 5.0,
#                 'sound_p95': 20.0,
#                 'sound_p5': 2.0,
#                 'movement_mean': 5.0,
#                 'movement_std': 3.0,
#                 'movement_p95': 12.0,
#             }
#         # Day time - more activity
#         else:
#             return {
#                 'sound_mean': 25.0,
#                 'sound_std': 10.0,
#                 'sound_p95': 45.0,
#                 'sound_p5': 8.0,
#                 'movement_mean': 15.0,
#                 'movement_std': 8.0,
#                 'movement_p95': 30.0,
#             }
    
#     def add_reading(self, hour, temp, sound, movement):
#         """Add a reading to history"""
#         self.reading_history.append({
#             'timestamp': time.time(),
#             'hour': hour,
#             'temp': temp,
#             'sound': sound,
#             'movement': movement
#         })
    
#     def extract_features(self):
#         """Extract features matching training data"""
#         if len(self.reading_history) < 1:
#             return None
        
#         df = pd.DataFrame(list(self.reading_history))
        
#         # Use ONLY the features the model was trained with
#         features = {
#             'hour': df['hour'].iloc[-1],
#             'temp': df['temp'].iloc[-1],
#             'sound': df['sound'].iloc[-1],
#             'movement': df['movement'].iloc[-1]
#         }
        
#         return pd.DataFrame([features])

    
#     def detect_anomaly(self, hour, temp, sound, movement):
#         """Enhanced anomaly detection with explanations (FIX #3 & #4)"""
        
#         # Add to history
#         self.add_reading(hour, temp, sound, movement)
        
#         # Initialize result with explanations
#         result = {
#             'is_anomaly': False,
#             'score': 0.0,
#             'explanations': [],
#             'confidence': 0.0,
#             'severity': 'low'  # low, medium, high
#         }
        
#         explanations = []
#         anomaly_votes = 0
#         total_checks = 0
        
#         # ========== METHOD 1: ML-based Anomaly Detection ==========
#         if self.model and len(self.reading_history) >= 5:
#             features = self.extract_features()
#             if features is not None:
#                 total_checks += 1
#                 # ML prediction
#                 ml_pred = self.model.predict(features)[0]
#                 if hasattr(self.model, 'decision_function'):
#                     ml_score = self.model.decision_function(features)[0]
#                     result['score'] = float(ml_score)
                
#                 if ml_pred == -1:  # -1 indicates anomaly in IsolationForest
#                     anomaly_votes += 1
#                     explanations.append("🤖 ML model detected unusual pattern")
        
#         # ========== METHOD 2: Statistical Anomaly Detection ==========
#         hour_int = int(hour)
#         if hour_int in self.baseline_stats:
#             stats = self.baseline_stats[hour_int]
            
#             # Check sound level against historical percentiles
#             total_checks += 1
#             if sound > stats['sound_p95']:
#                 anomaly_votes += 1
#                 explanations.append(
#                     f"🔊 Sound level ({sound:.1f}) is above 95th percentile ({stats['sound_p95']:.1f}) for {hour_int}:00"
#                 )
            
#             total_checks += 1
#             if sound < stats['sound_p5']:
#                 anomaly_votes += 1
#                 explanations.append(
#                     f"🔇 Sound level ({sound:.1f}) is below 5th percentile ({stats['sound_p5']:.1f}) for {hour_int}:00"
#                 )
            
#             # Z-score check (values > 2 standard deviations from mean)
#             total_checks += 1
#             sound_z = abs((sound - stats['sound_mean']) / (stats['sound_std'] + 0.001))
#             if sound_z > 2:
#                 anomaly_votes += 1
#                 explanations.append(
#                     f"📊 Sound level is {sound_z:.1f}σ from normal (unusual variation)"
#                 )
        
#         # ========== METHOD 3: Rule-based Anomaly Detection ==========
#         # Time-based rules
#         total_checks += 1
#         if 23 <= hour or hour <= 6:  # Night hours (11 PM - 6 AM)
#             if sound > 30:  # Loud at night
#                 anomaly_votes += 1
#                 explanations.append("🌙 High noise level during sleeping hours")
        
#         total_checks += 1
#         if 7 <= hour <= 20:  # Day hours
#             if sound < 5 and movement < 2:  # Too quiet during day
#                 anomaly_votes += 1
#                 explanations.append("☀️ Unusually quiet during daytime - check if baby is OK")
        
#         # Movement vs Sound anomalies
#         total_checks += 1
#         if movement > 40 and sound < 5:  # High movement but completely silent
#             anomaly_votes += 1
#             explanations.append("🎥 High movement with no sound - possible camera issue or baby in distress")
        
#         total_checks += 1
#         if sound > 70:  # Extremely loud
#             anomaly_votes += 1
#             explanations.append("🚨 Extreme noise level detected")
#             result['severity'] = 'high'
        
#         # Trend anomalies
#         if len(self.reading_history) >= 10:
#             df = pd.DataFrame(list(self.reading_history))
            
#             total_checks += 1
#             # Sudden spike in sound
#             if len(df) > 5 and df['sound'].iloc[-1] > df['sound'].iloc[-6:-1].mean() * 3:
#                 anomaly_votes += 1
#                 explanations.append("📈 Sudden spike in sound level")
            
#             total_checks += 1
#             # Prolonged high movement
#             if df['movement'].tail(10).mean() > 30:
#                 anomaly_votes += 1
#                 explanations.append("🏃 Prolonged high movement detected")
        
#         # ========== DETERMINE FINAL RESULT ==========
#         if total_checks > 0:
#             anomaly_ratio = anomaly_votes / total_checks
            
#             # Determine if it's an anomaly
#             if anomaly_ratio >= 0.3:  # At least 30% of checks flagged
#                 result['is_anomaly'] = True
#                 result['confidence'] = min(anomaly_ratio * 1.5, 0.95)  # Scale but cap at 0.95
                
#                 # Set severity
#                 if anomaly_ratio >= 0.6:
#                     result['severity'] = 'high'
#                 elif anomaly_ratio >= 0.4:
#                     result['severity'] = 'medium'
                
#                 # Deduplicate explanations
#                 result['explanations'] = list(dict.fromkeys(explanations))
#             else:
#                 result['is_anomaly'] = False
#                 result['confidence'] = 1 - anomaly_ratio
#                 result['explanations'] = ["✅ All patterns normal"]
        
#         return result

# # ============================================
# # PAGE CONFIG
# # ============================================
# st.set_page_config(
#     page_title="AI Baby Monitor", 
#     page_icon="👶",
#     layout="wide"
# )

# st.title("👶 Smart Baby Monitor")
# st.markdown("*Using AI to keep your baby safe*")

# # ---------------------------------
# # AI TECHNIQUES OVERVIEW
# # ---------------------------------
# with st.sidebar:
#     st.header("🤖 AI Techniques Used")
    
#     st.markdown("""
#     ### ✅ SUPERVISED LEARNING
#     - **Algorithm:** Random Forest Classifier
#     - **Purpose:** Cry detection
#     - **Training:** 137 labeled samples
#     - **Output:** Cry probability with confidence
    
#     ### ✅ UNSUPERVISED LEARNING (Anomaly Detection)
#     - **Algorithm:** IsolationForest + Statistical Methods
#     - **Purpose:** Detect unusual behavior patterns
#     - **Features:** Time (cyclical), sound, movement, temperature
#     - **Output:** Anomaly score + explanations
#     """)

# # ---------------------------------
# # MYSQL CONNECTION
# # ---------------------------------
# def get_db_connection():
#     return mysql.connector.connect(
#         host="localhost",
#         user="root",
#         password="root1234",
#         database="baby_monitor"
#     )

# def log_to_mysql(hour, temp, sound, movement, cry_prob, anomaly_result, is_cry):
#     """Log monitoring data to database with SQL injection prevention"""
#     try:
#         # SECURITY: Validate ALL inputs before database insertion
#         hour = validate_and_sanitize(hour, 'int', 0, 23, 0)
#         temp = validate_and_sanitize(temp, 'float', 18, 35, 22)
#         sound = validate_and_sanitize(sound, 'float', 0, 100, 0)
#         movement = validate_and_sanitize(movement, 'float', 0, 255, 0)
#         cry_prob = validate_and_sanitize(cry_prob, 'float', 0, 1, 0)
        
#         # SECURITY: If any validation failed, don't proceed
#         if None in [hour, temp, sound, movement, cry_prob]:
#             print("❌ Security: Invalid input values detected, skipping database insert")
#             return
        
#         conn = get_db_connection()
#         cursor = conn.cursor()
        
#         # Check if new columns exist
#         cursor.execute("SHOW COLUMNS FROM monitor_activity")
#         columns = [col[0] for col in cursor.fetchall()]
        
#         # SECURITY: Using parameterized query with %s placeholders
#         # This prevents SQL injection by separating query structure from data
#         if 'anomaly_explanations' in columns and 'severity' in columns:
#             query = """
#             INSERT INTO monitor_activity
#             (hour, temperature, sound_level, movement_level,
#              cry_probability, anomaly_score, is_cry, is_anomaly,
#              anomaly_explanations, severity)
#             VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
#             """
#             # SECURITY: Sanitize string inputs as well
#             explanations_str = ', '.join(anomaly_result['explanations'])[:255]
#             explanations_str = validate_and_sanitize(explanations_str, 'str')
#             severity = validate_and_sanitize(anomaly_result['severity'], 'str')
            
#             values = (
#                 hour, temp, sound, movement,
#                 cry_prob, float(anomaly_result['score']), 
#                 int(is_cry), int(anomaly_result['is_anomaly']),
#                 explanations_str, severity
#             )
#         else:
#             # Fallback to basic columns
#             query = """
#             INSERT INTO monitor_activity
#             (hour, temperature, sound_level, movement_level,
#              cry_probability, anomaly_score, is_cry, is_anomaly)
#             VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
#             """
#             values = (
#                 hour, temp, sound, movement, 
#                 cry_prob, float(anomaly_result['score']), 
#                 int(is_cry), int(anomaly_result['is_anomaly'])
#             )
        
#         # SECURITY: Execute with separate values tuple - SQL injection impossible here
#         cursor.execute(query, values)
#         conn.commit()
#         cursor.close()
#         conn.close()
        
#     except mysql.connector.Error as e:
#         st.error(f"MySQL Error: {e}")
#         print(f"❌ Database error: {e}")
#     except Exception as e:
#         st.error(f"Error logging to database: {e}")
#         print(f"❌ Unexpected error: {e}")

# # ---------------------------------
# # INITIALIZE MODELS
# # ---------------------------------
# st.sidebar.markdown("---")
# st.sidebar.subheader("📊 Model Status")

# # Initialize detectors
# cry_detector = CryDetector()
# anomaly_detector = AnomalyDetector(window_size=30)

# # Show status
# if cry_detector.model:
#     st.sidebar.success("✅ Supervised (Cry Detection): Ready")
# else:
#     st.sidebar.error("❌ Supervised Model Missing")

# if anomaly_detector.model:
#     st.sidebar.success("✅ Unsupervised (Anomaly): ML Ready")
# else:
#     st.sidebar.info("ℹ️ Unsupervised: Using Statistical Methods")

# # ---------------------------------
# # UI CONTROLS
# # ---------------------------------
# col1, col2, col3 = st.columns(3)

# with col1:
#     temp_sim = st.slider("🌡️ Room Temperature (°C)", 18, 35, 22)
#     # SECURITY: Validate slider input (though sliders are safe, double-check)
#     temp_sim = validate_and_sanitize(temp_sim, 'float', 18, 35, 22)

# with col2:
#     sensitivity = st.select_slider(
#         "🎚️ Anomaly Sensitivity",
#         options=["Low", "Medium", "High"],
#         value="Medium"
#     )
#     # SECURITY: Validate sensitivity selection
#     sensitivity = validate_and_sanitize(sensitivity, 'str')
#     # Adjust threshold based on sensitivity
#     anomaly_threshold = {"Low": 0.5, "Medium": 0.3, "High": 0.15}[sensitivity]

# with col3:
#     status_box = st.empty()

# # Audio test section
# with st.expander("🎤 Test Microphone"):
#     if st.button("Test Microphone"):
#         with st.spinner("Recording for 2 seconds..."):
#             fs = 44100
#             recording = sd.rec(int(2.0 * fs), samplerate=fs, channels=1, dtype='float32', device=1)
#             sd.wait()
#             audio = recording.flatten()
#             rms = np.sqrt(np.mean(audio**2))
#             test_level = float(min(rms * 2000, 100.0))
#             # SECURITY: Validate test level
#             test_level = validate_and_sanitize(test_level, 'float', 0, 100, 0)
            
#             col_mic1, col_mic2 = st.columns(2)
#             with col_mic1:
#                 st.write(f"📊 Audio level: {test_level:.2f}/100")
#             with col_mic2:
#                 if test_level > 10:
#                     st.success("✅ Microphone working!")
#                 elif test_level > 5:
#                     st.warning("⚠️ Low audio - try speaking louder")
#                 else:
#                     st.error("❌ No audio detected")

# # Start/Stop buttons
# col_start, col_stop, col_clear = st.columns(3)
# with col_start:
#     start_btn = st.button("▶ Start Monitoring", type="primary", use_container_width=True)
# with col_stop:
#     stop_btn = st.button("⏹ Stop", use_container_width=True)
# with col_clear:
#     if st.button("🗑️ Clear History", use_container_width=True):
#         st.session_state.sound_history = deque(maxlen=100)
#         st.session_state.move_history = deque(maxlen=100)
#         st.session_state.cry_history = deque(maxlen=100)
#         st.rerun()

# # Initialize session state for history
# if 'sound_history' not in st.session_state:
#     st.session_state.sound_history = deque(maxlen=100)
#     st.session_state.move_history = deque(maxlen=100)
#     st.session_state.cry_history = deque(maxlen=100)
#     st.session_state.timestamps = deque(maxlen=100)
#     st.session_state.run = False

# # Update run state
# if start_btn:
#     st.session_state.run = True
# if stop_btn:
#     st.session_state.run = False

# # Create layout
# chart_col1, chart_col2 = st.columns(2)

# with chart_col1:
#     st.subheader("📊 Sound Level")
#     sound_chart = st.line_chart(np.zeros((1, 1)), use_container_width=True)

# with chart_col2:
#     st.subheader("🏃 Movement Level")
#     move_chart = st.line_chart(np.zeros((1, 1)), use_container_width=True)

# # Metrics row
# metric_cols = st.columns(4)
# with metric_cols[0]:
#     sound_metric = st.empty()
# with metric_cols[1]:
#     cry_metric = st.empty()
# with metric_cols[2]:
#     anomaly_metric = st.empty()
# with metric_cols[3]:
#     confidence_metric = st.empty()

# # Anomaly explanations area
# st.subheader("🔍 Anomaly Analysis")
# explanation_box = st.container()

# # ---------------------------------
# # CAMERA INITIALIZATION
# # ---------------------------------
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# if not cap.isOpened():
#     st.error("❌ Webcam not detected. Please check your camera connection.")
#     st.stop()

# # Warm up camera
# time.sleep(1)
# ret, prev_frame = cap.read()
# if not ret or prev_frame is None:
#     st.error("❌ Failed to read from webcam.")
#     st.stop()

# prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# # ---------------------------------
# # AUDIO FUNCTION
# # ---------------------------------
# def get_mic_audio_and_score():
#     """Record audio and calculate sound level"""
#     try:
#         fs = 44100
#         duration = 2.0
#         chunk = int(duration * fs)
        
#         recording = sd.rec(chunk, samplerate=fs, channels=1, dtype='float32', device=1)
#         sd.wait()
        
#         audio = recording.flatten()
#         rms = np.sqrt(np.mean(audio**2))
#         loudness = float(min(rms * 2000, 100.0))
#         # SECURITY: Validate loudness
#         loudness = validate_and_sanitize(loudness, 'float', 0, 100, 0)
        
#         return audio, fs, loudness
        
#     except Exception as e:
#         print(f"Microphone error: {e}")
#         return np.zeros(int(2.0 * 44100)), 44100, 0.0

# # ---------------------------------
# # MAIN LOOP
# # ---------------------------------
# if st.session_state.run:
#     st.info("🟢 Monitoring active - press 'Stop' to end")
    
#     # Create placeholder for real-time updates
#     explanation_placeholder = st.empty()
    
#     while st.session_state.run:
#         loop_start = time.time()
        
#         # -------- VIDEO PROCESSING --------
#         ret, frame = cap.read()
#         if not ret or frame is None:
#             st.warning("⚠️ Webcam frame not received. Reconnecting...")
#             cap.release()
#             time.sleep(1)
#             cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
#             continue
        
#         # Convert to grayscale and calculate movement
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         frame_diff = cv2.absdiff(prev_gray, gray)
#         move_score = float(np.mean(frame_diff))
#         # SECURITY: Validate movement score
#         move_score = validate_and_sanitize(move_score, 'float', 0, 255, 0)
#         prev_gray = gray
        
#         # -------- AUDIO PROCESSING --------
#         audio, audio_fs, sound_score = get_mic_audio_and_score()
        
#         # -------- SUPERVISED: CRY DETECTION --------
#         cry_prob, is_cry, cry_confidence = cry_detector.detect_cry(audio, audio_fs)
#         # SECURITY: Validate cry probability
#         cry_prob = validate_and_sanitize(cry_prob, 'float', 0, 1, 0)
        
#         # -------- UNSUPERVISED: ANOMALY DETECTION --------
#         hour_now = time.localtime().tm_hour
#         # SECURITY: Validate hour
#         hour_now = validate_and_sanitize(hour_now, 'int', 0, 23, 0)
        
#         anomaly_result = anomaly_detector.detect_anomaly(
#             hour=hour_now,
#             temp=temp_sim,
#             sound=sound_score,
#             movement=move_score
#         )
        
#         # Override is_anomaly based on sensitivity
#         if anomaly_result['confidence'] >= anomaly_threshold:
#             anomaly_result['is_anomaly'] = True

#         if is_cry or anomaly_result['is_anomaly']:
#             try:
#                 # This sends the data to the "AI Input" node in Node-RED
#                 payload = {
#                     "is_cry": bool(is_cry),
#                     "is_anomaly": bool(anomaly_result['is_anomaly']),
#                     "severity": anomaly_result['severity']
#                 }
#                 requests.post("http://localhost:1880/baby-alert", json=payload, timeout=0.1)
#             except Exception as e:
#                 pass # Prevents app from crashing if Node-RED is closed
        
#         # -------- UPDATE HISTORY --------
#         current_time = time.time()
#         st.session_state.timestamps.append(current_time)
#         st.session_state.sound_history.append(sound_score)
#         st.session_state.move_history.append(move_score)
#         st.session_state.cry_history.append(cry_prob)
        
#         # -------- UPDATE METRICS --------
#         sound_metric.metric(
#             "🎤 Sound Level", 
#             f"{sound_score:.1f}/100",
#             delta=f"{sound_score - (st.session_state.sound_history[-2] if len(st.session_state.sound_history) > 1 else 0):.1f}"
#         )
        
#         cry_metric.metric(
#             "😢 Cry Probability", 
#             f"{cry_prob*100:.1f}%",
#             delta=f"{(cry_prob*100 - (st.session_state.cry_history[-2]*100 if len(st.session_state.cry_history) > 1 else 0)):.1f}%",
#             delta_color="inverse"
#         )
        
#         anomaly_metric.metric(
#             "🚨 Anomaly Score", 
#             f"{anomaly_result['score']:.2f}",
#             delta=None
#         )
        
#         confidence_metric.metric(
#             "📊 Confidence", 
#             f"{anomaly_result['confidence']*100:.0f}%",
#             delta=None
#         )
        
#         # -------- UPDATE CHARTS --------
#         if len(st.session_state.sound_history) > 1:
#             sound_df = pd.DataFrame({
#                 'Sound Level': list(st.session_state.sound_history)
#             })
#             sound_chart.line_chart(sound_df, use_container_width=True)
            
#             move_df = pd.DataFrame({
#                 'Movement': list(st.session_state.move_history)
#             })
#             move_chart.line_chart(move_df, use_container_width=True)
        
#         # -------- DISPLAY ANOMALY EXPLANATIONS --------
#         with explanation_box:
#             cols = st.columns([1, 3])
            
#             with cols[0]:
#                 if anomaly_result['is_anomaly']:
#                     if anomaly_result['severity'] == 'high':
#                         st.error(f"🚨 **HIGH RISK**")
#                     elif anomaly_result['severity'] == 'medium':
#                         st.warning(f"⚠️ **MEDIUM RISK**")
#                     else:
#                         st.info(f"🔔 **LOW RISK**")
#                 else:
#                     st.success(f"✅ **NORMAL**")
            
#             with cols[1]:
#                 for exp in anomaly_result['explanations']:
#                     if '✅' in exp:
#                         st.success(exp)
#                     elif '🚨' in exp or '🔊' in exp or '🌙' in exp:
#                         st.error(exp)
#                     elif '⚠️' in exp:
#                         st.warning(exp)
#                     else:
#                         st.info(exp)
        
#         # -------- OVERALL STATUS --------
#         if anomaly_result['is_anomaly'] and is_cry:
#             status_box.error(f"🚨 **CRITICAL: Baby crying with unusual patterns**")
#         elif is_cry:
#             status_box.warning(f"😢 **Baby is crying** ({(cry_prob*100):.1f}% confidence)")
#         elif anomaly_result['is_anomaly']:
#             if anomaly_result['severity'] == 'high':
#                 status_box.error(f"🚨 **Unusual behavior detected - Check on baby**")
#             else:
#                 status_box.warning(f"⚠️ **Unusual pattern detected**")
#         else:
#             status_box.success(f"✅ **All normal - Baby is okay**")
        
#         # -------- LOG TO DATABASE --------
#         log_to_mysql(
#             hour=hour_now,
#             temp=temp_sim,
#             sound=sound_score,
#             movement=move_score,
#             cry_prob=cry_prob,
#             anomaly_result=anomaly_result,
#             is_cry=is_cry
#         )
        
#         # Control loop speed (target ~10 FPS)
#         elapsed = time.time() - loop_start
#         if elapsed < 0.1:
#             time.sleep(0.1 - elapsed)

# # ---------------------------------
# # CLEANUP
# # ---------------------------------
# cap.release()
# if not st.session_state.run:
#     st.success("🛑 Monitoring stopped")
    
#     # Show summary if we have data
#     if len(st.session_state.sound_history) > 0:
#         st.subheader("📈 Session Summary")
#         col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
        
#         with col_sum1:
#             avg_sound = np.mean(list(st.session_state.sound_history))
#             st.metric("Avg Sound Level", f"{avg_sound:.1f}/100")
        
#         with col_sum2:
#             avg_move = np.mean(list(st.session_state.move_history))
#             st.metric("Avg Movement", f"{avg_move:.1f}")
        
#         with col_sum3:
#             cry_events = sum(1 for c in list(st.session_state.cry_history) if c > 0.8)
#             st.metric("Cry Events", cry_events)
        
#         with col_sum4:
#             anomaly_pct = sum(1 for a in anomaly_detector.reading_history) / max(len(st.session_state.sound_history), 1) * 100
#             st.metric("Anomaly %", f"{anomaly_pct:.1f}%")







# working app.py (just very cramped ui)


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
#     .stProgress > div > div > div > div {
#         background-color: #0066cc;
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
#         'timestamps': deque(maxlen=30),  # Show last 30 points (1 minute at 2s intervals)
#         'sound_levels': deque(maxlen=30),
#         'movement_levels': deque(maxlen=30),
#         'cry_probs': deque(maxlen=30),
#         'risk_scores': deque(maxlen=30),
#         'anomaly_scores': deque(maxlen=30),
#         'is_cry': deque(maxlen=30),
#         'is_anomaly': deque(maxlen=30)
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
    
# if 'chart_placeholders' not in st.session_state:
#     st.session_state.chart_placeholders = {}

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
# # CREATE CHART FUNCTION
# # ---------------------------------
# def create_sound_chart(sound_levels, cry_probs):
#     """Create a Plotly chart for sound and cry probability"""
#     if len(sound_levels) == 0:
#         return go.Figure()
    
#     # Create x-axis values (time steps)
#     x_values = list(range(len(sound_levels)))
    
#     fig = go.Figure()
    
#     # Add sound level trace
#     fig.add_trace(go.Scatter(
#         x=x_values,
#         y=list(sound_levels),
#         mode='lines+markers',
#         name='Sound Level',
#         line=dict(color='#1f77b4', width=2),
#         marker=dict(size=4)
#     ))
    
#     # Add cry probability trace (scaled to 0-100 for comparison)
#     cry_scaled = [p * 100 for p in cry_probs]
#     fig.add_trace(go.Scatter(
#         x=x_values,
#         y=cry_scaled,
#         mode='lines+markers',
#         name='Cry Probability %',
#         line=dict(color='#ff7f0e', width=2),
#         marker=dict(size=4)
#     ))
    
#     fig.update_layout(
#         title="Sound Level & Cry Probability",
#         xaxis_title="Time (last 30 samples)",
#         yaxis_title="Value",
#         hovermode='x unified',
#         legend=dict(
#             yanchor="top",
#             y=0.99,
#             xanchor="left",
#             x=0.01
#         ),
#         margin=dict(l=40, r=40, t=40, b=40)
#     )
    
#     return fig

# def create_movement_chart(movement_levels, anomaly_scores):
#     """Create a Plotly chart for movement and anomaly scores"""
#     if len(movement_levels) == 0:
#         return go.Figure()
    
#     # Create x-axis values (time steps)
#     x_values = list(range(len(movement_levels)))
    
#     fig = go.Figure()
    
#     # Add movement trace
#     fig.add_trace(go.Scatter(
#         x=x_values,
#         y=list(movement_levels),
#         mode='lines+markers',
#         name='Movement',
#         line=dict(color='#2ca02c', width=2),
#         marker=dict(size=4)
#     ))
    
#     # Add anomaly score trace
#     fig.add_trace(go.Scatter(
#         x=x_values,
#         y=list(anomaly_scores),
#         mode='lines+markers',
#         name='Anomaly Score',
#         line=dict(color='#d62728', width=2),
#         marker=dict(size=4)
#     ))
    
#     fig.update_layout(
#         title="Movement & Anomaly Score",
#         xaxis_title="Time (last 30 samples)",
#         yaxis_title="Value",
#         hovermode='x unified',
#         legend=dict(
#             yanchor="top",
#             y=0.99,
#             xanchor="left",
#             x=0.01
#         ),
#         margin=dict(l=40, r=40, t=40, b=40)
#     )
    
#     return fig

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
            
#             # Clear history
#             st.session_state.history = {
#                 'timestamps': deque(maxlen=30),
#                 'sound_levels': deque(maxlen=30),
#                 'movement_levels': deque(maxlen=30),
#                 'cry_probs': deque(maxlen=30),
#                 'risk_scores': deque(maxlen=30),
#                 'anomaly_scores': deque(maxlen=30),
#                 'is_cry': deque(maxlen=30),
#                 'is_anomaly': deque(maxlen=30)
#             }
            
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
#         st.metric("Cry Probability", f"{cry_val:.1f}%",
#                  delta=f"{'Crying' if st.session_state.get('current_is_cry', False) else 'Not crying'}")
    
#     with col_metrics4:
#         anomaly_val = st.session_state.get('current_anomaly', 0)
#         st.metric("Anomaly Score", f"{anomaly_val:.2f}",
#                  delta=f"{'Anomaly' if st.session_state.get('current_is_anomaly', False) else 'Normal'}")
    
#     # Confidence indicator
#     if 'current_cry_prob' in st.session_state:
#         confidence = max(st.session_state.current_cry_prob, 
#                         1 - st.session_state.current_cry_prob)
#         st.progress(confidence, text=f"Model Confidence: {confidence:.1%}")
    
#     # Charts row using Plotly for better real-time updates
#     col_chart1, col_chart2 = st.columns(2)
    
#     with col_chart1:
#         # Sound chart
#         sound_levels = list(st.session_state.history['sound_levels'])
#         cry_probs = list(st.session_state.history['cry_probs'])
        
#         if len(sound_levels) > 1:  # Need at least 2 points to show movement
#             fig1 = create_sound_chart(sound_levels, cry_probs)
#             st.plotly_chart(fig1, use_container_width=True, key=f"sound_chart_{len(sound_levels)}")
#         else:
#             st.info("📊 Collecting sound data...")
#             # Show placeholder
#             placeholder_df = pd.DataFrame({'Waiting': [0]})
#             st.line_chart(placeholder_df)
    
#     with col_chart2:
#         # Movement chart
#         movement_levels = list(st.session_state.history['movement_levels'])
#         anomaly_scores = list(st.session_state.history['anomaly_scores'])
        
#         if len(movement_levels) > 1:  # Need at least 2 points to show movement
#             fig2 = create_movement_chart(movement_levels, anomaly_scores)
#             st.plotly_chart(fig2, use_container_width=True, key=f"move_chart_{len(movement_levels)}")
#         else:
#             st.info("📊 Collecting movement data...")
#             # Show placeholder
#             placeholder_df = pd.DataFrame({'Waiting': [0]})
#             st.line_chart(placeholder_df)
    
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
            
#             # Show last update time
#             if st.session_state.history['timestamps']:
#                 last_time = list(st.session_state.history['timestamps'])[-1]
#                 st.write(f"- Last Update: {last_time.strftime('%H:%M:%S')}")

# with tab2:
#     st.subheader("Historical Analytics")
    
#     if len(st.session_state.history['timestamps']) > 0:
#         # Create historical data
#         timestamps = list(st.session_state.history['timestamps'])
#         risk_scores = list(st.session_state.history['risk_scores'])
#         cry_probs = list(st.session_state.history['cry_probs'])
#         sound_levels = list(st.session_state.history['sound_levels'])
#         movements = list(st.session_state.history['movement_levels'])
        
#         # Create DataFrame for plotting
#         hist_df = pd.DataFrame({
#             'Time': timestamps,
#             'Risk Score': risk_scores,
#             'Cry Probability': cry_probs,
#             'Sound Level': sound_levels,
#             'Movement': movements
#         })
        
#         # Risk score trend
#         fig = px.line(hist_df, x='Time', y=['Risk Score', 'Cry Probability'],
#                       title="Risk Score Trend Over Time")
#         st.plotly_chart(fig, use_container_width=True)
        
#         # Summary statistics
#         col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
#         with col_stat1:
#             cry_count = sum(list(st.session_state.history['is_cry']))
#             st.metric("Total Cry Events", cry_count)
#         with col_stat2:
#             anomaly_count = sum(list(st.session_state.history['is_anomaly']))
#             st.metric("Total Anomaly Events", anomaly_count)
#         with col_stat3:
#             avg_risk = np.mean(risk_scores) if risk_scores else 0
#             st.metric("Avg Risk Score", f"{avg_risk:.2%}")
#         with col_stat4:
#             max_risk = max(risk_scores) if risk_scores else 0
#             st.metric("Peak Risk", f"{max_risk:.2%}")
        
#         # Data table
#         with st.expander("View Raw Data"):
#             st.dataframe(hist_df.tail(10))
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

# # ---------------------------------
# # AUTO REFRESH WHILE MONITORING
# # ---------------------------------
# if st.session_state.monitoring_active:
#     time.sleep(2)  # match monitoring interval
#     st.rerun()

# # ---------------------------------
# # CLEANUP on exit
# # ---------------------------------
# def cleanup():
#     if st.session_state.monitoring_active:
#         st.session_state.stop_event.set()

# atexit.register(cleanup)































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

















import streamlit as st
import cv2
import sounddevice as sd
import numpy as np
import pandas as pd
import joblib
import time
import mysql.connector
from cry_infer import audio_to_features
import matplotlib.pyplot as plt
from collections import deque
import requests

# ============================================
# SECURITY: Input Validation Function
# ============================================
def validate_and_sanitize(value, value_type, min_val=None, max_val=None, default=None):
    """
    SECURITY: Validate and sanitize all input values
    - value: the input to validate
    - value_type: 'int', 'float', or 'str'
    - min_val: minimum allowed value
    - max_val: maximum allowed value
    - default: return this if validation fails
    """
    try:
        # Convert to correct type
        if value_type == 'int':
            val = int(float(value))  # Handle strings that look like numbers
        elif value_type == 'float':
            val = float(value)
        else:  # string
            val = str(value).strip()
            # Remove any non-printable characters
            val = ''.join(char for char in val if char.isprintable())
            # Limit length to prevent buffer overflow
            return val[:255]
        
        # Check range for numbers
        if min_val is not None and val < min_val:
            print(f"⚠️ Security: Value {val} below minimum {min_val}, using minimum")
            return min_val
        if max_val is not None and val > max_val:
            print(f"⚠️ Security: Value {val} above maximum {max_val}, using maximum")
            return max_val
            
        return val
        
    except (ValueError, TypeError) as e:
        print(f"⚠️ Security: Validation failed for {value}: {e}")
        # Return default or safe value based on type
        if default is not None:
            return default
        return 0 if value_type in ['int', 'float'] else ''

# ============================================
# SUPERVISED LEARNING: Cry Detection Classifier
# ============================================
class CryDetector:
    """SUPERVISED LEARNING: Binary classification (cry vs not-cry)"""
    
    def __init__(self):  # FIXED: Changed from _init_ to __init__
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
                return 0.0, False, 0.0
            
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
                
                # Get confidence (margin between top two classes)
                if len(prob) > 1:
                    confidence = abs(prob[1] - prob[0])
                else:
                    confidence = prob[0]
            else:
                cry_prob = float(self.model.decision_function(feats)[0])
                cry_prob = 1.0 / (1.0 + np.exp(-cry_prob))
                confidence = abs(cry_prob - 0.5) * 2  # Scale to 0-1
            
            is_cry = cry_prob >= 0.80
            
            return cry_prob, is_cry, confidence
            
        except Exception as e:
            print(f"Cry detection error: {e}")
            return 0.0, False, 0.0

# ============================================
# UNSUPERVISED LEARNING: Anomaly Detection
# ============================================
class AnomalyDetector:
    """UNSUPERVISED LEARNING: IsolationForest with temporal context"""
    
    def __init__(self, window_size=30):  # FIXED: Changed from _init_ to __init__
        try:
            self.model = joblib.load("baby_model.pkl")
            print("✅ UNSUPERVISED ANOMALY MODEL LOADED: IsolationForest")
        except Exception as e:
            print(f"⚠️ Anomaly model not found, using statistical method: {e}")
            self.model = None
        
        self.window_size = window_size
        self.reading_history = deque(maxlen=window_size)
        self.baseline_stats = self.load_baseline_stats()
    
    def load_baseline_stats(self):
        """Load or create baseline statistics from database"""
        baseline = {}
        try:
            # SECURITY: Use parameterized query (even though no user input here)
            conn = mysql.connector.connect(
                host="localhost",
                user="root",
                password="root1234",
                database="baby_monitor"
            )
            
            # SECURITY: Using parameterized query with pandas read_sql
            query = """
            SELECT hour, sound_level, movement_level, temperature 
            FROM monitor_activity 
            WHERE (is_anomaly = 0 OR is_anomaly IS NULL)
            """
            # SECURITY: Parameters tuple (empty in this case)
            df = pd.read_sql(query, conn)
            conn.close()
            
            if len(df) > 100:  # Enough data
                for hour in range(24):
                    hour_data = df[df['hour'] == hour]
                    if len(hour_data) > 5:
                        baseline[hour] = {
                            'sound_mean': hour_data['sound_level'].mean(),
                            'sound_std': hour_data['sound_level'].std(),
                            'sound_p95': hour_data['sound_level'].quantile(0.95),
                            'sound_p5': hour_data['sound_level'].quantile(0.05),
                            'movement_mean': hour_data['movement_level'].mean(),
                            'movement_std': hour_data['movement_level'].std(),
                            'movement_p95': hour_data['movement_level'].quantile(0.95),
                        }
                    else:
                        # Fallback for hours with no data
                        baseline[hour] = self.get_default_stats(hour)
            else:
                # Not enough data, use defaults
                for hour in range(24):
                    baseline[hour] = self.get_default_stats(hour)
                    
        except Exception as e:
            print(f"Could not load baseline from DB: {e}")
            # Use reasonable defaults
            for hour in range(24):
                baseline[hour] = self.get_default_stats(hour)
        
        return baseline
    
    def get_default_stats(self, hour):
        """Provide reasonable default stats based on time of day"""
        # Night time (11 PM - 6 AM) should be quieter
        if 23 <= hour or hour <= 6:
            return {
                'sound_mean': 10.0,
                'sound_std': 5.0,
                'sound_p95': 20.0,
                'sound_p5': 2.0,
                'movement_mean': 5.0,
                'movement_std': 3.0,
                'movement_p95': 12.0,
            }
        # Day time - more activity
        else:
            return {
                'sound_mean': 25.0,
                'sound_std': 10.0,
                'sound_p95': 45.0,
                'sound_p5': 8.0,
                'movement_mean': 15.0,
                'movement_std': 8.0,
                'movement_p95': 30.0,
            }
    
    def add_reading(self, hour, temp, sound, movement):
        """Add a reading to history"""
        self.reading_history.append({
            'timestamp': time.time(),
            'hour': hour,
            'temp': temp,
            'sound': sound,
            'movement': movement
        })
    
    def extract_features(self):
        """Extract features matching training data"""
        if len(self.reading_history) < 1:
            return None
        
        df = pd.DataFrame(list(self.reading_history))
        
        # Use ONLY the features the model was trained with
        features = {
            'hour': df['hour'].iloc[-1],
            'temp': df['temp'].iloc[-1],
            'sound': df['sound'].iloc[-1],
            'movement': df['movement'].iloc[-1]
        }
        
        return pd.DataFrame([features])

    
    def detect_anomaly(self, hour, temp, sound, movement):
        """Enhanced anomaly detection with explanations"""
        
        # Add to history
        self.add_reading(hour, temp, sound, movement)
        
        # Initialize result with explanations
        result = {
            'is_anomaly': False,
            'score': 0.0,
            'explanations': [],
            'confidence': 0.0,
            'severity': 'low'  # low, medium, high
        }
        
        explanations = []
        anomaly_votes = 0
        total_checks = 0
        
        # ========== METHOD 1: ML-based Anomaly Detection ==========
        if self.model and len(self.reading_history) >= 5:
            features = self.extract_features()
            if features is not None:
                total_checks += 1
                # ML prediction
                ml_pred = self.model.predict(features)[0]
                if hasattr(self.model, 'decision_function'):
                    ml_score = self.model.decision_function(features)[0]
                    result['score'] = float(ml_score)
                
                if ml_pred == -1:  # -1 indicates anomaly in IsolationForest
                    anomaly_votes += 1
                    explanations.append("🤖 AI detected unusual pattern")
        
        # ========== METHOD 2: Statistical Anomaly Detection ==========
        hour_int = int(hour)
        if hour_int in self.baseline_stats:
            stats = self.baseline_stats[hour_int]
            
            # Check sound level against historical percentiles
            total_checks += 1
            if sound > stats['sound_p95']:
                anomaly_votes += 1
                explanations.append(
                    f"🔊 Sound is louder than usual for this time ({sound:.1f})"
                )
            
            total_checks += 1
            if sound < stats['sound_p5']:
                anomaly_votes += 1
                explanations.append(
                    f"🔇 Sound is quieter than usual for this time ({sound:.1f})"
                )
            
            # Z-score check (values > 2 standard deviations from mean)
            total_checks += 1
            sound_z = abs((sound - stats['sound_mean']) / (stats['sound_std'] + 0.001))
            if sound_z > 2:
                anomaly_votes += 1
                explanations.append(
                    f"📊 Unusual sound variation detected"
                )
        
        # ========== METHOD 3: Rule-based Anomaly Detection ==========
        # Time-based rules
        total_checks += 1
        if 23 <= hour or hour <= 6:  # Night hours (11 PM - 6 AM)
            if sound > 30:  # Loud at night
                anomaly_votes += 1
                explanations.append("🌙 High noise during sleeping hours")
        
        total_checks += 1
        if 7 <= hour <= 20:  # Day hours
            if sound < 5 and movement < 2:  # Too quiet during day
                anomaly_votes += 1
                explanations.append("☀️ Unusually quiet during daytime")
        
        # Movement vs Sound anomalies
        total_checks += 1
        if movement > 40 and sound < 5:  # High movement but completely silent
            anomaly_votes += 1
            explanations.append("🎥 Movement with no sound - please check camera")
        
        total_checks += 1
        if sound > 70:  # Extremely loud
            anomaly_votes += 1
            explanations.append("🚨 Very loud noise detected")
            result['severity'] = 'high'
        
        # Trend anomalies
        if len(self.reading_history) >= 10:
            df = pd.DataFrame(list(self.reading_history))
            
            total_checks += 1
            # Sudden spike in sound
            if len(df) > 5 and df['sound'].iloc[-1] > df['sound'].iloc[-6:-1].mean() * 3:
                anomaly_votes += 1
                explanations.append("📈 Sudden increase in sound")
            
            total_checks += 1
            # Prolonged high movement
            if df['movement'].tail(10).mean() > 30:
                anomaly_votes += 1
                explanations.append("🏃 Baby has been very active for a while")
        
        # ========== DETERMINE FINAL RESULT ==========
        if total_checks > 0:
            anomaly_ratio = anomaly_votes / total_checks
            
            # Determine if it's an anomaly
            if anomaly_ratio >= 0.3:  # At least 30% of checks flagged
                result['is_anomaly'] = True
                result['confidence'] = min(anomaly_ratio * 1.5, 0.95)  # Scale but cap at 0.95
                
                # Set severity
                if anomaly_ratio >= 0.6:
                    result['severity'] = 'high'
                elif anomaly_ratio >= 0.4:
                    result['severity'] = 'medium'
                
                # Deduplicate explanations
                result['explanations'] = list(dict.fromkeys(explanations))
            else:
                result['is_anomaly'] = False
                result['confidence'] = 1 - anomaly_ratio
                result['explanations'] = ["✅ Everything looks normal"]
        
        return result

# ============================================
# PAGE CONFIG - PARENT FRIENDLY UI
# ============================================
st.set_page_config(
    page_title="BabyGuardian - AI Baby Monitor", 
    page_icon="👶",
    layout="wide"
)

# Custom CSS for parent-friendly interface
st.markdown("""
<style>
    /* Main container styling */
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
    }
    
    /* Status cards */
    .status-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 5px solid #667eea;
    }
    
    .status-good {
        border-left-color: #48bb78;
    }
    
    .status-warning {
        border-left-color: #ecc94b;
    }
    
    .status-danger {
        border-left-color: #f56565;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: center;
    }
    
    /* Guide section */
    .guide-section {
        background: #f7fafc;
        padding: 2rem;
        border-radius: 15px;
        margin-top: 2rem;
        border: 1px solid #e2e8f0;
    }
    
    .guide-step {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
        padding: 0.5rem;
        background: white;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .step-number {
        width: 30px;
        height: 30px;
        background: #667eea;
        color: white;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        margin-right: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# PARENT-FRIENDLY HEADER
# ============================================
st.markdown("""
<div class="main-header">
    <h1>👶 BabyGuardian</h1>
    <p style="font-size: 1.2rem;">Your AI-Powered Baby Monitoring Assistant</p>
    <p>Peace of mind, powered by intelligent technology</p>
</div>
""", unsafe_allow_html=True)

# ============================================
# HOW TO USE SECTION (Parent-Friendly Guide)
# ============================================
with st.expander("📖 How to Use BabyGuardian (Click to open)", expanded=True):
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 👋 Welcome to BabyGuardian!
        
        This app helps you monitor your baby using AI technology. Here's how to get started:
        
        **📋 Quick Start Guide:**
        
        1. **🎤 Test Your Microphone** - Click the "Test Microphone" button below to make sure we can hear your baby
        2. **🎚️ Adjust Sensitivity** - Choose how sensitive you want the AI to be (start with "Medium")
        3. **▶️ Start Monitoring** - Click the green "Start Monitoring" button to begin
        4. **👀 Watch the Dashboard** - You'll see real-time updates of sound, movement, and AI insights
        5. **⏹️ Stop When Done** - Click "Stop" when you're finished
        
        **🎯 What the Colors Mean:**
        - 🟢 **Green** - Everything is normal
        - 🟡 **Yellow** - Something unusual detected (check on baby)
        - 🔴 **Red** - Immediate attention recommended
        
        **❓ Need Help?** 
        - The app will explain any alerts in plain language
        - All controls have helpful tooltips (hover over them!)
        - You can always stop monitoring and review the summary
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Understanding the Display
        
        **Sound Level:** Shows how loud it is in the nursery
        
        **Cry Detection:** How likely the AI thinks baby is crying
        
        **Unusual Activity:** Unusual patterns the AI detects
        
        **AI Confidence:** How sure the AI is about its predictions
        
        *Don't worry about the numbers - just watch the colors!*
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

def log_to_mysql(hour, temp, sound, movement, cry_prob, anomaly_result, is_cry):
    """Log monitoring data to database with SQL injection prevention"""
    try:
        # SECURITY: Validate ALL inputs before database insertion
        hour = validate_and_sanitize(hour, 'int', 0, 23, 0)
        temp = validate_and_sanitize(temp, 'float', 18, 35, 22)
        sound = validate_and_sanitize(sound, 'float', 0, 100, 0)
        movement = validate_and_sanitize(movement, 'float', 0, 255, 0)
        cry_prob = validate_and_sanitize(cry_prob, 'float', 0, 1, 0)
        
        # SECURITY: If any validation failed, don't proceed
        if None in [hour, temp, sound, movement, cry_prob]:
            print("❌ Security: Invalid input values detected, skipping database insert")
            return
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if new columns exist
        cursor.execute("SHOW COLUMNS FROM monitor_activity")
        columns = [col[0] for col in cursor.fetchall()]
        
        # SECURITY: Using parameterized query with %s placeholders
        # This prevents SQL injection by separating query structure from data
        if 'anomaly_explanations' in columns and 'severity' in columns:
            query = """
            INSERT INTO monitor_activity
            (hour, temperature, sound_level, movement_level,
             cry_probability, anomaly_score, is_cry, is_anomaly,
             anomaly_explanations, severity)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            # SECURITY: Sanitize string inputs as well
            explanations_str = ', '.join(anomaly_result['explanations'])[:255]
            explanations_str = validate_and_sanitize(explanations_str, 'str')
            severity = validate_and_sanitize(anomaly_result['severity'], 'str')
            
            values = (
                hour, temp, sound, movement,
                cry_prob, float(anomaly_result['score']), 
                int(is_cry), int(anomaly_result['is_anomaly']),
                explanations_str, severity
            )
        else:
            # Fallback to basic columns
            query = """
            INSERT INTO monitor_activity
            (hour, temperature, sound_level, movement_level,
             cry_probability, anomaly_score, is_cry, is_anomaly)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            values = (
                hour, temp, sound, movement, 
                cry_prob, float(anomaly_result['score']), 
                int(is_cry), int(anomaly_result['is_anomaly'])
            )
        
        # SECURITY: Execute with separate values tuple - SQL injection impossible here
        cursor.execute(query, values)
        conn.commit()
        cursor.close()
        conn.close()
        
    except mysql.connector.Error as e:
        st.error(f"MySQL Error: {e}")
        print(f"❌ Database error: {e}")
    except Exception as e:
        st.error(f"Error logging to database: {e}")
        print(f"❌ Unexpected error: {e}")

# ---------------------------------
# INITIALIZE MODELS
# ---------------------------------
# Sidebar for model status (moved to a less prominent position)
with st.sidebar:
    st.markdown("### 🛠️ System Status")
    
    # Initialize detectors
    cry_detector = CryDetector()
    anomaly_detector = AnomalyDetector(window_size=30)
    
    # Show status in a parent-friendly way
    if cry_detector.model:
        st.success("✅ Baby Cry Detector: Ready")
    else:
        st.error("❌ Cry Detector: Not Available")
    
    if anomaly_detector.model:
        st.success("✅ AI Pattern Detector: Ready")
    else:
        st.info("ℹ️ Pattern Detector: Using Smart Rules")

# ---------------------------------
# PARENT-FRIENDLY UI CONTROLS
# ---------------------------------
st.markdown("### 🎮 Controls")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 🌡️ Room Temperature")
    temp_sim = st.slider(
        "Adjust if you have a thermometer", 
        18, 35, 22,
        help="Set this to match your room temperature for more accurate AI analysis"
    )
    # SECURITY: Validate slider input
    temp_sim = validate_and_sanitize(temp_sim, 'float', 18, 35, 22)

with col2:
    st.markdown("#### 🎚️ AI Sensitivity")
    sensitivity = st.select_slider(
        "How sensitive should the AI be?",
        options=["Low (Fewer Alerts)", "Medium (Balanced)", "High (More Alerts)"],
        value="Medium (Balanced)",
        help="Low = fewer alerts (may miss subtle signs), High = more alerts (may have false alarms)"
    )
    # SECURITY: Validate sensitivity selection
    sensitivity = validate_and_sanitize(sensitivity, 'str')
    # Adjust threshold based on sensitivity
    sensitivity_map = {
        "Low (Fewer Alerts)": 0.5,
        "Medium (Balanced)": 0.3,
        "High (More Alerts)": 0.15
    }
    anomaly_threshold = sensitivity_map[sensitivity]

with col3:
    st.markdown("#### 🏷️ Current Status")
    status_box = st.empty()

# Audio test section - made more parent-friendly
with st.expander("🎤 Test Your Microphone (Recommended before first use)"):
    st.markdown("""
    **Why test?** This ensures we can hear your baby clearly. Just click the button and make a sound!
    """)
    
    if st.button("🔊 Run Microphone Test", use_container_width=True):
        with st.spinner("Listening for 2 seconds..."):
            fs = 44100
            recording = sd.rec(int(2.0 * fs), samplerate=fs, channels=1, dtype='float32', device=1)
            sd.wait()
            audio = recording.flatten()
            rms = np.sqrt(np.mean(audio**2))
            test_level = float(min(rms * 2000, 100.0))
            # SECURITY: Validate test level
            test_level = validate_and_sanitize(test_level, 'float', 0, 100, 0)
            
            col_mic1, col_mic2 = st.columns(2)
            with col_mic1:
                st.markdown(f"**Audio Level:** {test_level:.1f}/100")
            with col_mic2:
                if test_level > 10:
                    st.success("✅ Great! Microphone is working well")
                elif test_level > 5:
                    st.warning("⚠️ Sound is a bit low - try moving closer to baby")
                else:
                    st.error("❌ Can't hear anything - please check microphone")

# Start/Stop buttons - made prominent and parent-friendly
st.markdown("---")
col_start, col_stop, col_clear = st.columns([2, 1, 1])

with col_start:
    start_btn = st.button(
        "▶️ START MONITORING", 
        type="primary", 
        use_container_width=True,
        help="Click to begin monitoring your baby"
    )
    
with col_stop:
    stop_btn = st.button(
        "⏹️ STOP", 
        use_container_width=True,
        help="Click to stop monitoring"
    )
    
with col_clear:
    if st.button(
        "🗑️ Clear History", 
        use_container_width=True,
        help="Remove all past data and start fresh"
    ):
        st.session_state.sound_history = deque(maxlen=100)
        st.session_state.move_history = deque(maxlen=100)
        st.session_state.cry_history = deque(maxlen=100)
        st.rerun()

# Initialize session state for history
if 'sound_history' not in st.session_state:
    st.session_state.sound_history = deque(maxlen=100)
    st.session_state.move_history = deque(maxlen=100)
    st.session_state.cry_history = deque(maxlen=100)
    st.session_state.timestamps = deque(maxlen=100)
    st.session_state.run = False

# Update run state
if start_btn:
    st.session_state.run = True
if stop_btn:
    st.session_state.run = False

# Create layout with better spacing
st.markdown("---")
st.markdown("### 📊 Live Monitoring Dashboard")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown("#### 🔊 Sound Level")
    sound_chart = st.line_chart(np.zeros((1, 1)), use_container_width=True)

with chart_col2:
    st.markdown("#### 🏃 Baby's Movement")
    move_chart = st.line_chart(np.zeros((1, 1)), use_container_width=True)

# Metrics row with parent-friendly labels
st.markdown("#### 📈 Current Readings")
metric_cols = st.columns(4)

with metric_cols[0]:
    sound_metric = st.empty()
with metric_cols[1]:
    cry_metric = st.empty()
with metric_cols[2]:
    anomaly_metric = st.empty()
with metric_cols[3]:
    confidence_metric = st.empty()

# Anomaly explanations area
st.markdown("#### 🔍 AI Insights")
explanation_box = st.container()

# ---------------------------------
# CAMERA INITIALIZATION
# ---------------------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    st.error("❌ Webcam not detected. Please check your camera connection.")
    st.stop()

# Warm up camera
time.sleep(1)
ret, prev_frame = cap.read()
if not ret or prev_frame is None:
    st.error("❌ Failed to read from webcam.")
    st.stop()

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# ---------------------------------
# AUDIO FUNCTION
# ---------------------------------
def get_mic_audio_and_score():
    """Record audio and calculate sound level"""
    try:
        fs = 44100
        duration = 2.0
        chunk = int(duration * fs)
        
        recording = sd.rec(chunk, samplerate=fs, channels=1, dtype='float32', device=1)
        sd.wait()
        
        audio = recording.flatten()
        rms = np.sqrt(np.mean(audio**2))
        loudness = float(min(rms * 2000, 100.0))
        # SECURITY: Validate loudness
        loudness = validate_and_sanitize(loudness, 'float', 0, 100, 0)
        
        return audio, fs, loudness
        
    except Exception as e:
        print(f"Microphone error: {e}")
        return np.zeros(int(2.0 * 44100)), 44100, 0.0

# ---------------------------------
# MAIN LOOP
# ---------------------------------
if st.session_state.run:
    st.info("🟢 Monitoring active - you'll see live updates below")
    
    # Create placeholder for real-time updates
    explanation_placeholder = st.empty()
    
    while st.session_state.run:
        loop_start = time.time()
        
        # -------- VIDEO PROCESSING --------
        ret, frame = cap.read()
        if not ret or frame is None:
            st.warning("⚠️ Webcam frame not received. Reconnecting...")
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            continue
        
        # Convert to grayscale and calculate movement
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_diff = cv2.absdiff(prev_gray, gray)
        move_score = float(np.mean(frame_diff))
        # SECURITY: Validate movement score
        move_score = validate_and_sanitize(move_score, 'float', 0, 255, 0)
        prev_gray = gray
        
        # -------- AUDIO PROCESSING --------
        audio, audio_fs, sound_score = get_mic_audio_and_score()
        
        # -------- SUPERVISED: CRY DETECTION --------
        cry_prob, is_cry, cry_confidence = cry_detector.detect_cry(audio, audio_fs)
        # SECURITY: Validate cry probability
        cry_prob = validate_and_sanitize(cry_prob, 'float', 0, 1, 0)
        
        # -------- UNSUPERVISED: ANOMALY DETECTION --------
        hour_now = time.localtime().tm_hour
        # SECURITY: Validate hour
        hour_now = validate_and_sanitize(hour_now, 'int', 0, 23, 0)
        
        anomaly_result = anomaly_detector.detect_anomaly(
            hour=hour_now,
            temp=temp_sim,
            sound=sound_score,
            movement=move_score
        )
        
        # Override is_anomaly based on sensitivity
        if anomaly_result['confidence'] >= anomaly_threshold:
            anomaly_result['is_anomaly'] = True

        # Send to Node-RED if needed
        if is_cry or anomaly_result['is_anomaly']:
            try:
                # This sends the data to the "AI Input" node in Node-RED
                payload = {
                    "is_cry": bool(is_cry),
                    "is_anomaly": bool(anomaly_result['is_anomaly']),
                    "severity": anomaly_result['severity']
                }
                requests.post("http://localhost:1880/baby-alert", json=payload, timeout=0.1)
            except Exception as e:
                pass # Prevents app from crashing if Node-RED is closed
        
        # -------- UPDATE HISTORY --------
        current_time = time.time()
        st.session_state.timestamps.append(current_time)
        st.session_state.sound_history.append(sound_score)
        st.session_state.move_history.append(move_score)
        st.session_state.cry_history.append(cry_prob)
        
        # -------- UPDATE METRICS (Parent-Friendly Labels) --------
        with metric_cols[0]:
            delta = f"{sound_score - (st.session_state.sound_history[-2] if len(st.session_state.sound_history) > 1 else 0):.1f}"
            sound_metric.metric(
                "🔊 Sound Level", 
                f"{sound_score:.1f}/100",
                delta=delta,
                help="Higher numbers mean louder noise"
            )
        
        with metric_cols[1]:
            delta = f"{(cry_prob*100 - (st.session_state.cry_history[-2]*100 if len(st.session_state.cry_history) > 1 else 0)):.1f}%"
            cry_metric.metric(
                "😢 Cry Detection", 
                f"{cry_prob*100:.1f}%",
                delta=delta,
                delta_color="inverse",
                help="How likely the AI thinks baby is crying"
            )
        
        with metric_cols[2]:
            anomaly_metric.metric(
                "⚠️ Unusual Activity", 
                f"{anomaly_result['score']:.2f}",
                help="Higher numbers indicate more unusual patterns"
            )
        
        with metric_cols[3]:
            confidence_metric.metric(
                "🎯 AI Confidence", 
                f"{anomaly_result['confidence']*100:.0f}%",
                help="How sure the AI is about its analysis"
            )
        
        # -------- UPDATE CHARTS --------
        if len(st.session_state.sound_history) > 1:
            sound_df = pd.DataFrame({
                'Sound Level': list(st.session_state.sound_history)
            })
            sound_chart.line_chart(sound_df, use_container_width=True)
            
            move_df = pd.DataFrame({
                'Movement': list(st.session_state.move_history)
            })
            move_chart.line_chart(move_df, use_container_width=True)
        
        # -------- DISPLAY ANOMALY EXPLANATIONS (Parent-Friendly) --------
        with explanation_box:
            # Create columns for better layout
            exp_col1, exp_col2 = st.columns([1, 3])
            
            with exp_col1:
                if anomaly_result['is_anomaly'] and is_cry:
                    st.error(f"🚨 **CRITICAL**")
                elif anomaly_result['is_anomaly']:
                    if anomaly_result['severity'] == 'high':
                        st.error(f"🔴 **HIGH ALERT**")
                    elif anomaly_result['severity'] == 'medium':
                        st.warning(f"🟡 **MEDIUM ALERT**")
                    else:
                        st.info(f"🔵 **LOW ALERT**")
                elif is_cry:
                    st.warning(f"😢 **CRYING**")
                else:
                    st.success(f"✅ **NORMAL**")
            
            with exp_col2:
                for exp in anomaly_result['explanations']:
                    if '✅' in exp:
                        st.success(exp)
                    elif '🚨' in exp or '🔊' in exp or '🌙' in exp:
                        st.error(exp)
                    elif '⚠️' in exp or '😢' in exp:
                        st.warning(exp)
                    else:
                        st.info(exp)
        
        # -------- OVERALL STATUS (Parent-Friendly) --------
        if anomaly_result['is_anomaly'] and is_cry:
            status_box.error("🚨 **CRITICAL: Baby crying with unusual patterns - Please check immediately!**")
        elif is_cry:
            status_box.warning(f"😢 **Baby is crying** ({(cry_prob*100):.1f}% confidence)")
        elif anomaly_result['is_anomaly']:
            if anomaly_result['severity'] == 'high':
                status_box.error("🔴 **Unusual behavior detected - Please check on baby**")
            else:
                status_box.warning("🟡 **Unusual pattern detected - Keep an eye on baby**")
        else:
            status_box.success("✅ **All normal - Baby is doing well**")
        
        # -------- LOG TO DATABASE --------
        log_to_mysql(
            hour=hour_now,
            temp=temp_sim,
            sound=sound_score,
            movement=move_score,
            cry_prob=cry_prob,
            anomaly_result=anomaly_result,
            is_cry=is_cry
        )
        
        # Control loop speed (target ~10 FPS)
        elapsed = time.time() - loop_start
        if elapsed < 0.1:
            time.sleep(0.1 - elapsed)

# ---------------------------------
# CLEANUP
# ---------------------------------
cap.release()
if not st.session_state.run:
    st.success("🛑 Monitoring stopped")
    
    # Show summary if we have data
    if len(st.session_state.sound_history) > 0:
        st.markdown("---")
        st.markdown("### 📈 Session Summary")
        col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
        
        with col_sum1:
            avg_sound = np.mean(list(st.session_state.sound_history))
            st.metric("Avg Sound Level", f"{avg_sound:.1f}/100")
        
        with col_sum2:
            avg_move = np.mean(list(st.session_state.move_history))
            st.metric("Avg Movement", f"{avg_move:.1f}")
        
        with col_sum3:
            cry_events = sum(1 for c in list(st.session_state.cry_history) if c > 0.8)
            st.metric("Cry Events", cry_events)
        
        with col_sum4:
            anomaly_count = len([r for r in anomaly_detector.reading_history]) if hasattr(anomaly_detector, 'reading_history') else 0
            anomaly_pct = (anomaly_count / max(len(st.session_state.sound_history), 1)) * 100
            st.metric("Unusual Patterns", f"{anomaly_pct:.1f}%")