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