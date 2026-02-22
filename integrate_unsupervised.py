# integrate_unsupervised.py
"""
Add this to your main app.py to integrate unsupervised learning
"""

import joblib
import numpy as np
import cv2
from datetime import datetime

class BabyStateDetector:
    """
    Real-time baby state detection using unsupervised learning
    """
    
    def __init__(self):
        # Load your trained model
        try:
            model_data = joblib.load('baby_state_model.pkl')
            self.kmeans = model_data['kmeans']
            self.state_names = model_data['state_names']
            print("✅ Unsupervised model loaded")
        except:
            print("⚠️ Using fallback rules")
            self.kmeans = None
            self.state_names = {
                0: "😴 Deep Sleep",
                1: "😊 Active & Awake", 
                2: "😢 Intense Crying"
            }
    
    def extract_movement_features(self, frame, prev_frame):
        """
        Extract simple movement features from video frame
        Similar to how you trained the model
        """
        if prev_frame is None:
            return [0, 0, 0, 0]
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate frame difference
        diff = cv2.absdiff(prev_gray, gray)
        
        # Extract features (same as training)
        mean_movement = float(np.mean(diff))
        std_movement = float(np.std(diff))
        max_movement = float(np.max(diff))
        movement_range = float(np.ptp(diff))
        
        return [mean_movement, std_movement, max_movement, movement_range]
    
    def detect_state(self, frame, prev_frame, hour=None):
        """
        Detect current baby state from video frame
        """
        if hour is None:
            hour = datetime.now().hour
        
        # Extract features
        features = self.extract_movement_features(frame, prev_frame)
        
        if self.kmeans:
            # Use ML model
            cluster = self.kmeans.predict([features])[0]
            state = self.state_names.get(cluster, f"State {cluster}")
            
            # Get confidence (distance to cluster center)
            distances = np.linalg.norm(
                self.kmeans.cluster_centers_ - features, axis=1
            )
            confidence = 1 - (distances[cluster] / np.sum(distances))
            
            return state, confidence, features
        else:
            # Fallback rules
            if features[0] < 2:
                return "😴 Deep Sleep", 0.7, features
            elif features[0] < 5:
                return "😊 Active & Awake", 0.6, features
            else:
                return "😢 Intense Crying", 0.8, features

# ============================================
# HOW TO ADD TO YOUR APP.PY
# ============================================

"""
Add this code to your existing app.py:

--------------------------------------------------------------------------------

# At the top with other imports
from integrate_unsupervised import BabyStateDetector

# After loading your other models
state_detector = BabyStateDetector()

# In your main loop where you have video frames
if 'prev_frame' not in st.session_state:
    st.session_state.prev_frame = None

# After reading frame
if st.session_state.prev_frame is not None:
    state, confidence, features = state_detector.detect_state(
        frame, 
        st.session_state.prev_frame
    )
    
    # Display in Streamlit
    st.sidebar.markdown("---")
    st.sidebar.subheader("👶 Baby State Detection")
    st.sidebar.markdown(f"### {state}")
    st.sidebar.progress(confidence)
    st.sidebar.caption(f"Confidence: {confidence:.1%}")
    
    # Show movement features
    with st.sidebar.expander("Movement Details"):
        st.write(f"Mean: {features[0]:.2f}")
        st.write(f"Std: {features[1]:.2f}")
        st.write(f"Max: {features[2]:.2f}")

# Update previous frame
st.session_state.prev_frame = frame.copy()
"""

# ============================================
# TEST THE INTEGRATION
# ============================================

if __name__ == "__main__":
    print("="*60)
    print("🔍 TESTING BABY STATE DETECTOR")
    print("="*60)
    
    # Initialize detector
    detector = BabyStateDetector()
    
    # Test with simulated frames
    import cv2
    
    # Create test frames (simulated)
    test_cases = [
        ("Still baby", np.zeros((480, 640, 3), dtype=np.uint8)),
        ("Moving baby", np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)),
    ]
    
    prev_frame = test_cases[0][1]
    
    for name, frame in test_cases:
        print(f"\n📝 Testing: {name}")
        state, confidence, features = detector.detect_state(frame, prev_frame)
        print(f"   State: {state}")
        print(f"   Confidence: {confidence:.1%}")
        print(f"   Features: {features}")
        prev_frame = frame
    
    print("\n✅ Integration test complete!")