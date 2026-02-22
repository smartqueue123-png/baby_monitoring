# test_model.py
"""
Quick test for your trained model - FIXED VERSION
"""

import sys
from pathlib import Path
import joblib
import numpy as np
import librosa

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent.absolute()))

def extract_features(audio_path):
    """
    Extract EXACTLY 40 features to match the model
    """
    try:
        # Load audio
        audio, sr = librosa.load(audio_path, sr=44100)
        
        features = []
        
        # 1. MFCC (13 mean + 13 std = 26 features)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        features.extend(mfcc_mean)  # 13
        features.extend(mfcc_std)   # 13 (total 26)
        
        # 2. Spectral features (5 features)
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)
        features.append(np.mean(spectral_centroids))  # 1 (27)
        features.append(np.std(spectral_centroids))   # 1 (28)
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        features.append(np.mean(spectral_rolloff))    # 1 (29)
        features.append(np.std(spectral_rolloff))     # 1 (30)
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        features.append(np.mean(spectral_bandwidth))  # 1 (31)
        
        # 3. Zero crossing rate (2 features)
        zcr = librosa.feature.zero_crossing_rate(audio)
        features.append(np.mean(zcr))                  # 1 (32)
        features.append(np.std(zcr))                   # 1 (33)
        
        # 4. RMS energy (2 features)
        rms = librosa.feature.rms(y=audio)
        features.append(np.mean(rms))                  # 1 (34)
        features.append(np.std(rms))                   # 1 (35)
        
        # 5. Chroma features (2 features)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        features.append(np.mean(chroma))               # 1 (36)
        features.append(np.std(chroma))                # 1 (37)
        
        # 6. Mel-spectrogram (2 features)
        mel = librosa.feature.melspectrogram(y=audio, sr=sr)
        features.append(np.mean(mel))                  # 1 (38)
        features.append(np.std(mel))                   # 1 (39)
        
        # 7. Spectral contrast (1 feature)
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        features.append(np.mean(contrast))             # 1 (40)
        
        return np.array(features).reshape(1, -1)  # Return as 2D array for prediction
        
    except Exception as e:
        print(f"   ❌ Error extracting features: {e}")
        return None

def load_model():
    """Load model and try to get label encoder"""
    # Try to load the complete model first
    try:
        model_data = joblib.load('baby_cry_model_complete.pkl')
        if isinstance(model_data, dict):
            model = model_data.get('model')
            scaler = model_data.get('scaler')
            label_encoder = model_data.get('label_encoder')
            print("✅ Loaded complete model with label encoder")
            return model, scaler, label_encoder
    except:
        pass
    
    # Try baby_cry_model.pkl
    try:
        model_data = joblib.load('baby_cry_model.pkl')
        if isinstance(model_data, dict):
            model = model_data.get('model')
            scaler = model_data.get('scaler')
            label_encoder = model_data.get('label_encoder')
            print("✅ Loaded model from baby_cry_model.pkl")
            return model, scaler, label_encoder
    except:
        pass
    
    # Try cry_model.pkl
    try:
        model = joblib.load('cry_model.pkl')
        print("✅ Loaded cry_model.pkl (no label encoder)")
        return model, None, None
    except Exception as e:
        print(f"❌ No model found! Error: {e}")
        return None, None, None

def interpret_prediction(prediction, label_encoder):
    """Convert numeric prediction to label"""
    if label_encoder is not None:
        # Use label encoder to convert back to text
        return label_encoder.inverse_transform([prediction])[0]
    else:
        # Manual mapping (assuming 0=cry, 1=not-cry or vice versa)
        if prediction == 0:
            return 'cry'
        elif prediction == 1:
            return 'not-cry'
        else:
            return str(prediction)

def test_model():
    """Test the model on a few sample clips"""
    print("="*60)
    print("🔍 TESTING BABY CRY MODEL")
    print("="*60)
    
    # Load model
    model, scaler, label_encoder = load_model()
    
    if model is None:
        print("\n❌ Please train the model first with: python train_model_final.py")
        return
    
    # Find test clips
    from config import CLIPS_5S_FOLDER
    
    # Get clips
    all_clips = list(CLIPS_5S_FOLDER.glob("*.wav"))
    cry_clips = [c for c in all_clips if 'crying' in c.stem or 'cry' in c.stem]
    not_cry_clips = [c for c in all_clips if any(x in c.stem for x in ['laugh', 'giggle', 'talk', 'baby-boy'])]
    
    # Take 3 from each
    test_clips = cry_clips[:3] + not_cry_clips[:3]
    
    if not test_clips:
        print("❌ No test clips found in clips_5seconds folder!")
        return
    
    print(f"\n📋 Testing on {len(test_clips)} clips:\n")
    print("-"*60)
    
    correct = 0
    total = 0
    
    for i, clip_path in enumerate(test_clips):
        try:
            print(f"\nTest {i+1}/{len(test_clips)}:")
            print(f"   📁 File: {clip_path.name}")
            
            # Extract features
            features = extract_features(str(clip_path))
            
            if features is None:
                continue
            
            # Apply scaler if available
            if scaler is not None:
                features = scaler.transform(features)
            
            # Make prediction
            prediction_num = model.predict(features)[0]
            
            # Convert to label
            prediction = interpret_prediction(prediction_num, label_encoder)
            
            # Get expected label from filename
            if 'crying' in clip_path.stem or 'cry' in clip_path.stem:
                expected = 'cry'
            else:
                expected = 'not-cry'
            
            # Get probability if available
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(features)[0]
                confidence = max(proba) * 100
                print(f"   📊 Confidence: {confidence:.1f}%")
            
            # Show result
            print(f"   🎯 Expected: {expected.upper()}")
            print(f"   🤖 Predicted: {prediction.upper()}")
            
            # Check if correct
            if prediction.lower() == expected.lower():
                print(f"   ✅ CORRECT!")
                correct += 1
            else:
                print(f"   ❌ INCORRECT!")
            total += 1
            
        except Exception as e:
            print(f"   ❌ Error on {clip_path.name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*60)
    if total > 0:
        accuracy = (correct / total) * 100
        print(f"📊 SUMMARY: {correct}/{total} correct ({accuracy:.1f}%)")
    else:
        print("📊 SUMMARY: No tests completed successfully")
    print("="*60)

if __name__ == "__main__":
    test_model()