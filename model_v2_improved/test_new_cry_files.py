# test_new_cry_files_final.py
"""
Complete script to test 5 new cry files
Run this from your model_v2_improved folder
"""

import sys
from pathlib import Path
import joblib
import numpy as np
import librosa
import json
from datetime import datetime

print("="*70)
print("🎯 TESTING 5 NEW CRY FILES - COMPLETE VERSION")
print("="*70)

# Check current directory
current_dir = Path.cwd()
print(f"\n📁 Current folder: {current_dir}")

# Look for model in current folder
model_paths = [
    current_dir / "baby_cry_model_complete.pkl",
    current_dir / "cry_model.pkl"
]

model_data = None
model_file = None
for path in model_paths:
    if path.exists():
        model_file = path
        print(f"✅ Found model: {path.name}")
        model_data = joblib.load(path)
        break

if model_data is None:
    print("\n❌ No model found in current folder!")
    print("   Please make sure you're in the model_v2_improved folder")
    print("   Or copy the model to current folder")
    sys.exit(1)

# Extract model components
if isinstance(model_data, dict):
    model = model_data.get('model')
    scaler = model_data.get('scaler')
    label_encoder = model_data.get('label_encoder')
    print(f"   Model type: {type(model).__name__}")
    if label_encoder:
        print(f"   Classes: {label_encoder.classes_}")
else:
    model = model_data
    scaler = None
    label_encoder = None
    print(f"   Model type: {type(model).__name__}")

# Define the 5 new cry files (they should be in current folder)
new_cry_files = [
    "freesound_community-babycrying2018-62503.mp3",
    "freesound_community-babycry-6473.mp3", 
    "u_ak0pt95kqu-baby-crying-high-pitch-434113.mp3",
    "zubair9900-a-small-childx27s-cry-292598.mp3",
    "freesound_community-baby-crying-loud-100441.mp3"
]

# Check which files exist
print("\n🔍 Checking for test files in current folder:")
available_files = []
missing_files = []

for filename in new_cry_files:
    file_path = current_dir / filename
    if file_path.exists():
        size_kb = file_path.stat().st_size / 1024
        print(f"   ✅ {filename} ({size_kb:.1f} KB)")
        available_files.append(filename)
    else:
        print(f"   ❌ {filename} - NOT FOUND")
        missing_files.append(filename)

if not available_files:
    print("\n❌ No test files found! Please copy them to current folder:")
    print("   copy ..\\freesound_community-babycrying2018-62503.mp3 .")
    print("   copy ..\\freesound_community-babycry-6473.mp3 .")
    print("   copy ..\\u_ak0pt95kqu-baby-crying-high-pitch-434113.mp3 .")
    print("   copy ..\\zubair9900-a-small-childx27s-cry-292598.mp3 .")
    print("   copy ..\\freesound_community-baby-crying-loud-100441.mp3 .")
    sys.exit(1)

# Complete feature extraction function (40 features)
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
        
        # Convert to numpy array and reshape for model
        return np.array(features).reshape(1, -1)
        
    except Exception as e:
        print(f"   ❌ Error extracting features: {e}")
        return None

print("\n" + "="*70)
print("🔬 TESTING 5 NEW CRY FILES")
print("="*70)

results = []
cry_count = 0
not_cry_count = 0

for i, filename in enumerate(available_files, 1):
    file_path = current_dir / filename
    print(f"\n[{i}/{len(available_files)}] Testing: {filename}")
    
    # Extract features
    features = extract_features(str(file_path))
    if features is None:
        continue
    
    print(f"   📊 Features extracted: {features.shape[1]}")
    
    # Apply scaler if available
    if scaler is not None:
        features = scaler.transform(features)
    
    # Make prediction
    pred_num = model.predict(features)[0]
    
    # Convert to label
    if label_encoder:
        prediction = label_encoder.inverse_transform([pred_num])[0]
    else:
        prediction = 'cry' if pred_num == 0 else 'not-cry'
    
    # Count predictions
    if prediction.lower() == 'cry':
        cry_count += 1
    else:
        not_cry_count += 1
    
    # Get confidence scores
    confidence = 0
    probabilities = {}
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(features)[0]
        confidence = max(proba) * 100
        
        # Get class names for probabilities
        if label_encoder:
            class_names = label_encoder.classes_
        else:
            class_names = ['cry', 'not-cry']
        
        # Create probabilities dictionary
        probabilities = {
            class_names[j]: f"{proba[j]*100:.1f}%"
            for j in range(len(class_names))
        }
        
        print(f"   📊 Probabilities: {probabilities}")
    
    # Since these are all cry files
    expected = 'cry'
    correct = (prediction == expected)
    
    print(f"   🤖 Prediction: {prediction.upper()}")
    print(f"   📈 Confidence: {confidence:.1f}%")
    print(f"   {'✅ CORRECT' if correct else '❌ INCORRECT'}")
    
    results.append({
        'file': filename,
        'expected': expected,
        'predicted': prediction,
        'confidence': confidence,
        'probabilities': probabilities,
        'correct': correct
    })

# Summary
print("\n" + "="*70)
print("📊 TEST SUMMARY")
print("="*70)

if results:
    correct = sum(1 for r in results if r['correct'])
    total = len(results)
    accuracy = (correct / total) * 100
    
    print(f"\n📈 Overall Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
    # ADD THIS SECTION - CRY vs NOT-CRY PERCENTAGES
    print("\n" + "="*70)
    print("📊 PREDICTION DISTRIBUTION")
    print("="*70)
    cry_percentage = (cry_count / total) * 100
    not_cry_percentage = (not_cry_count / total) * 100
    
    print(f"\n🔴 CRY predictions: {cry_count}/{total} ({cry_percentage:.1f}%)")
    print(f"🟢 NOT-CRY predictions: {not_cry_count}/{total} ({not_cry_percentage:.1f}%)")
    
    # Visual bar
    bar_length = 40
    cry_bars = int((cry_percentage / 100) * bar_length)
    not_cry_bars = bar_length - cry_bars
    print(f"\n   [{'🔴' * cry_bars}{'🟢' * not_cry_bars}]")
    print(f"   {cry_percentage:.1f}% CRY {' ' * 15} {not_cry_percentage:.1f}% NOT-CRY")
    
    print("\n📋 Detailed Results:")
    print("-" * 80)
    for r in results:
        mark = "✅" if r['correct'] else "❌"
        print(f"{mark} {r['file']}")
        print(f"   Expected: {r['expected'].upper()}, Predicted: {r['predicted'].upper()} ({r['confidence']:.1f}%)")
        if r['probabilities']:
            print(f"   Probs: {r['probabilities']}")
    
    # Comparison with previous test
    print("\n🔍 Performance Comparison:")
    print(f"   • Previous test set (10 files): 100%")
    print(f"   • New cry files ({total} files): {accuracy:.2f}%")
    
    if accuracy == 100:
        print("\n🎉 PERFECT! Model handles all new cry files!")
    elif accuracy >= 80:
        print("\n👍 GOOD! Model is robust on most new cries")
    elif accuracy >= 50:
        print("\n⚠️ ACCEPTABLE! Model gets about half right")
    else:
        print("\n📌 Model struggles with these cry types")
        print("   These represent NEW variations not in training")
        print("   Consider adding them to training for v3")
    
    # Recommendation
    print("\n💡 Recommendation:")
    if accuracy < 100:
        print("   Add these files to training to improve model:")
        for r in results:
            if not r['correct']:
                print(f"   • {r['file']} (predicted as {r['predicted'].upper()})")
else:
    print("❌ No files were tested")

# Save results
report = {
    'timestamp': datetime.now().isoformat(),
    'test_type': '5 New Cry Files',
    'total_files': len(results),
    'correct': correct if results else 0,
    'accuracy': accuracy if results else 0,
    'cry_predictions': cry_count,
    'not_cry_predictions': not_cry_count,
    'cry_percentage': cry_percentage if results else 0,
    'not_cry_percentage': not_cry_percentage if results else 0,
    'results': results
}

report_file = current_dir / 'new_cry_test_results.json'
with open(report_file, 'w') as f:
    json.dump(report, f, indent=2)

print(f"\n💾 Results saved to: {report_file}")
print("="*70)

# Instructions if files missing
if missing_files:
    print("\n📋 Files not found in current folder:")
    for f in missing_files:
        print(f"   • {f}")
    print("\nTo copy them from root folder:")
    print("cd ..")
    for f in missing_files:
        print(f"copy {f} model_v2_improved\\")
    print("cd model_v2_improved")
    print("python test_new_cry_files_final.py")