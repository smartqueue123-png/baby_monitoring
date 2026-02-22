# ultimate_test.py
"""
Combined test script - with COMPLETE 40-feature extraction
"""

import sys
from pathlib import Path
import joblib
import numpy as np
import librosa

print("="*70)
print("🎯 ULTIMATE BABY CRY MODEL TEST (FIXED 40 FEATURES)")
print("="*70)

# Find current folder
current_folder = Path.cwd()
print(f"\n📁 Current folder: {current_folder}")

# Look for model in current folder and parent folder
model_paths = [
    current_folder / "baby_cry_model_complete.pkl",
    current_folder / "cry_model.pkl",
    current_folder.parent / "baby_cry_model_complete.pkl",
    current_folder.parent / "cry_model.pkl"
]

model_data = None
model_file = None
for path in model_paths:
    if path.exists():
        model_file = path
        print(f"✅ Found model: {path}")
        model_data = joblib.load(path)
        break

if model_data is None:
    print("❌ No model found anywhere!")
    sys.exit(1)

# Extract model components
if isinstance(model_data, dict):
    model = model_data.get('model')
    scaler = model_data.get('scaler')
    label_encoder = model_data.get('label_encoder')
else:
    model = model_data
    scaler = None
    label_encoder = None

print(f"   Model type: {type(model).__name__}")
if label_encoder:
    print(f"   Classes: {label_encoder.classes_}")

# Look for test set in multiple locations
test_folders = [
    current_folder / "test set",
    current_folder.parent / "test set",
    current_folder / "test_set",
    current_folder.parent / "test_set",
    Path("..") / "test set",
    Path("../test set")
]

test_files = []
for folder in test_folders:
    if folder.exists():
        print(f"\n🔍 Found test folder: {folder}")
        test_files = list(folder.glob("*.mp3")) + list(folder.glob("*.wav"))
        if test_files:
            print(f"✅ Found {len(test_files)} test files")
            break

if not test_files:
    print("\n❌ No test files found anywhere!")
    print("   Looked in:")
    for f in test_folders:
        print(f"   • {f}")
    sys.exit(1)

# COMPLETE feature extraction with 40 features
def extract_features(audio_path):
    """Extract EXACTLY 40 features to match the model"""
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
        
        return np.array(features).reshape(1, -1)
        
    except Exception as e:
        print(f"   ❌ Feature extraction error: {e}")
        return None

# Simple label guesser from filename
def guess_label(filename):
    f = filename.lower()
    if any(word in f for word in ['cry', 'crying', 'moaning', 'newborn']):
        return 'cry'
    return 'not-cry'

# Test each file
print("\n" + "="*70)
print("🔬 RUNNING TESTS")
print("="*70)

results = []
for i, file in enumerate(test_files, 1):
    print(f"\n[{i}/{len(test_files)}] {file.name}")
    
    expected = guess_label(file.name)
    print(f"   Expected: {expected.upper()}")
    
    # Extract features
    features = extract_features(str(file))
    if features is None:
        print("   ❌ Skipping - feature extraction failed")
        continue
    
    print(f"   Features extracted: {features.shape[1]}")
    
    # Apply scaler if available
    if scaler is not None:
        features = scaler.transform(features)
    
    # Predict
    pred_num = model.predict(features)[0]
    
    # Convert to label
    if label_encoder:
        prediction = label_encoder.inverse_transform([pred_num])[0]
    else:
        prediction = 'cry' if pred_num == 0 else 'not-cry'
    
    # Get confidence
    confidence = 0
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(features)[0]
        confidence = max(proba) * 100
    
    correct = (prediction == expected)
    results.append({
        'file': file.name,
        'expected': expected,
        'predicted': prediction,
        'confidence': confidence,
        'correct': correct
    })
    
    print(f"   Predicted: {prediction.upper()} ({confidence:.1f}%)")
    print(f"   {'✅ CORRECT' if correct else '❌ INCORRECT'}")

# Summary
print("\n" + "="*70)
print("📊 RESULTS SUMMARY")
print("="*70)

if not results:
    print("\n❌ No tests completed successfully")
    sys.exit(1)

total = len(results)
correct = sum(1 for r in results if r['correct'])
accuracy = (correct / total) * 100

print(f"\n📈 Overall Accuracy: {accuracy:.2f}% ({correct}/{total})")

# Per-category
cry_results = [r for r in results if r['expected'] == 'cry']
not_cry_results = [r for r in results if r['expected'] == 'not-cry']

if cry_results:
    cry_correct = sum(1 for r in cry_results if r['correct'])
    cry_acc = (cry_correct / len(cry_results)) * 100
    print(f"\n🔴 CRY Detection: {cry_acc:.2f}% ({cry_correct}/{len(cry_results)})")

if not_cry_results:
    not_cry_correct = sum(1 for r in not_cry_results if r['correct'])
    not_cry_acc = (not_cry_correct / len(not_cry_results)) * 100
    print(f"🟢 NOT-CRY Detection: {not_cry_acc:.2f}% ({not_cry_correct}/{len(not_cry_results)})")

# Detailed table
print("\n📋 DETAILED RESULTS:")
print("-" * 80)
print(f"{'File':<40} {'Expected':<10} {'Predicted':<10} {'Confidence':<10} {'Result'}")
print("-" * 80)

for r in sorted(results, key=lambda x: x['file']):
    result_mark = "✅" if r['correct'] else "❌"
    print(f"{r['file'][:38]:<40} {r['expected']:<10} {r['predicted']:<10} {r['confidence']:.1f}%     {result_mark}")

# Save results
import json
from datetime import datetime
report = {
    'timestamp': datetime.now().isoformat(),
    'accuracy': accuracy,
    'cry_accuracy': cry_acc if cry_results else None,
    'not_cry_accuracy': not_cry_acc if not_cry_results else None,
    'total_files': total,
    'correct_files': correct,
    'results': results
}

report_file = Path('ultimate_test_results.json')
with open(report_file, 'w') as f:
    json.dump(report, f, indent=2)

print(f"\n💾 Full report saved to: {report_file}")
print("="*70)