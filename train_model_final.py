# train_model_final.py
"""
Train a baby cry detection model using your labeled dataset
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent.absolute()))

try:
    import librosa
    import soundfile as sf
    import matplotlib.pyplot as plt
    import seaborn as sns
    print("✅ Imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nInstalling required packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn", "matplotlib", "seaborn", "joblib"])
    print("✅ Packages installed. Please run the script again.")
    sys.exit(1)

from config import LABELING_FOLDER, PROJECT_ROOT

def extract_features(audio_path):
    """
    Extract 40 audio features for classification
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
        
        return np.array(features)
        
    except Exception as e:
        print(f"❌ Error extracting features from {audio_path}: {e}")
        return None

def load_labeled_data():
    """Load labeled data from CSV"""
    print("\n📊 Loading labeled data...")
    
    csv_path = LABELING_FOLDER / "labels_clips_5seconds.csv"
    
    if not csv_path.exists():
        print(f"❌ CSV not found: {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    print(f"   Total rows: {len(df)}")
    
    # Filter labeled data
    df_labeled = df[df['labeled'] == True]
    print(f"   Labeled rows: {len(df_labeled)}")
    
    # Check class distribution
    class_dist = df_labeled['label'].value_counts()
    print(f"\n📈 Class Distribution:")
    for label, count in class_dist.items():
        percentage = (count / len(df_labeled)) * 100
        print(f"   • {label}: {count} ({percentage:.1f}%)")
    
    return df_labeled

def prepare_dataset(df_labeled):
    """Extract features for all labeled files"""
    print("\n🔍 Extracting 40 features from audio files...")
    
    X = []
    y = []
    failed_files = []
    
    for idx, row in df_labeled.iterrows():
        audio_path = row['file_path']
        
        # Check if file exists
        if not Path(audio_path).exists():
            # Try in clips_5seconds folder
            alt_path = PROJECT_ROOT / "clips_5seconds" / Path(audio_path).name
            if alt_path.exists():
                audio_path = str(alt_path)
            else:
                failed_files.append(row['filename'])
                continue
        
        # Extract features
        features = extract_features(audio_path)
        if features is not None:
            X.append(features)
            y.append(row['label'])
        
        # Progress
        if (idx + 1) % 20 == 0:
            print(f"   Processed {idx + 1}/{len(df_labeled)} files")
    
    print(f"\n✅ Successfully processed: {len(X)} files")
    print(f"   Feature vector size: {len(X[0]) if X else 0} features")
    
    if failed_files:
        print(f"⚠️  Failed to process: {len(failed_files)} files")
    
    return np.array(X), np.array(y)

def train_model(X, y):
    """Train models with scaling"""
    print("\n" + "="*60)
    print("🎯 TRAINING MODELS")
    print("="*60)
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"\n📋 Label encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\n📊 Data split:")
    print(f"   • Training: {len(X_train)} samples")
    print(f"   • Testing: {len(X_test)} samples")
    
    # Train models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42)
    }
    
    best_model = None
    best_score = 0
    best_name = ""
    results = {}
    
    for name, model in models.items():
        print(f"\n🔄 Training {name}...")
        
        # Train
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        results[name] = accuracy
        print(f"   ✅ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        if accuracy > best_score:
            best_score = accuracy
            best_model = model
            best_name = name
    
    print(f"\n🏆 Best model: {best_name} with accuracy {best_score:.4f} ({best_score*100:.2f}%)")
    
    # Detailed metrics
    print(f"\n📈 Classification Report for {best_name}:")
    y_pred_best = best_model.predict(X_test_scaled)
    print(classification_report(y_test, y_pred_best, target_names=le.classes_))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_best)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'Confusion Matrix - {best_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print(f"\n📸 Confusion matrix saved")
    
    return best_model, scaler, le, results

def save_model(model, scaler, label_encoder, results):
    """Save the trained model and scaler"""
    print("\n" + "="*60)
    print("💾 SAVING MODEL")
    print("="*60)
    
    # Save everything
    model_data = {
        'model': model,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'feature_size': 40,
        'results': results
    }
    
    joblib.dump(model_data, 'baby_cry_model_complete.pkl')
    print(f"✅ Complete model saved to: baby_cry_model_complete.pkl")
    
    # Save just the model for compatibility
    joblib.dump(model, 'cry_model.pkl')
    print(f"✅ Model only saved to: cry_model.pkl")
    
    # Save info
    with open('model_info.txt', 'w') as f:
        f.write("Baby Cry Detection Model\n")
        f.write("="*50 + "\n")
        f.write(f"Training date: {pd.Timestamp.now()}\n")
        f.write(f"Best model: {type(model).__name__}\n")
        f.write(f"Features: 40\n")
        f.write(f"Accuracy: {max(results.values())*100:.2f}%\n\n")
        f.write("Class distribution:\n")
        for name, acc in results.items():
            f.write(f"  {name}: {acc*100:.2f}%\n")

def main():
    """Main training function"""
    print("="*60)
    print("🎵 BABY CRY DETECTION - MODEL TRAINING")
    print("="*60)
    
    # Load data
    df_labeled = load_labeled_data()
    
    if df_labeled is None or len(df_labeled) < 10:
        print("❌ Not enough labeled data!")
        return
    
    # Prepare dataset
    X, y = prepare_dataset(df_labeled)
    
    if len(X) == 0:
        print("❌ No features could be extracted!")
        return
    
    # Train model
    best_model, scaler, label_encoder, results = train_model(X, y)
    
    # Save model
    save_model(best_model, scaler, label_encoder, results)
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print("\n📋 Next steps:")
    print("   1. Test your model: python test_model.py")
    print("   2. Use in your app: python app.py")

if __name__ == "__main__":
    main()