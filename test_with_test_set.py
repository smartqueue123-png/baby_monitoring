# test_with_test_set.py
"""
Test the model using ONLY the files in the 'test set' folder
These are completely new files not used in training
"""

import sys
from pathlib import Path
import joblib
import numpy as np
import librosa
import soundfile as sf
import pandas as pd
from datetime import datetime
import json

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent.absolute()))

try:
    from config import WAV_FOLDER, TARGET_SR
    print("✅ Imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    # Define fallback values
    WAV_FOLDER = Path("converted_wav")
    TARGET_SR = 44100

class TestSetEvaluator:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.test_results = []
        self.test_folder = Path("test set")
        self.temp_folder = Path("test_temp")
        self.temp_folder.mkdir(exist_ok=True)
        
    def load_model(self):
        """Load the trained model"""
        print("\n📦 Loading model...")
        
        # Try different model files
        model_files = [
            'baby_cry_model_complete.pkl',
            'baby_cry_model.pkl', 
            'cry_model.pkl'
        ]
        
        for model_file in model_files:
            try:
                if not Path(model_file).exists():
                    continue
                    
                data = joblib.load(model_file)
                
                if isinstance(data, dict):
                    self.model = data.get('model')
                    self.scaler = data.get('scaler')
                    self.label_encoder = data.get('label_encoder')
                else:
                    self.model = data
                    self.scaler = None
                    self.label_encoder = None
                
                print(f"✅ Loaded model from: {model_file}")
                
                # Try to determine label mapping
                if self.label_encoder is not None:
                    print(f"   Label classes: {self.label_encoder.classes_}")
                elif hasattr(self.model, 'classes_'):
                    print(f"   Model classes: {self.model.classes_}")
                
                return True
                
            except Exception as e:
                print(f"   ⚠️  Error loading {model_file}: {e}")
                continue
        
        print("❌ Could not load any model file!")
        return False
    
    def find_test_files(self):
        """Find all audio files in the test set folder"""
        print(f"\n🔍 Scanning test folder: {self.test_folder}")
        
        if not self.test_folder.exists():
            print(f"❌ Test folder not found: {self.test_folder}")
            return []
        
        # Find all audio files
        test_files = []
        for ext in ['*.mp3', '*.wav', '*.m4a', '*.ogg']:
            test_files.extend(self.test_folder.glob(ext))
        
        # Sort for consistency
        test_files = sorted(test_files)
        
        print(f"   Found {len(test_files)} test files:")
        for f in test_files:
            size = f.stat().st_size / (1024 * 1024)
            print(f"   • {f.name} ({size:.2f} MB)")
        
        return test_files
    
    def determine_expected_label(self, filename):
        """Determine expected label from filename"""
        filename = filename.lower()
        
        # Cry indicators
        cry_words = ['cry', 'crying', 'moaning', 'newborn']
        # Not-cry indicators
        not_cry_words = ['laugh', 'giggle', 'gurgle', 'lach']
        
        if any(word in filename for word in cry_words):
            return 'cry'
        elif any(word in filename for word in not_cry_words):
            return 'not-cry'
        else:
            return 'unknown'
    
    def convert_to_wav(self, audio_path):
        """Convert any audio file to WAV for processing"""
        try:
            # Load audio (librosa handles many formats)
            audio, sr = librosa.load(str(audio_path), sr=TARGET_SR, mono=True)
            
            # Save as WAV in temp folder
            wav_filename = f"test_{audio_path.stem}.wav"
            wav_path = self.temp_folder / wav_filename
            sf.write(str(wav_path), audio, sr)
            
            return wav_path
            
        except Exception as e:
            print(f"   ❌ Conversion error: {e}")
            return None
    
    def extract_features(self, audio_path):
        """Extract 40 features for prediction"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=TARGET_SR)
            
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
    
    def predict_file(self, audio_path):
        """Predict on a single file"""
        try:
            # Convert to WAV if needed
            if audio_path.suffix.lower() != '.wav':
                wav_path = self.convert_to_wav(audio_path)
                if not wav_path:
                    return None
            else:
                wav_path = audio_path
            
            # Extract features
            features = self.extract_features(str(wav_path))
            if features is None:
                return None
            
            # Apply scaler if available
            if self.scaler is not None:
                features = self.scaler.transform(features)
            
            # Get prediction
            pred_num = self.model.predict(features)[0]
            
            # Convert to label
            if self.label_encoder is not None:
                prediction = self.label_encoder.inverse_transform([pred_num])[0]
            elif hasattr(self.model, 'classes_'):
                prediction = self.model.classes_[pred_num]
            else:
                # Manual mapping based on your training data
                prediction = 'cry' if pred_num == 0 else 'not-cry'
            
            # Get confidence
            confidence = 0
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(features)[0]
                confidence = max(proba) * 100
                # Get all class probabilities
                all_probs = dict(zip(
                    self.model.classes_ if hasattr(self.model, 'classes_') else ['cry', 'not-cry'],
                    proba
                ))
            else:
                all_probs = {}
            
            # Get duration
            duration = librosa.get_duration(path=str(wav_path))
            
            return {
                'file': audio_path.name,
                'predicted': prediction,
                'confidence': confidence,
                'all_probabilities': all_probs,
                'duration': round(duration, 2)
            }
            
        except Exception as e:
            print(f"   ❌ Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_test(self):
        """Run the complete test on all test set files"""
        print("="*70)
        print("🧪 TESTING MODEL WITH TEST SET FILES")
        print("="*70)
        
        # Load model
        if not self.load_model():
            return
        
        # Find test files
        test_files = self.find_test_files()
        
        if not test_files:
            print("\n❌ No test files found!")
            return
        
        print("\n" + "="*70)
        print("🔬 RUNNING PREDICTIONS")
        print("="*70)
        
        # Test each file
        for i, file_path in enumerate(test_files, 1):
            print(f"\n[{i}/{len(test_files)}] Testing: {file_path.name}")
            
            # Determine expected label
            expected = self.determine_expected_label(file_path.name)
            print(f"   Expected: {expected.upper() if expected != 'unknown' else 'UNKNOWN'}")
            
            # Make prediction
            result = self.predict_file(file_path)
            
            if result:
                result['expected'] = expected
                result['correct'] = (result['predicted'].lower() == expected.lower()) if expected != 'unknown' else None
                self.test_results.append(result)
                
                print(f"   Predicted: {result['predicted'].upper()}")
                print(f"   Confidence: {result['confidence']:.1f}%")
                
                if expected != 'unknown':
                    print(f"   {'✅ CORRECT' if result['correct'] else '❌ INCORRECT'}")
                
                # Show top probabilities
                if result['all_probabilities']:
                    probs_str = ", ".join([f"{k}: {v:.1f}%" for k, v in result['all_probabilities'].items()])
                    print(f"   Probabilities: {probs_str}")
        
        # Generate report
        self.generate_report()
    
    def generate_report(self):
        """Generate comprehensive test report"""
        print("\n" + "="*70)
        print("📊 TEST SET EVALUATION REPORT")
        print("="*70)
        
        if not self.test_results:
            print("\n❌ No test results to report")
            return
        
        # Overall statistics
        total = len(self.test_results)
        known_results = [r for r in self.test_results if r['expected'] != 'unknown']
        unknown_results = [r for r in self.test_results if r['expected'] == 'unknown']
        
        print(f"\n📈 SUMMARY:")
        print(f"   • Total files tested: {total}")
        print(f"   • Files with known labels: {len(known_results)}")
        print(f"   • Files with unknown labels: {len(unknown_results)}")
        
        if known_results:
            correct = sum(1 for r in known_results if r['correct'])
            accuracy = (correct / len(known_results)) * 100
            
            print(f"\n🎯 ACCURACY (on known files):")
            print(f"   • Correct: {correct}/{len(known_results)}")
            print(f"   • Accuracy: {accuracy:.2f}%")
        
        # Per-category breakdown
        if known_results:
            print(f"\n📊 PER-CATEGORY PERFORMANCE:")
            categories = {}
            for r in known_results:
                cat = r['expected']
                if cat not in categories:
                    categories[cat] = {'total': 0, 'correct': 0}
                categories[cat]['total'] += 1
                if r['correct']:
                    categories[cat]['correct'] += 1
            
            for cat, stats in categories.items():
                cat_acc = (stats['correct'] / stats['total']) * 100
                print(f"\n   {cat.upper()}:")
                print(f"      • Files: {stats['total']}")
                print(f"      • Correct: {stats['correct']}")
                print(f"      • Accuracy: {cat_acc:.2f}%")
        
        # Confidence analysis
        print(f"\n📉 CONFIDENCE ANALYSIS:")
        avg_conf = np.mean([r['confidence'] for r in self.test_results])
        print(f"   • Average confidence: {avg_conf:.2f}%")
        
        if known_results:
            correct_conf = np.mean([r['confidence'] for r in known_results if r['correct']])
            incorrect_conf = np.mean([r['confidence'] for r in known_results if not r['correct']]) if any(not r['correct'] for r in known_results) else 0
            print(f"   • Avg confidence (correct): {correct_conf:.2f}%")
            print(f"   • Avg confidence (incorrect): {incorrect_conf:.2f}%")
        
        # Detailed results table
        print(f"\n📋 DETAILED RESULTS:")
        print("-" * 90)
        print(f"{'File':<45} {'Expected':<10} {'Predicted':<10} {'Confidence':<10} {'Result':<8}")
        print("-" * 90)
        
        for r in sorted(self.test_results, key=lambda x: x['file']):
            result_mark = "✅" if r.get('correct') else "❌" if r.get('correct') is False else "⚠️"
            expected_display = r['expected'].upper() if r['expected'] != 'unknown' else '?'
            print(f"{r['file'][:43]:<45} {expected_display:<10} {r['predicted'].upper():<10} {r['confidence']:.1f}%     {result_mark}")
        
        # Save detailed report
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_files': total,
            'known_files': len(known_results),
            'accuracy': accuracy if known_results else None,
            'average_confidence': float(avg_conf),
            'per_category': categories if known_results else {},
            'results': self.test_results
        }
        
        report_file = Path('test_set_report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n💾 Full report saved to: {report_file}")
        
        # Recommendations
        print(f"\n🎯 MODEL ASSESSMENT:")
        if known_results:
            if accuracy >= 90:
                print("   🏆 EXCELLENT! Model is production-ready")
            elif accuracy >= 85:
                print("   👍 VERY GOOD! Model performs well")
            elif accuracy >= 80:
                print("   ✅ GOOD! Model is acceptable")
            elif accuracy >= 75:
                print("   ⚠️ ACCEPTABLE! Consider more training data")
            else:
                print("   🔧 NEEDS IMPROVEMENT! Add more diverse training data")
        
        # List unknown files for manual review
        if unknown_results:
            print(f"\n📌 FILES WITH UNKNOWN LABELS (verify manually):")
            for r in unknown_results:
                print(f"   • {r['file']} (predicted: {r['predicted'].upper()}, confidence: {r['confidence']:.1f}%)")
        
        print("="*70)

def main():
    """Main function"""
    print("\n" + "="*70)
    print("🎯 BABY CRY MODEL - TEST SET EVALUATION")
    print("="*70)
    
    # Ask for confirmation
    print("\nThis will test your model using the 13 files in the 'test set' folder.")
    print("These files were NOT used in training, so this is a valid test.")
    
    response = input("\nProceed with testing? (y/n): ").strip().lower()
    
    if response != 'y':
        print("❌ Test cancelled")
        return
    
    # Run test
    tester = TestSetEvaluator()
    tester.run_test()
    
    print("\n✅ Testing complete!")

if __name__ == "__main__":
    main()