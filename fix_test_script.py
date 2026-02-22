# test_with_test_set_fixed.py
"""
Fixed version - handles numpy types for JSON serialization
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
    WAV_FOLDER = Path("converted_wav")
    TARGET_SR = 44100

# Helper function to convert numpy types to Python native types
def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj

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
        
        test_files = []
        for ext in ['*.mp3', '*.wav', '*.m4a', '*.ogg']:
            test_files.extend(self.test_folder.glob(ext))
        
        test_files = sorted(test_files)
        
        print(f"   Found {len(test_files)} test files:")
        for f in test_files:
            size = f.stat().st_size / (1024 * 1024)
            print(f"   • {f.name} ({size:.2f} MB)")
        
        return test_files
    
    def determine_expected_label(self, filename):
        """Determine expected label from filename"""
        filename = filename.lower()
        
        cry_words = ['cry', 'crying', 'moaning', 'newborn']
        not_cry_words = ['laugh', 'giggle', 'gurgle', 'lach', 'happy', 'squealing', 'making-sounds']
        
        if any(word in filename for word in cry_words):
            return 'cry'
        elif any(word in filename for word in not_cry_words):
            return 'not-cry'
        else:
            return 'unknown'
    
    def convert_to_wav(self, audio_path):
        """Convert any audio file to WAV for processing"""
        try:
            audio, sr = librosa.load(str(audio_path), sr=TARGET_SR, mono=True)
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
            audio, sr = librosa.load(audio_path, sr=TARGET_SR)
            features = []
            
            # MFCC (26 features)
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            features.extend(np.mean(mfcc, axis=1))
            features.extend(np.std(mfcc, axis=1))
            
            # Spectral features (5)
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)
            features.append(np.mean(spectral_centroids))
            features.append(np.std(spectral_centroids))
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
            features.append(np.mean(spectral_rolloff))
            features.append(np.std(spectral_rolloff))
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
            features.append(np.mean(spectral_bandwidth))
            
            # Zero crossing rate (2)
            zcr = librosa.feature.zero_crossing_rate(audio)
            features.append(np.mean(zcr))
            features.append(np.std(zcr))
            
            # RMS energy (2)
            rms = librosa.feature.rms(y=audio)
            features.append(np.mean(rms))
            features.append(np.std(rms))
            
            # Chroma (2)
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            features.append(np.mean(chroma))
            features.append(np.std(chroma))
            
            # Mel-spectrogram (2)
            mel = librosa.feature.melspectrogram(y=audio, sr=sr)
            features.append(np.mean(mel))
            features.append(np.std(mel))
            
            # Contrast (1)
            contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
            features.append(np.mean(contrast))
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            print(f"   ❌ Feature extraction error: {e}")
            return None
    
    def predict_file(self, audio_path):
        """Predict on a single file"""
        try:
            if audio_path.suffix.lower() != '.wav':
                wav_path = self.convert_to_wav(audio_path)
                if not wav_path:
                    return None
            else:
                wav_path = audio_path
            
            features = self.extract_features(str(wav_path))
            if features is None:
                return None
            
            if self.scaler is not None:
                features = self.scaler.transform(features)
            
            pred_num = self.model.predict(features)[0]
            
            if self.label_encoder is not None:
                prediction = self.label_encoder.inverse_transform([pred_num])[0]
            elif hasattr(self.model, 'classes_'):
                prediction = self.model.classes_[pred_num]
            else:
                prediction = 'cry' if pred_num == 0 else 'not-cry'
            
            confidence = 0
            all_probs = {}
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(features)[0]
                confidence = float(max(proba) * 100)  # Convert to Python float
                
                # Get class names
                if self.label_encoder is not None:
                    class_names = self.label_encoder.classes_
                elif hasattr(self.model, 'classes_'):
                    class_names = self.model.classes_
                else:
                    class_names = ['cry', 'not-cry']
                
                # Convert probabilities to Python floats
                all_probs = {str(name): float(prob * 100) for name, prob in zip(class_names, proba)}
            
            duration = librosa.get_duration(path=str(wav_path))
            
            return {
                'file': audio_path.name,
                'predicted': prediction,
                'confidence': confidence,
                'all_probabilities': all_probs,
                'duration': float(round(duration, 2))  # Convert to Python float
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
        
        if not self.load_model():
            return
        
        test_files = self.find_test_files()
        
        if not test_files:
            print("\n❌ No test files found!")
            return
        
        print("\n" + "="*70)
        print("🔬 RUNNING PREDICTIONS")
        print("="*70)
        
        for i, file_path in enumerate(test_files, 1):
            print(f"\n[{i}/{len(test_files)}] Testing: {file_path.name}")
            
            expected = self.determine_expected_label(file_path.name)
            print(f"   Expected: {expected.upper() if expected != 'unknown' else 'UNKNOWN'}")
            
            result = self.predict_file(file_path)
            
            if result:
                result['expected'] = expected
                result['correct'] = (result['predicted'].lower() == expected.lower()) if expected != 'unknown' else None
                self.test_results.append(result)
                
                print(f"   Predicted: {result['predicted'].upper()}")
                print(f"   Confidence: {result['confidence']:.1f}%")
                
                if expected != 'unknown':
                    print(f"   {'✅ CORRECT' if result['correct'] else '❌ INCORRECT'}")
                
                if result['all_probabilities']:
                    probs_str = ", ".join([f"{k}: {v:.1f}%" for k, v in result['all_probabilities'].items()])
                    print(f"   Probabilities: {probs_str}")
        
        self.generate_report()
    
    def generate_report(self):
        """Generate comprehensive test report"""
        print("\n" + "="*70)
        print("📊 TEST SET EVALUATION REPORT")
        print("="*70)
        
        if not self.test_results:
            print("\n❌ No test results to report")
            return
        
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
        
        print(f"\n📉 CONFIDENCE ANALYSIS:")
        avg_conf = float(np.mean([r['confidence'] for r in self.test_results]))
        print(f"   • Average confidence: {avg_conf:.2f}%")
        
        if known_results:
            correct_conf = float(np.mean([r['confidence'] for r in known_results if r['correct']]))
            incorrect_conf = float(np.mean([r['confidence'] for r in known_results if not r['correct']])) if any(not r['correct'] for r in known_results) else 0
            print(f"   • Avg confidence (correct): {correct_conf:.2f}%")
            print(f"   • Avg confidence (incorrect): {incorrect_conf:.2f}%")
        
        print(f"\n📋 DETAILED RESULTS:")
        print("-" * 90)
        print(f"{'File':<45} {'Expected':<10} {'Predicted':<10} {'Confidence':<10} {'Result':<8}")
        print("-" * 90)
        
        for r in sorted(self.test_results, key=lambda x: x['file']):
            result_mark = "✅" if r.get('correct') else "❌" if r.get('correct') is False else "⚠️"
            expected_display = r['expected'].upper() if r['expected'] != 'unknown' else '?'
            print(f"{r['file'][:43]:<45} {expected_display:<10} {r['predicted'].upper():<10} {r['confidence']:.1f}%     {result_mark}")
        
        # Convert results to serializable format before saving
        serializable_results = convert_to_serializable(self.test_results)
        serializable_categories = convert_to_serializable(categories if known_results else {})
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_files': total,
            'known_files': len(known_results),
            'accuracy': float(accuracy) if known_results else None,
            'average_confidence': float(avg_conf),
            'per_category': serializable_categories,
            'results': serializable_results
        }
        
        report_file = Path('test_set_report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n💾 Full report saved to: {report_file}")
        
        print(f"\n🎯 MODEL ASSESSMENT:")
        if known_results:
            if accuracy >= 90:
                print("   🏆 EXCELLENT! Model is production-ready")
            elif accuracy >= 85:
                print("   👍 VERY GOOD! Model performs well")
            elif accuracy >= 80:
                print("   ✅ GOOD! Model is acceptable")
            elif accuracy >= 70:
                print("   ⚠️ ACCEPTABLE! Consider more cry samples in training")
            else:
                print("   🔧 NEEDS IMPROVEMENT! Add more diverse training data")
        
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
    
    print("\nThis will test your model using the 13 files in the 'test set' folder.")
    print("These files were NOT used in training, so this is a valid test.")
    
    response = input("\nProceed with testing? (y/n): ").strip().lower()
    
    if response != 'y':
        print("❌ Test cancelled")
        return
    
    tester = TestSetEvaluator()
    tester.run_test()
    
    print("\n✅ Testing complete!")

if __name__ == "__main__":
    main()