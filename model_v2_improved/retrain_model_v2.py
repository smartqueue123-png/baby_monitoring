# retrain_model_v2.py - UPDATED VERSION for your folder structure
"""
Retrain script for model_v2_improved
Files are already in the current folder
"""

import sys
from pathlib import Path
import subprocess
import time

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent.absolute()))

print("="*70)
print("🚀 MODEL V2 IMPROVED - RETRAINING PIPELINE")
print("="*70)
print("\n📋 This script will:")
print("   1. Process the 3 challenging cry files in current folder")
print("   2. Retrain the model with improved data")
print("   3. Test the new model")
print("="*70)

# Files to process (already in current folder)
CHALLENGING_FILES = [
    "freesound_community-moaning-baby-for-freesound-33342.mp3",
    "nematoki-little-baby-crying-newborn-infant-cries-317180.mp3",
    "u_xg7ssi08yr-baby-cry2-355132.mp3"
]

def run_command(command, description):
    """Run a command and print output"""
    print(f"\n📌 {description}...")
    print("-" * 50)
    result = subprocess.run(command, shell=True, text=True)
    return result

def check_files_exist():
    """Check if all required files exist in current folder"""
    print("\n🔍 Checking for cry files in current folder...")
    
    all_exist = True
    for file in CHALLENGING_FILES:
        file_path = Path(file)
        if file_path.exists():
            size = file_path.stat().st_size / (1024 * 1024)
            print(f"   ✅ Found: {file} ({size:.2f} MB)")
        else:
            print(f"   ❌ Missing: {file}")
            all_exist = False
    
    return all_exist

def process_new_files():
    """Run the processing script on new files"""
    print("\n" + "="*70)
    print("🔄 STEP 1: Processing new audio files")
    print("="*70)
    
    # Run the processing script from parent folder
    return run_command("python ../08_process_all_laugh_files.py", "Converting and creating clips")

def retrain_model():
    """Retrain the model with improved data"""
    print("\n" + "="*70)
    print("🎯 STEP 2: Retraining model with improved data")
    print("="*70)
    
    # Run the training script from parent folder
    return run_command("python ../train_model_final.py", "Training model")

def test_new_model():
    """Test the newly trained model"""
    print("\n" + "="*70)
    print("🧪 STEP 3: Testing improved model")
    print("="*70)
    
    # Run the test script from parent folder
    return run_command("python ../test_with_test_set_fixed.py", "Testing model")

def show_instructions():
    """Show next steps"""
    print("\n" + "="*70)
    print("📋 NEXT STEPS")
    print("="*70)
    print("\nAfter processing completes:")
    print("   1. The 3 cry files will be converted to WAV")
    print("   2. Clips will be created in clips_5seconds/")
    print("   3. They'll be added to your CSV with label 'cry'")
    print("   4. Model will retrain with improved data")
    print("\nThen test again with:")
    print("   python ../test_with_test_set_fixed.py")

def main():
    """Main retraining pipeline"""
    print("\n" + "="*70)
    print("🏁 STARTING MODEL V2 IMPROVED RETRAINING")
    print("="*70)
    
    print(f"\n📁 Current folder: {Path.cwd()}")
    
    # Check if files exist
    if not check_files_exist():
        print("\n❌ Some cry files are missing!")
        print("   Please make sure these files are in the current folder:")
        for file in CHALLENGING_FILES:
            print(f"   • {file}")
        return
    
    show_instructions()
    
    response = input("\nContinue with retraining? (y/n): ").strip().lower()
    if response != 'y':
        print("❌ Retraining cancelled")
        return
    
    # Step 1: Process new files
    if process_new_files().returncode != 0:
        print("\n⚠️  Processing had issues, but continuing...")
    
    # Step 2: Retrain model
    if retrain_model().returncode != 0:
        print("\n⚠️  Training had issues, but continuing...")
    
    # Step 3: Test new model
    print("\n" + "="*70)
    print("🔍 Ready to test the improved model")
    print("="*70)
    
    response = input("\nRun tests on the new model? (y/n): ").strip().lower()
    if response == 'y':
        test_new_model()
    
    print("\n" + "="*70)
    print("✅ MODEL V2 IMPROVED RETRAINING COMPLETE!")
    print("="*70)
    print("\n📁 Check these locations for results:")
    print("   • clips_5seconds/ - New cry clips")
    print("   • labeling_data/labels_clips_5seconds.csv - Updated labels")
    print("   • baby_cry_model_complete.pkl - New model")
    print("\n🚀 Run test again: python ../test_with_test_set_fixed.py")

if __name__ == "__main__":
    main()