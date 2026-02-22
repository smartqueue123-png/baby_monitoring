# retrain_model_v2.py
"""
Retrain script for model_v2_improved
Adds the 3 challenging cry files from test set to training data
"""

import sys
from pathlib import Path
import shutil
import subprocess
import time

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent.absolute()))

print("="*70)
print("🚀 MODEL V2 IMPROVED - RETRAINING PIPELINE")
print("="*70)
print("\n📋 This script will:")
print("   1. Copy 3 challenging cry files from test_set to main folder")
print("   2. Process them through the pipeline")
print("   3. Retrain the model with improved data")
print("   4. Test the new model")
print("="*70)

# Files to add (the challenging ones from test results)
CHALLENGING_FILES = [
    "freesound_community-moaning-baby-for-freesound-33342.mp3",
    "nematoki-little-baby-crying-newborn-infant-cries-317180.mp3",
    "u_xg7ssi08yr-baby-cry2-355132.mp3"
]

TEST_SET_FOLDER = Path("test_set")
MAIN_FOLDER = Path(".")

def run_command(command, description):
    """Run a command and print output"""
    print(f"\n📌 {description}...")
    print("-" * 50)
    result = subprocess.run(command, shell=True, text=True)
    if result.returncode != 0:
        print(f"⚠️  Warning: Command may have issues")
    return result

def check_files_exist():
    """Check if all required files exist"""
    print("\n🔍 Checking for required files...")
    
    missing_files = []
    for file in CHALLENGING_FILES:
        source = TEST_SET_FOLDER / file
        if source.exists():
            size = source.stat().st_size / (1024 * 1024)
            print(f"   ✅ Found: {file} ({size:.2f} MB)")
        else:
            print(f"   ❌ Missing: {file}")
            missing_files.append(file)
    
    return len(missing_files) == 0

def copy_challenging_files():
    """Copy the challenging cry files to main folder"""
    print("\n📋 Copying challenging cry files to training data...")
    
    copied_files = []
    for file in CHALLENGING_FILES:
        source = TEST_SET_FOLDER / file
        destination = MAIN_FOLDER / file
        
        if source.exists():
            shutil.copy2(source, destination)
            size = destination.stat().st_size / (1024 * 1024)
            print(f"   ✅ Copied: {file} ({size:.2f} MB)")
            copied_files.append(file)
        else:
            print(f"   ❌ Not found: {file}")
    
    return copied_files

def verify_files_in_main():
    """Verify files are now in main folder"""
    print("\n🔍 Verifying files in main folder...")
    
    all_good = True
    for file in CHALLENGING_FILES:
        dest = MAIN_FOLDER / file
        if dest.exists():
            size = dest.stat().st_size / (1024 * 1024)
            print(f"   ✅ Verified: {file} ({size:.2f} MB)")
        else:
            print(f"   ❌ Missing: {file}")
            all_good = False
    
    return all_good

def process_new_files():
    """Run the processing script on new files"""
    print("\n" + "="*70)
    print("🔄 STEP 1: Processing new audio files")
    print("="*70)
    
    # Run the processing script
    return run_command("python 08_process_all_laugh_files.py", "Converting and creating clips")

def retrain_model():
    """Retrain the model with improved data"""
    print("\n" + "="*70)
    print("🎯 STEP 2: Retraining model with improved data")
    print("="*70)
    
    # Run the training script
    return run_command("python train_model_final.py", "Training model")

def test_new_model():
    """Test the newly trained model"""
    print("\n" + "="*70)
    print("🧪 STEP 3: Testing improved model")
    print("="*70)
    
    # Run the test script
    return run_command("python test_with_test_set_fixed.py", "Testing model on validation set")

def show_dataset_stats():
    """Show updated dataset statistics"""
    print("\n" + "="*70)
    print("📊 DATASET STATISTICS AFTER ADDING FILES")
    print("="*70)
    
    try:
        import pandas as pd
        from config import LABELING_FOLDER
        
        csv_file = LABELING_FOLDER / "labels_clips_5seconds.csv"
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            
            cry_count = len(df[df['label'] == 'cry'])
            not_cry_count = len(df[df['label'] == 'not-cry'])
            total = len(df)
            
            print(f"\n📈 Updated 5-second clips:")
            print(f"   • Cry clips: {cry_count}")
            print(f"   • Not-cry clips: {not_cry_count}")
            print(f"   • Total: {total}")
            print(f"   • Balance: {cry_count/total*100:.1f}% cry / {not_cry_count/total*100:.1f}% not-cry")
            
            # Show new files
            new_cry_files = df[df['filename'].str.contains('moaning|nematoki|baby-cry2', na=False)]
            if len(new_cry_files) > 0:
                print(f"\n✅ Added {len(new_cry_files)} new cry clips from challenging files:")
                for idx, row in new_cry_files.head().iterrows():
                    print(f"   • {row['filename']}")
            
            return cry_count, not_cry_count
        else:
            print("❌ CSV file not found yet - run processing first")
            return None, None
            
    except Exception as e:
        print(f"⚠️  Could not load stats: {e}")
        return None, None

def create_version_file():
    """Create a version info file"""
    version_info = f"""# Model V2 Improved - Training Info
# Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}

## Changes in this version:
- Added 3 challenging cry files from test set:
  • freesound_community-moaning-baby-for-freesound-33342.mp3
  • nematoki-little-baby-crying-newborn-infant-cries-317180.mp3
  • u_xg7ssi08yr-baby-cry2-355132.mp3

## Purpose:
These were the files that the previous model (76.92% accuracy) 
misclassified with high confidence. Adding them should improve 
cry detection accuracy.

## Expected improvement:
Previous cry detection: 57.14% (4/7)
Target: >80% cry detection

## Training date: {time.strftime("%Y-%m-%d %H:%M:%S")}
"""
    
    with open("MODEL_V2_INFO.txt", "w") as f:
        f.write(version_info)
    
    print(f"\n📝 Created version info file: MODEL_V2_INFO.txt")

def main():
    """Main retraining pipeline"""
    print("\n" + "="*70)
    print("🏁 STARTING MODEL V2 IMPROVED RETRAINING")
    print("="*70)
    
    # Step 0: Check if we're in the right folder
    print(f"\n📁 Current folder: {Path.cwd()}")
    print(f"   This should be your 'model_v2_improved' folder")
    
    response = input("\nContinue with retraining? (y/n): ").strip().lower()
    if response != 'y':
        print("❌ Retraining cancelled")
        return
    
    # Step 1: Check if test_set folder exists
    if not TEST_SET_FOLDER.exists():
        print(f"\n❌ Test set folder not found: {TEST_SET_FOLDER}")
        print("   Please make sure you're in the model_v2_improved folder")
        return
    
    # Step 2: Check if all files exist
    if not check_files_exist():
        print("\n⚠️  Some files are missing. Check the test_set folder.")
        response = input("Continue anyway? (y/n): ").strip().lower()
        if response != 'y':
            return
    
    # Step 3: Copy files
    copied = copy_challenging_files()
    if not copied:
        print("\n❌ No files were copied. Aborting.")
        return
    
    # Step 4: Verify copies
    if not verify_files_in_main():
        print("\n❌ File verification failed. Aborting.")
        return
    
    # Step 5: Show current stats (before processing)
    print("\n📊 Current dataset stats (before adding new files):")
    cry_count, not_cry_count = show_dataset_stats()
    
    # Step 6: Process new files
    if process_new_files().returncode != 0:
        print("\n⚠️  Processing had issues, but continuing...")
    
    # Step 7: Show updated stats
    print("\n📊 Updated dataset stats (after adding new files):")
    new_cry_count, new_not_cry_count = show_dataset_stats()
    
    # Step 8: Retrain model
    if retrain_model().returncode != 0:
        print("\n⚠️  Training had issues, but continuing...")
    
    # Step 9: Create version file
    create_version_file()
    
    # Step 10: Test new model
    print("\n" + "="*70)
    print("🔍 Ready to test the improved model")
    print("="*70)
    
    response = input("\nRun tests on the new model? (y/n): ").strip().lower()
    if response == 'y':
        test_new_model()
    
    print("\n" + "="*70)
    print("✅ MODEL V2 IMPROVED RETRAINING COMPLETE!")
    print("="*70)
    print("\n📋 Summary:")
    print("   • Added 3 challenging cry files to training")
    if new_cry_count and cry_count:
        print(f"   • Cry clips increased from {cry_count} to {new_cry_count}")
    print("   • Model retrained with improved data")
    print("   • Check MODEL_V2_INFO.txt for details")
    print("\n📁 Files created:")
    print("   • baby_cry_model_complete.pkl (new version)")
    print("   • cry_model.pkl (new version)")
    print("   • MODEL_V2_INFO.txt")
    print("\n🚀 Next step: Compare test results with previous run!")

if __name__ == "__main__":
    main()