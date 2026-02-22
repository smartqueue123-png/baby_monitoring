# config.py
"""
Configuration file for audio preprocessing pipeline
"""
import os
from pathlib import Path

# Get the current directory
PROJECT_ROOT = Path(__file__).parent.absolute()

# File paths - UPDATE THIS WITH YOUR ACTUAL FILENAME
MP3_FILENAME = "freesound_community-crying-baby-mergedwav-14538.mp3"
MP3_FILE_PATH = PROJECT_ROOT / MP3_FILENAME

# Output directories
WAV_FOLDER = PROJECT_ROOT / "converted_wav"
CLIPS_3S_FOLDER = PROJECT_ROOT / "clips_3seconds"
CLIPS_5S_FOLDER = PROJECT_ROOT / "clips_5seconds"
SMART_CLIPS_FOLDER = PROJECT_ROOT / "smart_clips"
LABELING_FOLDER = PROJECT_ROOT / "labeling_data"

# Audio processing parameters
TARGET_SR = 44100  # Target sample rate (Hz)
CLIP_DURATION_3S = 3
CLIP_DURATION_5S = 5
MIN_CLIP_DURATION = 2
MAX_CLIP_DURATION = 5
TOP_DB = 30

# Create directories if they don't exist
print("="*60)
print("🔧 CONFIGURATION SETUP")
print("="*60)
print(f"📁 Project Root: {PROJECT_ROOT}")
print(f"🎵 MP3 File: {MP3_FILENAME}")
print(f"✅ MP3 Exists: {MP3_FILE_PATH.exists()}")

for folder in [WAV_FOLDER, CLIPS_3S_FOLDER, CLIPS_5S_FOLDER, SMART_CLIPS_FOLDER, LABELING_FOLDER]:
    folder.mkdir(exist_ok=True)
    print(f"📁 Created/Verified: {folder.name}")

if not MP3_FILE_PATH.exists():
    print("\n⚠️  WARNING: MP3 file not found!")
    print(f"   Expected: {MP3_FILE_PATH}")
    print("   Please make sure the MP3 file is in the project root folder")
    
    # List all MP3 files as help
    mp3_files = list(PROJECT_ROOT.glob("*.mp3"))
    if mp3_files:
        print("\n📋 Found these MP3 files:")
        for f in mp3_files:
            print(f"   - {f.name}")
else:
    file_size = MP3_FILE_PATH.stat().st_size / (1024 * 1024)
    print(f"📊 File size: {file_size:.2f} MB")

print("="*60)