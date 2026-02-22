#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
01_convert_to_wav.py
Convert MP3 to WAV format (44.1 kHz)
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent.absolute()))

try:
    import librosa
    import soundfile as sf
    import numpy as np
    from config import MP3_FILE_PATH, WAV_FOLDER, TARGET_SR
    print("✅ Imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nPlease install required packages:")
    print("pip install librosa soundfile numpy")
    print("\nAlso make sure 'config.py' exists in the current directory")
    sys.exit(1)

def convert_mp3_to_wav():
    """
    Convert MP3 to WAV with 44.1 kHz sample rate
    """
    print("="*60)
    print("STEP 1: MP3 to WAV Conversion")
    print("="*60)
    
    print(f"\n📁 Project root: {Path(__file__).parent.absolute()}")
    print(f"📁 Output folder: {WAV_FOLDER}")
    print(f"🎵 Input file: {MP3_FILE_PATH}")
    
    # Check if MP3 file exists
    if not MP3_FILE_PATH.exists():
        print(f"\n❌ Error: MP3 file not found!")
        print(f"   Expected: {MP3_FILE_PATH}")
        
        # List all MP3 files as help
        mp3_files = list(Path(__file__).parent.glob("*.mp3"))
        if mp3_files:
            print(f"\n📋 Found these MP3 files:")
            for f in mp3_files:
                print(f"   - {f.name}")
                print(f"     Try renaming this file to: {MP3_FILE_PATH.name}")
        return None
    
    try:
        # Load MP3 file
        print(f"\n📂 Loading audio file...")
        audio, sr = librosa.load(str(MP3_FILE_PATH), sr=TARGET_SR, mono=True)
        
        # Calculate duration
        duration = len(audio) / sr
        print(f"   ✅ Loaded successfully")
        print(f"   Sample rate: {sr} Hz")
        print(f"   Duration: {duration:.2f} seconds")
        print(f"   Samples: {len(audio):,}")
        
        # Create output filename
        output_filename = MP3_FILE_PATH.stem + '.wav'
        output_path = WAV_FOLDER / output_filename
        
        # Save as WAV
        print(f"\n💾 Saving to: {output_path}")
        sf.write(str(output_path), audio, sr)
        
        # Verify the saved file
        if output_path.exists():
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"   ✅ Conversion successful!")
            print(f"   File size: {file_size_mb:.2f} MB")
            
            # Basic audio statistics
            rms = np.sqrt(np.mean(audio**2))
            print(f"   Audio stats:")
            print(f"      RMS energy: {rms:.4f}")
            print(f"      Max amplitude: {np.max(np.abs(audio)):.4f}")
        else:
            print(f"   ❌ Error: Failed to save WAV file")
        
        return output_path
        
    except Exception as e:
        print(f"\n❌ Error during conversion: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("="*60)
    print("🎵 AUDIO CONVERSION TOOL")
    print("="*60)
    
    # Run conversion
    wav_file = convert_mp3_to_wav()
    
    if wav_file and wav_file.exists():
        print("\n✅ Step 1 complete!")
        print("📋 Next: Run 02_analyze_audio.py")
    else:
        print("\n❌ Conversion failed. Please check the error messages above.")
    
    print("="*60)