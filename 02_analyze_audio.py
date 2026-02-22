#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
02_analyze_audio.py
Analyze audio file characteristics
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent.absolute()))

try:
    import librosa
    import numpy as np
    import matplotlib.pyplot as plt
    from config import WAV_FOLDER, TARGET_SR, TOP_DB
    print("✅ Imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nPlease install required packages:")
    print("pip install librosa numpy matplotlib")
    sys.exit(1)

def find_wav_file():
    """Find the first WAV file in the converted folder"""
    wav_files = list(WAV_FOLDER.glob("*.wav"))
    
    if not wav_files:
        print(f"\n❌ No WAV files found in: {WAV_FOLDER}")
        print("Please run 01_convert_to_wav.py first")
        return None
    
    print(f"\n✅ Found WAV file: {wav_files[0].name}")
    return wav_files[0]

def analyze_audio():
    """
    Comprehensive audio analysis
    """
    print("="*60)
    print("STEP 2: Audio Analysis")
    print("="*60)
    
    # Find WAV file
    wav_file = find_wav_file()
    if not wav_file:
        return None, None
    
    print(f"\n📊 Analyzing: {wav_file.name}")
    print("-"*50)
    
    # Load audio
    print("Loading audio...")
    audio, sr = librosa.load(str(wav_file), sr=TARGET_SR)
    duration = len(audio) / sr
    
    # Basic information
    print(f"\n📈 Basic Information:")
    print(f"   Duration: {duration:.2f} seconds")
    print(f"   Sample rate: {sr} Hz")
    print(f"   Total samples: {len(audio):,}")
    print(f"   File size: {wav_file.stat().st_size / (1024*1024):.2f} MB")
    
    # Amplitude statistics
    print(f"\n📊 Amplitude Statistics:")
    print(f"   Mean: {np.mean(audio):.6f}")
    print(f"   Std: {np.std(audio):.6f}")
    print(f"   Max: {np.max(audio):.6f}")
    print(f"   Min: {np.min(audio):.6f}")
    print(f"   RMS: {np.sqrt(np.mean(audio**2)):.6f}")
    
    # Detect non-silent segments
    print(f"\n🔍 Activity Detection (threshold: {TOP_DB} dB):")
    
    non_silent_intervals = librosa.effects.split(audio, top_db=TOP_DB)
    
    print(f"   Found {len(non_silent_intervals)} active segments")
    
    if len(non_silent_intervals) > 0:
        segment_durations = []
        
        for i, (start, end) in enumerate(non_silent_intervals[:5]):  # Show first 5
            seg_duration = (end - start) / sr
            segment_durations.append(seg_duration)
            print(f"   Segment {i+1}: {seg_duration:.2f}s")
        
        if len(non_silent_intervals) > 5:
            print(f"   ... and {len(non_silent_intervals) - 5} more segments")
        
        # Statistics
        avg_seg = np.mean(segment_durations)
        print(f"\n📊 Segment Statistics:")
        print(f"   Average length: {avg_seg:.2f}s")
        print(f"   Min length: {np.min(segment_durations):.2f}s")
        print(f"   Max length: {np.max(segment_durations):.2f}s")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if avg_seg < 2:
            print(f"   • Use 2s clips or intelligent trimming")
            recommended = 2
        elif avg_seg < 4:
            print(f"   • 3s clips would work well")
            recommended = 3
        else:
            print(f"   • 5s clips would capture more context")
            recommended = 5
        
        print(f"   • Recommended clip length: {recommended} seconds")
        
        return segment_durations, recommended
    else:
        print("❌ No active segments detected!")
        return None, None

if __name__ == "__main__":
    print("="*60)
    print("📊 AUDIO ANALYSIS TOOL")
    print("="*60)
    
    segment_stats, recommended = analyze_audio()
    
    if segment_stats:
        print("\n✅ Analysis complete!")
        print("📋 Next steps:")
        print("   1. Run 03_create_clips.py for fixed-length clips")
        print("   2. Or run 04_intelligent_trim.py for smart trimming")
    
    print("="*60)