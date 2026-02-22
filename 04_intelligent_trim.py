#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
04_intelligent_trim.py
Intelligently trim audio based on activity detection
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent.absolute()))

try:
    import librosa
    import soundfile as sf
    import numpy as np
    from config import WAV_FOLDER, SMART_CLIPS_FOLDER, TARGET_SR, MIN_CLIP_DURATION, MAX_CLIP_DURATION, TOP_DB
    print("✅ Imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def find_wav_file():
    """Find the first WAV file in the converted folder"""
    wav_files = list(WAV_FOLDER.glob("*.wav"))
    
    if not wav_files:
        print(f"\n❌ No WAV files found in: {WAV_FOLDER}")
        print("Please run 01_convert_to_wav.py first")
        return None
    
    return wav_files[0]

def intelligent_trim(min_duration=MIN_CLIP_DURATION, max_duration=MAX_CLIP_DURATION):
    """
    Intelligently trim audio by detecting active segments
    """
    print("="*60)
    print("STEP 3: Intelligent Trimming")
    print("="*60)
    
    # Find WAV file
    wav_file = find_wav_file()
    if not wav_file:
        return []
    
    print(f"\n📂 Source: {wav_file.name}")
    print(f"📁 Output folder: {SMART_CLIPS_FOLDER}")
    print(f"⏱️  Clip range: {min_duration}s - {max_duration}s")
    print(f"🔇 Silence threshold: {TOP_DB} dB")
    
    # Load audio
    print("\n📀 Loading audio...")
    audio, sr = librosa.load(str(wav_file), sr=TARGET_SR)
    total_duration = len(audio) / sr
    
    print(f"   Duration: {total_duration:.2f}s")
    
    # Detect non-silent segments
    print("\n🔍 Detecting active segments...")
    non_silent_intervals = librosa.effects.split(audio, top_db=TOP_DB)
    
    print(f"   Found {len(non_silent_intervals)} active segments")
    
    # Process each segment
    clips_created = []
    base_name = wav_file.stem
    
    print("\n✂️ Creating smart clips...")
    
    for idx, (start, end) in enumerate(non_silent_intervals):
        segment_duration = (end - start) / sr
        
        # Segment within desired range
        if min_duration <= segment_duration <= max_duration:
            clip = audio[start:end]
            clip_filename = f"{base_name}_smart_{idx:03d}_{segment_duration:.1f}s.wav"
            clip_path = SMART_CLIPS_FOLDER / clip_filename
            sf.write(str(clip_path), clip, sr)
            clips_created.append(clip_path)
            print(f"   ✓ Created: {clip_filename}")
        
        # Segment too short
        elif segment_duration < min_duration:
            print(f"   ⚠️  Segment {idx}: too short ({segment_duration:.1f}s) - skipping")
        
        # Segment too long - split
        else:
            print(f"   📏 Segment {idx}: long ({segment_duration:.1f}s) - splitting")
            
            num_clips = int(np.ceil(segment_duration / max_duration))
            clip_samples = int(max_duration * sr)
            
            for j in range(num_clips):
                clip_start = start + j * clip_samples
                clip_end = min(start + (j + 1) * clip_samples, end)
                
                if (clip_end - clip_start) / sr >= min_duration:
                    clip = audio[clip_start:clip_end]
                    actual_duration = (clip_end - clip_start) / sr
                    clip_filename = f"{base_name}_smart_{idx:03d}_{j:02d}_{actual_duration:.1f}s.wav"
                    clip_path = SMART_CLIPS_FOLDER / clip_filename
                    sf.write(str(clip_path), clip, sr)
                    clips_created.append(clip_path)
                    print(f"      └─ Part {j+1}: {actual_duration:.1f}s")
    
    # Summary
    print(f"\n✅ Intelligent trimming complete!")
    print(f"   Total clips created: {len(clips_created)}")
    
    if clips_created:
        total_clip_duration = 0
        for c in clips_created:
            try:
                total_clip_duration += librosa.get_duration(filename=str(c))
            except:
                total_clip_duration += max_duration
        
        print(f"\n📊 Summary:")
        print(f"   Total clip duration: {total_clip_duration:.2f}s")
        print(f"   Minutes of audio: {total_clip_duration/60:.1f} minutes")
        print(f"   Coverage: {total_clip_duration/total_duration*100:.1f}% of original")
    
    return clips_created

if __name__ == "__main__":
    print("="*60)
    print("🎯 INTELLIGENT TRIMMING TOOL")
    print("="*60)
    
    clips = intelligent_trim()
    
    if clips:
        print(f"\n✅ Created {len(clips)} smart clips")
        print("\n📋 Next: Run 05_labeling_setup.py")
    else:
        print("\n❌ No clips created. Try adjusting parameters in config.py")
    
    print("="*60)