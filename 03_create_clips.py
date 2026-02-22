#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
03_create_clips.py
Create fixed-length clips from audio file
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent.absolute()))

try:
    import librosa
    import soundfile as sf
    import numpy as np
    from config import WAV_FOLDER, CLIPS_3S_FOLDER, CLIPS_5S_FOLDER, TARGET_SR
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

def create_fixed_clips(clip_duration=3, overlap=0, energy_threshold=0.01):
    """
    Create fixed-length clips
    """
    print("="*60)
    print(f"STEP 3: Creating {clip_duration}-Second Clips")
    print("="*60)
    
    # Find WAV file
    wav_file = find_wav_file()
    if not wav_file:
        return []
    
    # Select output folder
    if clip_duration == 3:
        output_folder = CLIPS_3S_FOLDER
    elif clip_duration == 5:
        output_folder = CLIPS_5S_FOLDER
    else:
        output_folder = Path(f"clips_{clip_duration}seconds")
        output_folder.mkdir(exist_ok=True)
    
    print(f"\n📂 Source: {wav_file.name}")
    print(f"📁 Output folder: {output_folder}")
    print(f"⏱️  Clip duration: {clip_duration}s")
    
    # Load audio
    print("\n📀 Loading audio...")
    audio, sr = librosa.load(str(wav_file), sr=TARGET_SR)
    total_duration = len(audio) / sr
    
    # Calculate clip parameters
    clip_samples = int(clip_duration * sr)
    hop_samples = clip_samples  # No overlap
    
    # Calculate number of clips
    num_clips = (len(audio) - clip_samples) // hop_samples + 1
    
    print(f"\n📊 Audio duration: {total_duration:.2f}s")
    print(f"📊 Maximum possible clips: {num_clips}")
    
    # Create clips
    clips_created = []
    clips_skipped = 0
    base_name = wav_file.stem
    
    print("\n✂️ Creating clips...")
    
    for i, start in enumerate(range(0, len(audio) - clip_samples + 1, hop_samples)):
        # Extract clip
        clip = audio[start:start + clip_samples]
        
        # Calculate energy
        energy = np.sqrt(np.mean(clip**2))
        
        # Skip silent clips
        if energy < energy_threshold:
            clips_skipped += 1
            continue
        
        # Create filename
        clip_filename = f"{base_name}_{clip_duration}s_clip_{i:04d}.wav"
        clip_path = output_folder / clip_filename
        
        # Save clip
        sf.write(str(clip_path), clip, sr)
        clips_created.append(clip_path)
        
        # Progress indicator
        if (i + 1) % 20 == 0:
            print(f"   Created {i+1} clips...")
    
    print(f"\n✅ Clipping complete!")
    print(f"   Total clips created: {len(clips_created)}")
    print(f"   Silent clips skipped: {clips_skipped}")
    
    # Statistics
    if clips_created:
        total_clip_duration = len(clips_created) * clip_duration
        print(f"\n📊 Summary:")
        print(f"   Total clip duration: {total_clip_duration:.2f}s")
        print(f"   Minutes of audio: {total_clip_duration/60:.1f} minutes")
        print(f"   Coverage: {total_clip_duration/total_duration*100:.1f}% of original")
    
    return clips_created

if __name__ == "__main__":
    print("="*60)
    print("✂️ CLIP CREATION TOOL")
    print("="*60)
    
    print("\nChoose clip duration:")
    print("1. 3-second clips")
    print("2. 5-second clips")
    print("3. Custom duration")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == "1":
        clips = create_fixed_clips(clip_duration=3)
    elif choice == "2":
        clips = create_fixed_clips(clip_duration=5)
    elif choice == "3":
        try:
            duration = float(input("Enter clip duration in seconds: "))
            clips = create_fixed_clips(clip_duration=duration)
        except ValueError:
            print("❌ Invalid input. Using 3-second clips.")
            clips = create_fixed_clips(clip_duration=3)
    else:
        print("❌ Invalid choice. Using 3-second clips.")
        clips = create_fixed_clips(clip_duration=3)
    
    if clips:
        print(f"\n✅ Created {len(clips)} clips in {clips[0].parent}")
        print("\n📋 Next: Run 04_intelligent_trim.py or 05_labeling_setup.py")
    
    print("="*60)