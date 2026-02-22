#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
08_process_all_laugh_files.py
Process ALL baby laugh/giggle/talk files with correct filenames
"""

import sys
from pathlib import Path
import time

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent.absolute()))

try:
    import librosa
    import soundfile as sf
    import numpy as np
    import pandas as pd
    from datetime import datetime
    from config import (WAV_FOLDER, CLIPS_3S_FOLDER, CLIPS_5S_FOLDER, 
                       LABELING_FOLDER, TARGET_SR, TOP_DB)
    print("✅ Imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nPlease install required packages:")
    print("pip install librosa soundfile numpy pandas")
    sys.exit(1)

# ALL LAUGH/GIGGLE/TALK FILES (with correct filenames from your folder)
LAUGH_FILES = [
    "freesound_community-baby-boy-laughing-70651.mp3",
    "freesound_community-baby-giggle-85158.mp3",
    "freesound_community-baby-giggles-80104.mp3",
    "freesound_community-baby-giggling-65391.mp3",
    "freesound_community-baby-laugh-70665.mp3",
    "freesound_community-baby-laughing-64684.mp3",
    "freesound_community-baby-laughs-85161.mp3",
    "freesound_community-baby-talk-76380.mp3",
    "scratchonix-a-baby-laughing-417358.mp3",
    "scratchonix-a-baby-laughing-hysterically-417356.mp3",
    "u_xg7ssi08yr-baby-laugh1-355138.mp3"
]

# CRYING FILE (already processed)
CRYING_FILES = [
    "freesound_community-crying-baby-mergedwav-14538.mp3",
    "freesound_community-little-baby-sounds-30768.mp3"
]

def check_already_converted(mp3_filename):
    """Check if this MP3 has already been converted to WAV"""
    wav_filename = Path(mp3_filename).stem + '.wav'
    wav_path = WAV_FOLDER / wav_filename
    return wav_path.exists()

def convert_mp3_to_wav(mp3_file):
    """Convert single MP3 to WAV"""
    print(f"\n🔄 Converting: {mp3_file.name}")
    
    try:
        # Load MP3 file
        audio, sr = librosa.load(str(mp3_file), sr=TARGET_SR, mono=True)
        duration = len(audio) / sr
        
        # Create output filename
        output_filename = mp3_file.stem + '.wav'
        output_path = WAV_FOLDER / output_filename
        
        # Save as WAV
        sf.write(str(output_path), audio, sr)
        
        print(f"   ✅ Converted successfully")
        print(f"   📊 Duration: {duration:.2f}s")
        print(f"   📁 Saved to: {output_path}")
        
        return output_path, duration
        
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return None, None

def analyze_audio(wav_file):
    """Analyze single WAV file"""
    print(f"\n📊 Analyzing: {wav_file.name}")
    
    try:
        audio, sr = librosa.load(str(wav_file), sr=TARGET_SR)
        duration = len(audio) / sr
        
        # Detect non-silent segments
        non_silent_intervals = librosa.effects.split(audio, top_db=TOP_DB)
        
        if len(non_silent_intervals) > 0:
            segment_durations = [(end - start) / sr for start, end in non_silent_intervals]
            avg_seg = np.mean(segment_durations)
            
            print(f"   ✅ Analysis complete")
            print(f"   📊 Duration: {duration:.2f}s")
            print(f"   📊 Active segments: {len(non_silent_intervals)}")
            print(f"   📊 Avg segment: {avg_seg:.2f}s")
            
            # Recommend clip length
            if avg_seg < 2:
                recommended = 2
            elif avg_seg < 4:
                recommended = 3
            else:
                recommended = 5
            
            print(f"   💡 Recommended clip length: {recommended}s")
            
            return recommended
        else:
            print(f"   ⚠️  No active segments found")
            return 3
            
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return 3

def create_clips(wav_file, clip_duration=3):
    """Create clips from WAV file"""
    print(f"\n✂️ Creating {clip_duration}s clips from: {wav_file.name}")
    
    # Select output folder
    if clip_duration == 3:
        output_folder = CLIPS_3S_FOLDER
    elif clip_duration == 5:
        output_folder = CLIPS_5S_FOLDER
    else:
        output_folder = Path(f"clips_{clip_duration}seconds")
        output_folder.mkdir(exist_ok=True)
    
    try:
        # Load audio
        audio, sr = librosa.load(str(wav_file), sr=TARGET_SR)
        
        # Calculate clip parameters
        clip_samples = int(clip_duration * sr)
        
        # Create clips
        clips_created = []
        base_name = wav_file.stem
        
        for i, start in enumerate(range(0, len(audio) - clip_samples + 1, clip_samples)):
            clip = audio[start:start + clip_samples]
            
            # Skip silent clips
            energy = np.sqrt(np.mean(clip**2))
            if energy > 0.01:  # Energy threshold
                clip_filename = f"{base_name}_{clip_duration}s_clip_{i:04d}.wav"
                clip_path = output_folder / clip_filename
                sf.write(str(clip_path), clip, sr)
                clips_created.append(clip_path)
        
        print(f"   ✅ Created {len(clips_created)} clips")
        return clips_created
        
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return []

def safe_save_csv(df, filepath, max_retries=3):
    """Safely save CSV with retry logic if file is locked"""
    for attempt in range(max_retries):
        try:
            df.to_csv(filepath, index=False)
            return True
        except PermissionError:
            if attempt < max_retries - 1:
                print(f"   ⚠️  File {filepath.name} is locked. Retrying in 2 seconds... (Attempt {attempt + 2}/{max_retries})")
                time.sleep(2)
            else:
                print(f"   ❌ Could not save {filepath.name} after {max_retries} attempts.")
                print(f"      Please close the file if it's open in another program.")
                return False
    return False

def update_labeling_templates():
    """Update the labeling CSV with new clips"""
    print("\n📝 Updating labeling templates...")
    
    # Process each clips folder
    for clips_folder in [CLIPS_3S_FOLDER, CLIPS_5S_FOLDER]:
        if not clips_folder.exists():
            continue
            
        # Get all clip files
        clip_files = sorted(clips_folder.glob("*.wav"))
        if not clip_files:
            continue
        
        # Identify which folder this is (3s or 5s)
        folder_name = clips_folder.name
        
        print(f"\n📁 Processing folder: {folder_name}")
        print(f"   Total clips in folder: {len(clip_files)}")
        
        # Create or update labeling CSV
        labels_file = LABELING_FOLDER / f"labels_{folder_name}.csv"
        
        if labels_file.exists():
            try:
                # Load existing labels
                df_existing = pd.read_csv(labels_file)
                existing_filenames = set(df_existing['filename'])
                print(f"   Loaded existing file with {len(df_existing)} entries")
            except Exception as e:
                print(f"   ⚠️  Error reading existing file: {e}")
                print(f"   Creating new file instead")
                df_existing = pd.DataFrame()
                existing_filenames = set()
        else:
            df_existing = pd.DataFrame()
            existing_filenames = set()
            print(f"   Creating new labels file")
        
        # Find which clips are from the LAUGH files (to auto-label as not-cry)
        new_clips = []
        for clip_path in clip_files:
            # Check if this clip is from any of the laugh files
            for laugh_file in LAUGH_FILES:
                laugh_stem = Path(laugh_file).stem
                if laugh_stem in clip_path.stem:
                    if clip_path.name not in existing_filenames:
                        new_clips.append(clip_path)
                    break
        
        if not new_clips:
            print(f"   ℹ️  No new clips from laugh files to add")
            continue
        
        print(f"   Found {len(new_clips)} new clips from laugh files")
        
        # Create new entries for clips not already in CSV
        new_entries = []
        for clip_path in new_clips:
            try:
                duration = librosa.get_duration(path=str(clip_path))
            except:
                duration = 3 if '3' in folder_name else 5
            
            # Determine source file
            source = "unknown"
            for laugh_file in LAUGH_FILES:
                if Path(laugh_file).stem in clip_path.stem:
                    source = Path(laugh_file).stem
                    break
            
            new_entries.append({
                'filename': clip_path.name,
                'file_path': str(clip_path),
                'label': 'not-cry',  # Auto-set to not-cry for laughs/giggles
                'duration': round(duration, 2),
                'labeled': True,      # Auto-mark as labeled
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'notes': 'baby laugh/giggle/talk - auto-labeled as not-cry',
                'source_file': source
            })
        
        if new_entries:
            df_new = pd.DataFrame(new_entries)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            
            # Save with retry logic
            if safe_save_csv(df_combined, labels_file):
                print(f"   ✅ Added {len(new_entries)} new clips to {labels_file.name}")
                print(f"   📋 New clips added (sample):")
                for i, row in df_new.head(3).iterrows():
                    print(f"      • {row['filename']} - auto-labeled as not-cry")
                if len(df_new) > 3:
                    print(f"      ... and {len(df_new) - 3} more")
            else:
                print(f"   ❌ Failed to save {labels_file.name}")

def generate_summary():
    """Generate summary of all files"""
    print("\n" + "="*60)
    print("📊 DATASET SUMMARY")
    print("="*60)
    
    # Count files by type
    total_mp3 = len(list(Path(".").glob("*.mp3")))
    total_wav = len(list(WAV_FOLDER.glob("*.wav"))) if WAV_FOLDER.exists() else 0
    
    print(f"\n📁 Files Overview:")
    print(f"   • MP3 files: {total_mp3}")
    print(f"   • Converted WAV files: {total_wav}")
    
    # Count clips
    for folder in [CLIPS_3S_FOLDER, CLIPS_5S_FOLDER]:
        if folder.exists():
            clips = list(folder.glob("*.wav"))
            cry_clips = [c for c in clips if 'crying' in c.stem or 'cry' in c.stem]
            laugh_clips = [c for c in clips if 'laugh' in c.stem or 'giggle' in c.stem or 'talk' in c.stem]
            
            print(f"\n   {folder.name}:")
            print(f"      • Total clips: {len(clips)}")
            print(f"      • Cry clips: {len(cry_clips)}")
            print(f"      • Not-cry clips: {len(laugh_clips)}")
    
    # Check CSV files
    print(f"\n📋 Labeling Files:")
    for csv_file in LABELING_FOLDER.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file)
            cry_count = len(df[df['label'] == 'cry']) if 'label' in df.columns else 0
            not_cry_count = len(df[df['label'] == 'not-cry']) if 'label' in df.columns else 0
            print(f"   • {csv_file.name}: {len(df)} rows (cry: {cry_count}, not-cry: {not_cry_count})")
        except:
            print(f"   • {csv_file.name}: Unable to read")

def main():
    """Main function to process all laugh files"""
    print("="*60)
    print("🎵 PROCESSING ALL BABY LAUGH/GIGGLE/TALK FILES")
    print("="*60)
    
    # Check which laugh files exist
    existing_files = []
    missing_files = []
    
    print("\n📋 Checking laugh files...")
    for file in LAUGH_FILES:
        if Path(file).exists():
            status = "✅ Already converted" if check_already_converted(file) else "⏳ Ready to process"
            print(f"   ✓ {file} - {status}")
            existing_files.append(file)
        else:
            print(f"   ✗ {file} - NOT FOUND")
            missing_files.append(file)
    
    print(f"\n📊 Summary:")
    print(f"   • Laugh files found: {len(existing_files)}/{len(LAUGH_FILES)}")
    print(f"   • Crying files (already processed): {len(CRYING_FILES)}")
    
    print("\n" + "="*60)
    
    if not existing_files:
        print("❌ No laugh files to process!")
        return
    
    proceed = input("\nProceed with processing laugh files? (y/n): ").strip().lower()
    
    if proceed != 'y':
        print("❌ Operation cancelled")
        return
    
    # Process each existing laugh file
    for mp3_filename in existing_files:
        mp3_path = Path(mp3_filename)
        
        print("\n" + "="*60)
        print(f"PROCESSING: {mp3_filename}")
        print("="*60)
        
        # Step 1: Check if already converted
        if check_already_converted(mp3_filename):
            print(f"\n✅ {mp3_filename} already converted, using existing WAV")
            wav_path = WAV_FOLDER / (mp3_path.stem + '.wav')
        else:
            # Step 1: Convert to WAV
            wav_path, duration = convert_mp3_to_wav(mp3_path)
            if not wav_path:
                print(f"❌ Failed to convert {mp3_filename}, skipping...")
                continue
        
        # Step 2: Analyze audio
        recommended_duration = analyze_audio(wav_path)
        
        # Step 3: Create clips (both 3s and 5s)
        print(f"\n📋 Creating clips for {mp3_path.stem}:")
        
        # Create 3-second clips
        clips_3s = create_clips(wav_path, clip_duration=3)
        
        # Create 5-second clips
        clips_5s = create_clips(wav_path, clip_duration=5)
    
    # Step 4: Update labeling templates (auto-label as not-cry)
    update_labeling_templates()
    
    # Step 5: Generate summary
    generate_summary()
    
    print("\n" + "="*60)
    print("✅ ALL LAUGH FILES PROCESSED SUCCESSFULLY!")
    print("="*60)
    print("\n📋 Important Notes:")
    print("   1. All laugh/giggle/talk clips have been AUTO-LABELED as 'not-cry'")
    print("   2. The 'labeled' column is set to TRUE for these new clips")
    print("   3. Your dataset is now much more balanced!")
    print("\n📁 Output locations:")
    print(f"   • WAV files: {WAV_FOLDER}")
    print(f"   • 3s clips: {CLIPS_3S_FOLDER}")
    print(f"   • 5s clips: {CLIPS_5S_FOLDER}")
    print(f"   • Updated CSVs: {LABELING_FOLDER}")

if __name__ == "__main__":
    main()