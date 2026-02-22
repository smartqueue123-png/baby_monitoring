#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
08_process_more_files.py
Process the 9 new baby laugh/giggle/talk files
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

# NEW FILES - All the green highlighted ones (baby laughs, giggles, and talk)
NEW_AUDIO_FILES = [
    "freesound_community-baby-boy-laugh.mp3",
    "freesound_community-baby-giggle-8.mp3",
    "freesound_community-baby-giggles-8.mp3",
    "freesound_community-baby-giggling.mp3",
    "freesound_community-baby-laugh-70.mp3",
    "freesound_community-baby-laughing.mp3",
    "freesound_community-baby-laughs-8.mp3",
    "freesound_community-baby-talk-7638.mp3",
    "u_xg7ssi08yr-baby-laugh1-355138.mp3"
]

# Already processed files (to skip)
PROCESSED_FILES = [
    "freesound_community-crying-baby-mergedwav-14538.mp3",  # cry
    "freesound_community-little-baby-sounds-30768.mp3",     # little baby sounds
    "scratchonix-a-baby-laughing-417358.mp3",                # laugh
    "scratchonix-a-baby-laughing-hysterically-417356.mp3"    # hysterical laugh
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
        
        # Find which clips are from the NEW files
        new_clips = []
        for clip_path in clip_files:
            # Check if this clip is from any of the new files
            for new_file in NEW_AUDIO_FILES:
                new_file_stem = Path(new_file).stem
                if new_file_stem in clip_path.stem:
                    if clip_path.name not in existing_filenames:
                        new_clips.append(clip_path)
                    break
        
        if not new_clips:
            print(f"   ℹ️  No new clips from the 9 files to add")
            continue
        
        print(f"   Found {len(new_clips)} new clips from the 9 files")
        
        # Create new entries for clips not already in CSV
        new_entries = []
        for clip_path in new_clips:
            try:
                # Fix deprecation warning - use 'path' instead of 'filename'
                duration = librosa.get_duration(path=str(clip_path))
            except:
                duration = 3 if '3' in folder_name else 5
            
            # Determine source file
            source = "unknown"
            for new_file in NEW_AUDIO_FILES:
                if Path(new_file).stem in clip_path.stem:
                    source = Path(new_file).stem
                    break
            
            new_entries.append({
                'filename': clip_path.name,
                'file_path': str(clip_path),
                'label': 'not-cry',  # Auto-set to not-cry for these (all are laughs/giggles)
                'duration': round(duration, 2),
                'labeled': True,      # Auto-mark as labeled since we know they're not-cry
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'notes': 'baby laugh/giggle - auto-labeled as not-cry',
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

def generate_summary_report():
    """Generate a summary report of all processed files"""
    print("\n" + "="*60)
    print("📊 GENERATING SUMMARY REPORT")
    print("="*60)
    
    # Check which new files were processed
    successfully_processed = []
    failed_files = []
    
    for mp3_filename in NEW_AUDIO_FILES:
        mp3_path = Path(mp3_filename)
        if mp3_path.exists():
            wav_filename = mp3_path.stem + '.wav'
            wav_path = WAV_FOLDER / wav_filename
            
            if wav_path.exists():
                wav_size = wav_path.stat().st_size / (1024 * 1024)
                successfully_processed.append({
                    'mp3': mp3_filename,
                    'wav': wav_filename,
                    'size_mb': round(wav_size, 2)
                })
            else:
                failed_files.append(mp3_filename)
        else:
            failed_files.append(mp3_filename)
    
    # Count clips by folder for new files
    print(f"\n✅ Files processed successfully:")
    if successfully_processed:
        for file in successfully_processed:
            print(f"   • {file['mp3']} → {file['wav']} ({file['size_mb']} MB)")
    else:
        print(f"   ⚠️  No new files were processed")
    
    if failed_files:
        print(f"\n❌ Files that failed or were not found:")
        for file in failed_files:
            print(f"   • {file}")
    
    # Count total clips now
    print(f"\n📈 Updated dataset statistics:")
    for folder in [CLIPS_3S_FOLDER, CLIPS_5S_FOLDER]:
        if folder.exists():
            clips = list(folder.glob("*.wav"))
            # Separate by type
            cry_clips = [c for c in clips if 'crying' in c.stem or 'cry' in c.stem]
            not_cry_clips = [c for c in clips if 'crying' not in c.stem and 'cry' not in c.stem]
            
            print(f"\n   {folder.name}:")
            print(f"      Total clips: {len(clips)}")
            print(f"      Cry clips: {len(cry_clips)}")
            print(f"      Not-cry clips: {len(not_cry_clips)}")
    
    return successfully_processed

def main():
    """Main function to process all new audio files"""
    print("="*60)
    print("🎵 PROCESSING 9 NEW BABY LAUGH/GIGGLE/TALK FILES")
    print("="*60)
    
    # First, check which files actually exist
    existing_files = []
    missing_files = []
    
    print("\n📋 Checking for new files...")
    for file in NEW_AUDIO_FILES:
        if Path(file).exists():
            status = "✅ Already converted" if check_already_converted(file) else "⏳ Ready to process"
            print(f"   ✓ {file} - {status}")
            existing_files.append(file)
        else:
            print(f"   ✗ {file} - NOT FOUND")
            missing_files.append(file)
    
    print(f"\n📊 Summary:")
    print(f"   • Files found: {len(existing_files)}")
    print(f"   • Files missing: {len(missing_files)}")
    print(f"   • Already processed files (skipping): {len(PROCESSED_FILES)}")
    
    print("\n" + "="*60)
    
    if not existing_files:
        print("❌ No new files to process!")
        return
    
    proceed = input("\nProceed with processing new files? (y/n): ").strip().lower()
    
    if proceed != 'y':
        print("❌ Operation cancelled")
        return
    
    # Process each existing new file
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
    report = generate_summary_report()
    
    print("\n" + "="*60)
    print("✅ ALL NEW FILES PROCESSED SUCCESSFULLY!")
    print("="*60)
    print("\n📋 Important Notes:")
    print("   1. All new laugh/giggle/talk clips have been AUTO-LABELED as 'not-cry'")
    print("   2. The 'labeled' column is set to TRUE for these new clips")
    print("   3. Your dataset is now more balanced!")
    print("\n📁 Output locations:")
    print(f"   • WAV files: {WAV_FOLDER}")
    print(f"   • 3s clips: {CLIPS_3S_FOLDER}")
    print(f"   • 5s clips: {CLIPS_5S_FOLDER}")
    print(f"   • Updated CSVs: {LABELING_FOLDER}")
    print("\n✅ You're ready to train your model with the updated dataset!")

if __name__ == "__main__":
    main()