# 09_fix_csv_update.py
"""
Quick fix to add the missing 5-second clips to CSV
"""

import sys
from pathlib import Path
import time
import pandas as pd
from datetime import datetime

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent.absolute()))

try:
    import librosa
    from config import CLIPS_5S_FOLDER, LABELING_FOLDER
    print("✅ Imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def fix_5s_csv():
    """Add the missing 5-second clips to the CSV"""
    print("="*60)
    print("🔧 FIXING 5-SECOND LABELS CSV")
    print("="*60)
    
    # Check if the CSV is still open
    csv_path = LABELING_FOLDER / "labels_clips_5seconds.csv"
    
    if not csv_path.exists():
        print("❌ CSV file not found!")
        return
    
    print(f"\n📁 CSV file: {csv_path}")
    print("⚠️  MAKE SURE YOU CLOSED EXCEL BEFORE CONTINUING!")
    
    input("\nPress Enter after closing Excel...")
    
    try:
        # Load existing CSV
        df_existing = pd.read_csv(csv_path)
        print(f"\n📊 Loaded existing CSV with {len(df_existing)} entries")
        
        # Find the new laughing clips
        new_clips = []
        for clip_path in CLIPS_5S_FOLDER.glob("*.wav"):
            if "u_xg7ssi08yr-baby-laugh1-355138" in clip_path.stem:
                new_clips.append(clip_path)
        
        print(f"\n🔍 Found {len(new_clips)} new 5-second clips to add")
        
        # Check which ones are already in CSV
        existing_filenames = set(df_existing['filename'])
        clips_to_add = [c for c in new_clips if c.name not in existing_filenames]
        
        if not clips_to_add:
            print("\n✅ All clips already in CSV!")
            return
        
        print(f"\n📝 Adding {len(clips_to_add)} missing clips...")
        
        # Create new entries
        new_entries = []
        for clip_path in clips_to_add:
            try:
                duration = librosa.get_duration(path=str(clip_path))
            except:
                duration = 5.0
            
            new_entries.append({
                'filename': clip_path.name,
                'file_path': str(clip_path),
                'label': 'not-cry',
                'duration': round(duration, 2),
                'labeled': True,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'notes': 'baby laugh - auto-labeled as not-cry',
                'source_file': 'u_xg7ssi08yr-baby-laugh1-355138'
            })
        
        # Add to dataframe
        df_new = pd.DataFrame(new_entries)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        
        # Save
        df_combined.to_csv(csv_path, index=False)
        print(f"\n✅ Successfully added {len(new_entries)} clips to CSV!")
        
        # Show new balance
        cry_count = len(df_combined[df_combined['label'] == 'cry'])
        not_cry_count = len(df_combined[df_combined['label'] == 'not-cry'])
        
        print(f"\n📊 Updated 5-second dataset balance:")
        print(f"   • Cry clips: {cry_count}")
        print(f"   • Not-cry clips: {not_cry_count}")
        print(f"   • Total: {len(df_combined)}")
        
    except PermissionError:
        print("\n❌ Still can't access the file! Please:")
        print("   1. Close ALL Excel windows")
        print("   2. Check if file is open in another program")
        print("   3. Run this script again")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    fix_5s_csv()