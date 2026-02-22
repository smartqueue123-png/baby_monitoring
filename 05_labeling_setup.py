#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
05_labeling_setup.py
Setup labeling system for audio clips
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent.absolute()))

try:
    import pandas as pd
    import librosa
    import json
    from datetime import datetime
    from config import CLIPS_3S_FOLDER, CLIPS_5S_FOLDER, SMART_CLIPS_FOLDER, LABELING_FOLDER
    print("✅ Imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nPlease install pandas: pip install pandas")
    sys.exit(1)

class AudioLabelingSystem:
    def __init__(self, clips_folder):
        """
        Initialize labeling system for a specific clips folder
        """
        self.clips_folder = Path(clips_folder)
        self.labels_file = LABELING_FOLDER / f"labels_{self.clips_folder.name}.csv"
        
        print(f"\n📁 Initializing labeling system for: {self.clips_folder.name}")
        
    def scan_clips(self):
        """Scan for all WAV clips in the folder"""
        clip_files = sorted(self.clips_folder.glob("*.wav"))
        print(f"   Found {len(clip_files)} clip files")
        return clip_files
    
    def create_labeling_template(self):
        """Create a CSV template for labeling"""
        print("\n📝 Creating labeling template...")
        
        # Get all clip files
        clip_files = self.scan_clips()
        
        if not clip_files:
            print("❌ No clip files found!")
            return None
        
        # Create dataframe
        data = []
        for clip_path in clip_files:
            try:
                duration = librosa.get_duration(filename=str(clip_path))
            except:
                duration = 0
            
            data.append({
                'filename': clip_path.name,
                'file_path': str(clip_path),
                'label': '',  # 'cry' or 'not-cry'
                'duration': round(duration, 2),
                'labeled': False,
                'timestamp': '',
                'notes': ''
            })
        
        df = pd.DataFrame(data)
        
        # Save to CSV
        df.to_csv(self.labels_file, index=False)
        print(f"✅ Created labeling template: {self.labels_file}")
        print(f"   Total clips to label: {len(df)}")
        
        # Show sample
        print("\n📋 First 5 files to label:")
        for i, row in df.head().iterrows():
            print(f"   {i+1}. {row['filename']} ({row['duration']}s)")
        
        return df
    
    def generate_stats(self):
        """Generate statistics about the dataset"""
        print("\n📊 Generating dataset statistics...")
        
        clip_files = self.scan_clips()
        
        stats = {
            'dataset_name': self.clips_folder.name,
            'created': datetime.now().isoformat(),
            'total_clips': len(clip_files),
            'sampling_rate': 44100,
            'file_list': [f.name for f in clip_files]
        }
        
        # Save stats
        stats_file = LABELING_FOLDER / f"stats_{self.clips_folder.name}.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"✅ Saved statistics to: {stats_file}")
        
        return stats

def choose_clips_folder():
    """Let user choose which clips folder to label"""
    folders = []
    
    if CLIPS_3S_FOLDER.exists() and any(CLIPS_3S_FOLDER.glob("*.wav")):
        folders.append(("1", "3-second clips", CLIPS_3S_FOLDER))
    
    if CLIPS_5S_FOLDER.exists() and any(CLIPS_5S_FOLDER.glob("*.wav")):
        folders.append(("2", "5-second clips", CLIPS_5S_FOLDER))
    
    if SMART_CLIPS_FOLDER.exists() and any(SMART_CLIPS_FOLDER.glob("*.wav")):
        folders.append(("3", "Smart clips", SMART_CLIPS_FOLDER))
    
    if not folders:
        print("\n❌ No clips found! Please run 03_create_clips.py or 04_intelligent_trim.py first.")
        return None
    
    print("\n📁 Available clips folders:")
    for num, name, _ in folders:
        print(f"   {num}. {name}")
    
    choice = input("\nSelect folder to label (1/2/3): ").strip()
    
    for num, name, folder in folders:
        if choice == num:
            return folder
    
    print("❌ Invalid choice")
    return None

if __name__ == "__main__":
    print("="*60)
    print("🏷️  LABELING SETUP TOOL")
    print("="*60)
    
    # Choose folder
    selected_folder = choose_clips_folder()
    
    if selected_folder:
        # Initialize labeling system
        label_system = AudioLabelingSystem(selected_folder)
        
        # Create template
        df = label_system.create_labeling_template()
        
        # Generate stats
        if df is not None:
            label_system.generate_stats()
            
            print("\n✅ Labeling setup complete!")
            print("\n📋 Next steps:")
            print("   1. Open the CSV file in Excel or Google Sheets")
            print(f"      {label_system.labels_file}")
            print("   2. Fill in the 'label' column with 'cry' or 'not-cry'")
            print("   3. Save the file when done")
    
    print("="*60)