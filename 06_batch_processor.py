#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
06_batch_processor.py
Run the entire pipeline automatically
"""

import sys
from pathlib import Path
import subprocess

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent.absolute()))

try:
    from config import PROJECT_ROOT
    print("✅ Imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def run_script(script_name):
    """Run a Python script and return success status"""
    print(f"\n{'='*60}")
    print(f"▶️  Running {script_name}")
    print(f"{'='*60}")
    
    script_path = PROJECT_ROOT / script_name
    
    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        return False
    
    try:
        result = subprocess.run([sys.executable, str(script_path)], 
                              capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running {script_name}: {e}")
        return False

def main():
    """Run the complete pipeline"""
    print("="*60)
    print("🎵 AUDIO PREPROCESSING PIPELINE")
    print("="*60)
    
    # Step 1: Convert MP3 to WAV
    if not run_script("01_convert_to_wav.py"):
        print("\n❌ Pipeline stopped at Step 1")
        return
    
    # Step 2: Analyze audio
    if not run_script("02_analyze_audio.py"):
        print("\n❌ Pipeline stopped at Step 2")
        return
    
    # Ask user which clipping method to use
    print("\n" + "="*60)
    print("Choose clipping method:")
    print("1. Fixed-length clips (3s or 5s)")
    print("2. Intelligent trimming")
    
    choice = input("\nEnter choice (1/2): ").strip()
    
    if choice == "2":
        # Step 3: Intelligent trim
        if not run_script("04_intelligent_trim.py"):
            print("\n❌ Pipeline stopped at Step 3")
            return
    else:
        # Step 3: Fixed clips
        if not run_script("03_create_clips.py"):
            print("\n❌ Pipeline stopped at Step 3")
            return
    
    # Step 4: Setup labeling
    if not run_script("05_labeling_setup.py"):
        print("\n❌ Pipeline stopped at Step 4")
        return
    
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETE!")
    print("="*60)
    print("\n📋 Next steps:")
    print("   1. Label your clips in the CSV file")
    print("   2. Train your model with the labeled data")

if __name__ == "__main__":
    main()