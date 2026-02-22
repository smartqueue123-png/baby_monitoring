# 10_check_files.py
"""
Check what files actually exist in your folder
"""

from pathlib import Path

print("="*60)
print("📁 CHECKING ALL AUDIO FILES")
print("="*60)

# Look for all audio files
audio_files = list(Path(".").glob("*.mp3")) + list(Path(".").glob("*.wav"))

print(f"\nFound {len(audio_files)} audio files:")
for f in sorted(audio_files):
    size = f.stat().st_size / (1024 * 1024)
    print(f"   • {f.name} ({size:.2f} MB)")

# Check for baby laugh files
print("\n🔍 Looking for baby laugh files:")
laugh_files = [f for f in audio_files if 'laugh' in f.name.lower()]
for f in laugh_files:
    print(f"   ✓ {f.name}")