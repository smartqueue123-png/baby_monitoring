# youtube_baby_data_collector_SIMPLE.py
"""
Simplified version - NO MediaPipe required!
Uses basic frame differencing for movement detection
"""

import os
import yt_dlp
import cv2
import numpy as np
from pathlib import Path
import joblib
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ============================================
# YOUR YOUTUBE LINKS
# ============================================

BABY_VIDEOS = {
    'sleeping': [
        'https://youtu.be/Q6W3rwoVDrM?si=yDp5lUKoJ_dWSID0',
    ],
    'awake': [
        'https://youtu.be/3UCK4XCrvoc?si=O_TE1vxKaa6WrUe8',
    ],
    'crying': [
        'https://youtu.be/3q4F-d4AZEw?si=u2gMXLYQzZUJzCZj',
        'https://youtu.be/hdCbuF8_1p8?si=1JQyEGBaZpGmKC1n',
        'https://youtu.be/UznbhXLZ1Io?si=duKBJirLiZaYCLHn',
        'https://youtu.be/_AUtYTaYMrE?si=zhVZr2DXM9xgYknr',
    ]
}

# ============================================
# SIMPLE MOVEMENT EXTRACTOR (NO MEDIAPIPE)
# ============================================

class SimpleMovementExtractor:
    """
    Extracts movement using frame differencing
    No MediaPipe required!
    """
    
    def extract_from_video(self, video_path):
        """Extract movement patterns using simple frame differences"""
        print(f"\n🔍 Processing: {Path(video_path).name}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"   ❌ Cannot open video")
            return None
        
        movements = []
        frame_count = 0
        prev_gray = None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate movement between frames
            if prev_gray is not None:
                # Frame difference = movement
                diff = cv2.absdiff(prev_gray, gray)
                movement_amount = np.mean(diff)
                movements.append(movement_amount)
            
            prev_gray = gray
            frame_count += 1
            
            # Progress indicator
            if frame_count % 300 == 0:
                print(f"   Processed {frame_count} frames...", end="\r")
        
        cap.release()
        print(f"\n   ✅ Processed {frame_count} frames")
        
        if len(movements) > 100:  # Need enough frames
            movements = np.array(movements)
            
            # Extract statistical features
            features = {
                'mean_movement': float(np.mean(movements)),
                'std_movement': float(np.std(movements)),
                'max_movement': float(np.max(movements)),
                'min_movement': float(np.min(movements)),
                'movement_range': float(np.ptp(movements)),
                'movement_quantiles': np.percentile(movements, [25, 50, 75]).tolist(),
                'movement_histogram': np.histogram(movements, bins=10)[0].tolist(),
                'total_frames': frame_count
            }
            return features
        return None

# ============================================
# DOWNLOAD VIDEOS
# ============================================

class YouTubeBabyCollector:
    def __init__(self):
        self.download_folder = Path("youtube_baby_videos")
        self.download_folder.mkdir(exist_ok=True)
        
    def download_videos(self):
        """Download all videos from the links above"""
        print("="*60)
        print("📥 DOWNLOADING BABY VIDEOS FROM YOUTUBE")
        print("="*60)
        
        ydl_opts = {
            'format': 'best[height<=480]',
            'outtmpl': str(self.download_folder / '%(title)s.%(ext)s'),
            'quiet': True,
            'ignoreerrors': True,
        }
        
        all_videos = []
        for category, links in BABY_VIDEOS.items():
            print(f"\n📁 Category: {category.upper()}")
            for url in links:
                try:
                    print(f"   Downloading: {url}")
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        info = ydl.extract_info(url, download=True)
                        if info:
                            filename = ydl.prepare_filename(info)
                            print(f"   ✅ Saved: {Path(filename).name}")
                            all_videos.append({
                                'path': filename,
                                'category': category,
                                'title': info.get('title', 'unknown')
                            })
                except Exception as e:
                    print(f"   ⚠️ Skipped - {e}")
        
        return all_videos

# ============================================
# CREATE DATASET
# ============================================

def create_dataset():
    """Main function to create dataset"""
    
    print("="*60)
    print("🎯 CREATING UNSUPERVISED LEARNING DATASET")
    print("="*60)
    
    # Download videos
    collector = YouTubeBabyCollector()
    videos = collector.download_videos()
    
    if not videos:
        print("❌ No videos downloaded!")
        return None
    
    print(f"\n✅ Downloaded {len(videos)} videos")
    
    # Extract features
    extractor = SimpleMovementExtractor()
    dataset = []
    
    for i, video in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] Processing {Path(video['path']).name}...")
        
        if Path(video['path']).exists():
            features = extractor.extract_from_video(video['path'])
            if features:
                dataset.append({
                    'video': Path(video['path']).name,
                    'category': video['category'],
                    **features
                })
                print(f"   ✅ Added to dataset")
            else:
                print(f"   ❌ Could not extract features")
    
    # Save dataset
    if dataset:
        joblib.dump(dataset, 'baby_movement_dataset.pkl')
        print(f"\n💾 Dataset saved: {len(dataset)} videos processed")
        print(f"   📁 File: baby_movement_dataset.pkl")
        
        # Show composition
        categories = {}
        for item in dataset:
            cat = item['category']
            categories[cat] = categories.get(cat, 0) + 1
        
        print("\n📊 Dataset composition:")
        for cat, count in categories.items():
            print(f"   {cat}: {count} videos")
    else:
        print("❌ No features extracted")
        return None
    
    return dataset

# ============================================
# RUN CLUSTERING
# ============================================

def run_clustering():
    """Cluster the videos"""
    
    print("\n" + "="*60)
    print("🔄 RUNNING UNSUPERVISED CLUSTERING")
    print("="*60)
    
    if not Path('baby_movement_dataset.pkl').exists():
        print("❌ baby_movement_dataset.pkl not found!")
        return
    
    dataset = joblib.load('baby_movement_dataset.pkl')
    print(f"✅ Loaded {len(dataset)} videos")
    
    # Prepare features
    X = []
    video_names = []
    video_categories = []
    
    for item in dataset:
        # Use numerical features
        features = [
            item['mean_movement'],
            item['std_movement'],
            item['max_movement'],
            item['movement_range']
        ]
        X.append(features)
        video_names.append(item['video'])
        video_categories.append(item['category'])
    
    X = np.array(X)
    
    # Determine number of clusters
    n_clusters = min(3, len(dataset))
    print(f"\n🔄 Running K-Means with {n_clusters} clusters...")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    
    # Show results
    print("\n📊 DISCOVERED CLUSTERS:")
    print("="*50)
    
    for i in range(n_clusters):
        cluster_indices = [j for j in range(len(video_names)) if clusters[j] == i]
        print(f"\n🔵 CLUSTER {i}: {len(cluster_indices)} videos")
        
        for idx in cluster_indices:
            print(f"   • {video_names[idx]}")
            print(f"     Category: {video_categories[idx]}")
    
    # Save clustered dataset
    for i, item in enumerate(dataset):
        item['cluster'] = int(clusters[i])
    
    joblib.dump(dataset, 'baby_movement_dataset_clustered.pkl')
    print(f"\n💾 Saved to baby_movement_dataset_clustered.pkl")
    
    # Visualize
    if len(X) >= 2:
        plt.figure(figsize=(10, 6))
        
        # Use first 2 features for visualization
        colors = ['blue', 'green', 'orange']
        for i in range(n_clusters):
            mask = clusters == i
            plt.scatter(X[mask, 0], X[mask, 1], 
                       c=colors[i % len(colors)], 
                       label=f'Cluster {i}', s=100, alpha=0.7)
            
            # Add labels
            for j, txt in enumerate(video_names):
                if clusters[j] == i:
                    plt.annotate(f'V{j}', (X[j, 0], X[j, 1]), fontsize=8)
        
        plt.xlabel('Mean Movement')
        plt.ylabel('Std Movement')
        plt.title('Baby Movement Patterns (Discovered by AI)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('baby_clusters.png')
        print("\n📸 Visualization saved to baby_clusters.png")
        plt.close()

# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    print("""
    ⚠️  SIMPLE VERSION - No MediaPipe required!
    """)
    
    input("\nPress Enter to start...")
    
    # Create dataset
    dataset = create_dataset()
    
    # Run clustering
    if dataset:
        run_clustering()
        print("\n✅ DONE! Check these files:")
        print("   • baby_movement_dataset.pkl")
        print("   • baby_movement_dataset_clustered.pkl")
        print("   • baby_clusters.png")