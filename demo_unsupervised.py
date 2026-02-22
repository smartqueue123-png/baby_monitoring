# demo_unsupervised.py - UPDATED VERSION
"""
Show what your unsupervised model learned
Works with simplified movement features
"""

import joblib
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

print("="*60)
print("🎯 UNSUPERVISED LEARNING DEMO")
print("="*60)

# Load your dataset
try:
    dataset = joblib.load('baby_movement_dataset.pkl')
    print(f"\n✅ Loaded {len(dataset)} videos from dataset")
except FileNotFoundError:
    print("❌ baby_movement_dataset.pkl not found!")
    print("   Please run youtube_baby_data_collector_SIMPLE.py first")
    exit()

# Load state names if they exist
try:
    state_names = joblib.load('baby_state_names.pkl')
    print("✅ Loaded custom state names")
except:
    # Default names based on your earlier analysis
    state_names = {
        0: "😊 Active & Awake",
        1: "😴 Deep Sleep", 
        2: "😢 Intense Crying"
    }
    print("⚠️ Using default state names")

# ============================================
# PREPARE FEATURES FOR CLUSTERING
# ============================================

print("\n🔄 Preparing features...")

X = []
video_names = []
video_categories = []

for item in dataset:
    # Use the simple numerical features
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
print(f"📊 Feature matrix: {X.shape}")

# ============================================
# TRAIN K-MEANS (same as before)
# ============================================

print("\n🔄 Training K-Means clustering...")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)

print("\n📊 CLUSTER DISTRIBUTION:")
for i in range(3):
    count = sum(1 for c in clusters if c == i)
    percentage = (count/len(clusters))*100
    print(f"   {state_names[i]}: {count} videos ({percentage:.1f}%)")

# ============================================
# SHOW VIDEOS IN EACH CLUSTER
# ============================================

print("\n" + "="*60)
print("📋 VIDEOS BY CLUSTER")
print("="*60)

for cluster_id in range(3):
    print(f"\n{state_names[cluster_id]}:")
    cluster_videos = [video_names[i] for i in range(len(video_names)) if clusters[i] == cluster_id]
    for video in cluster_videos:
        # Find category for this video
        idx = video_names.index(video)
        category = video_categories[idx]
        print(f"   • {video} (original: {category})")

# ============================================
# DEMO: PREDICT ON NEW DATA
# ============================================

print("\n" + "="*60)
print("🔍 TESTING ON NEW SAMPLES")
print("="*60)

# Create test samples with different movement levels
test_samples = [
    {
        'name': 'Very still baby (simulated sleep)',
        'features': [1.5, 0.5, 2.0, 1.8]  # low movement
    },
    {
        'name': 'Moderate movement (simulated awake)',
        'features': [4.0, 2.0, 10.0, 8.0]  # medium movement
    },
    {
        'name': 'High movement (simulated crying)',
        'features': [7.0, 4.5, 30.0, 25.0]  # high movement
    }
]

for test in test_samples:
    print(f"\n📝 Test: {test['name']}")
    
    # Predict cluster
    pred_cluster = kmeans.predict([test['features']])[0]
    predicted_state = state_names[pred_cluster]
    
    # Calculate distance to each cluster center
    distances = np.linalg.norm(kmeans.cluster_centers_ - test['features'], axis=1)
    confidence = 1 - (distances[pred_cluster] / np.sum(distances))
    
    print(f"   🤖 AI predicts: {predicted_state}")
    print(f"   📊 Confidence: {confidence*100:.1f}%")

# ============================================
# VISUALIZE CLUSTER CENTERS
# ============================================

print("\n" + "="*60)
print("📊 CLUSTER CHARACTERISTICS")
print("="*60)

print("\n{:<20} {:<15} {:<15} {:<15}".format(
    "State", "Mean Movement", "Std Movement", "Max Movement"))
print("-"*65)

for i in range(3):
    center = kmeans.cluster_centers_[i]
    print("{:<20} {:<15.2f} {:<15.2f} {:<15.2f}".format(
        state_names[i],
        center[0],
        center[1],
        center[2]
    ))

# ============================================
# SAVE THE MODEL FOR LATER USE
# ============================================

# Save the trained kmeans model
model_data = {
    'kmeans': kmeans,
    'state_names': state_names,
    'feature_columns': ['mean_movement', 'std_movement', 'max_movement', 'movement_range']
}

joblib.dump(model_data, 'baby_state_model.pkl')
print(f"\n💾 Saved model to baby_state_model.pkl")

print("\n" + "="*60)
print("✅ DEMO COMPLETE! You can now use baby_state_model.pkl in your app")
print("="*60)