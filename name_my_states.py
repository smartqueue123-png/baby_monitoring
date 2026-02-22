# name_my_states.py - UPDATED for simple movement features
"""
Analyze each cluster to understand what the AI discovered
Works with the simplified movement features
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

print("="*60)
print("🔍 ANALYZING DISCOVERED BABY STATES")
print("="*60)

# Load your dataset
try:
    dataset = joblib.load('baby_movement_dataset.pkl')
    print(f"\n📊 Loaded {len(dataset)} videos from dataset")
except FileNotFoundError:
    print("❌ baby_movement_dataset.pkl not found!")
    print("   Please run youtube_baby_data_collector_SIMPLE.py first")
    exit()

if len(dataset) == 0:
    print("❌ Dataset is empty!")
    exit()

# ============================================
# PREPARE FEATURES FOR CLUSTERING
# ============================================

print("\n🔄 Preparing features for clustering...")

X = []
video_names = []
video_categories = []

for item in dataset:
    # For simple features, use the numerical values
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
print(f"📊 Feature matrix shape: {X.shape}")

# ============================================
# PERFORM CLUSTERING
# ============================================

# Determine number of clusters (use 3 for now)
n_clusters = min(3, len(dataset))
print(f"\n🔄 Running K-Means with {n_clusters} clusters...")

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)

# ============================================
# ANALYZE EACH CLUSTER
# ============================================

print("\n" + "="*60)
print("📊 CLUSTER ANALYSIS")
print("="*60)

cluster_summaries = []

for cluster_id in range(n_clusters):
    print(f"\n{'─'*50}")
    print(f"🔵 CLUSTER {cluster_id}")
    print(f"{'─'*50}")
    
    # Find videos in this cluster
    cluster_indices = [i for i in range(len(video_names)) if clusters[i] == cluster_id]
    
    print(f"\n📁 Videos in this cluster: {len(cluster_indices)}")
    
    # Show the videos
    print("\n🎬 Videos:")
    categories_in_cluster = []
    for idx in cluster_indices:
        print(f"   • {video_names[idx]}")
        print(f"     Category: {video_categories[idx]}")
        categories_in_cluster.append(video_categories[idx])
    
    # Calculate average features for this cluster
    cluster_features = X[cluster_indices]
    avg_movement = np.mean(cluster_features[:, 0])  # mean_movement
    avg_std = np.mean(cluster_features[:, 1])       # std_movement
    avg_max = np.mean(cluster_features[:, 2])       # max_movement
    
    print(f"\n📊 Movement Statistics:")
    print(f"   • Average movement: {avg_movement:.2f}")
    print(f"   • Movement variability: {avg_std:.2f}")
    print(f"   • Peak movement: {avg_max:.2f}")
    
    # Determine dominant category
    from collections import Counter
    cat_counts = Counter(categories_in_cluster)
    dominant_cat = cat_counts.most_common(1)[0][0]
    
    print(f"\n🎯 Dominant category: {dominant_cat}")
    
    # INTERPRETATION
    print("\n💡 What this likely means:")
    
    if avg_movement < 20:
        if dominant_cat == 'sleeping':
            suggestion = "😴 DEEP SLEEP - Low movement, consistent with sleeping"
        else:
            suggestion = "😴 LOW ACTIVITY - Baby is still/calm"
    elif avg_movement < 40:
        if dominant_cat == 'awake':
            suggestion = "😊 AWAKE & CALM - Moderate movement, baby is content"
        else:
            suggestion = "😊 MODERATE ACTIVITY - Baby is awake and moving"
    elif avg_movement < 60:
        suggestion = "😟 ACTIVE/FUSSING - Higher movement, baby may be unsettled"
    else:
        suggestion = "😢 HIGH ACTIVITY - Intense movement, possibly crying/distress"
    
    print(f"   → {suggestion}")
    
    cluster_summaries.append({
        'cluster': cluster_id,
        'size': len(cluster_indices),
        'avg_movement': avg_movement,
        'avg_std': avg_std,
        'dominant_category': dominant_cat,
        'interpretation': suggestion
    })

# ============================================
# VISUALIZATION
# ============================================

print("\n" + "="*60)
print("📈 CREATING VISUALIZATION")
print("="*60)

plt.figure(figsize=(12, 5))

# Plot 1: Movement by cluster
plt.subplot(1, 2, 1)
colors = ['blue', 'green', 'orange']
for i in range(n_clusters):
    cluster_data = X[clusters == i]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], 
               c=colors[i], label=f'Cluster {i}', s=100, alpha=0.7)
    
    # Add video numbers
    for j, idx in enumerate([idx for idx in range(len(video_names)) if clusters[idx] == i]):
        plt.annotate(str(j), (cluster_data[j, 0], cluster_data[j, 1]), 
                    fontsize=8, alpha=0.7)

plt.xlabel('Mean Movement')
plt.ylabel('Std Movement')
plt.title('Baby Movement Clusters')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Cluster summary
plt.subplot(1, 2, 2)
clusters_list = list(range(n_clusters))
sizes = [s['size'] for s in cluster_summaries]
movements = [s['avg_movement'] for s in cluster_summaries]

x = np.arange(len(clusters_list))
width = 0.35

plt.bar(x - width/2, sizes, width, label='Number of Videos', color='skyblue')
plt.bar(x + width/2, movements, width, label='Avg Movement', color='lightcoral')

plt.xlabel('Cluster')
plt.ylabel('Value')
plt.title('Cluster Summary')
plt.xticks(x, [f'Cluster {i}' for i in clusters_list])
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cluster_analysis.png')
print("\n📸 Visualization saved to cluster_analysis.png")
plt.show()

# ============================================
# FINAL SUMMARY TABLE
# ============================================

print("\n" + "="*60)
print("📊 FINAL CLUSTER SUMMARY")
print("="*60)

print("\n{:<10} {:<15} {:<15} {:<20} {:<30}".format(
    "Cluster", "Videos", "Avg Movement", "Dominant Category", "Interpretation"))
print("-"*90)

for summary in cluster_summaries:
    print("{:<10} {:<15} {:<15.2f} {:<20} {:<30}".format(
        f"Cluster {summary['cluster']}",
        summary['size'],
        summary['avg_movement'],
        summary['dominant_category'],
        summary['interpretation'][:30] + "..."
    ))

# ============================================
# NAME THE STATES (INTERACTIVE)
# ============================================

print("\n" + "="*60)
print("🎯 NAME YOUR DISCOVERED STATES")
print("="*60)
print("\nBased on the analysis above, let's name each cluster:")

state_names = {}
for summary in cluster_summaries:
    print(f"\n{'─'*40}")
    print(f"Cluster {summary['cluster']}:")
    print(f"   • {summary['interpretation']}")
    print(f"   • Avg movement: {summary['avg_movement']:.2f}")
    print(f"   • Dominant category: {summary['dominant_category']}")
    
    name = input(f"\nEnter a name for Cluster {summary['cluster']}: ").strip()
    if not name:
        # Default name based on interpretation
        if "SLEEP" in summary['interpretation'].upper():
            name = "Deep Sleep"
        elif "CALM" in summary['interpretation'].upper():
            name = "Awake & Calm"
        elif "FUSS" in summary['interpretation'].upper():
            name = "Fussing"
        else:
            name = "Active/Crying"
        print(f"   Using default: {name}")
    
    state_names[summary['cluster']] = name

# Save the names
joblib.dump(state_names, 'baby_state_names.pkl')
print(f"\n💾 State names saved to baby_state_names.pkl")

# Final mapping
print("\n" + "="*60)
print("✅ FINAL STATE MAPPING")
print("="*60)

for cluster_id, name in state_names.items():
    size = cluster_summaries[cluster_id]['size']
    print(f"\n🔵 Cluster {cluster_id}: {name}")
    print(f"   • {size} videos")
    print(f"   • Avg movement: {cluster_summaries[cluster_id]['avg_movement']:.2f}")

print("\n" + "="*60)
print("✅ DONE! You can now use these state names in your project")
print("="*60)