# ============================================================
# Toucanet Option B — Step 21: Acoustic Entropy Analysis
# Measure soundscape chaos during baseline vs simulation
# Same hours only (time-controlled)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hdbscan
import umap
import librosa
import os
from scipy.stats import entropy
from sklearn.metrics import silhouette_score

CLIP_FOLDER = "filtered_clips_4"
METADATA    = "cleaned_df (1).csv"
SAVE_EMB    = "am4_time_controlled_emb.npy"
SAVE_LABELS = "am4_time_controlled_labels.npy"

if __name__ == '__main__':

    # --- Load time-controlled embeddings ---
    print("Loading time-controlled embeddings...")
    all_embeddings = np.load(SAVE_EMB)
    all_labels = np.load(SAVE_LABELS)
    X = all_embeddings.reshape(len(all_embeddings), 1536)

    print(f"Total segments: {len(all_embeddings)}")
    print(f"Simulation: {all_labels.sum()}, Baseline: {(all_labels==0).sum()}")

    # --- Run UMAP + HDBSCAN ---
    print("Running UMAP...")
    X_2d = umap.UMAP(n_components=2, random_state=42).fit_transform(X)

    print("Running HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
    cluster_labels = clusterer.fit_predict(X_2d)

    # --- Metric 1: Cluster entropy ---
    # How evenly distributed are segments across clusters?
    # High entropy = segments spread across many clusters = chaotic
    # Low entropy = segments concentrated in few clusters = organized

    def cluster_entropy(labels, condition_mask):
        c = cluster_labels[condition_mask]
        c = c[c != -1]  # remove noise
        if len(c) == 0:
            return 0
        counts = np.bincount(c)
        counts = counts[counts > 0]
        probs = counts / counts.sum()
        return entropy(probs)

    baseline_mask = all_labels == 0
    sim_mask = all_labels == 1

    base_entropy = cluster_entropy(cluster_labels, baseline_mask)
    sim_entropy = cluster_entropy(cluster_labels, sim_mask)

    print(f"\n=== Cluster Entropy ===")
    print(f"Baseline entropy:   {base_entropy:.4f}")
    print(f"Simulation entropy: {sim_entropy:.4f}")
    print(f"Difference:         {sim_entropy - base_entropy:.4f}")
    print(f"Interpretation: {'MORE chaotic during simulation' if sim_entropy > base_entropy else 'LESS chaotic during simulation'}")

    # --- Metric 2: Noise fraction ---
    # HDBSCAN labels -1 = noise (doesn't fit any cluster)
    # More noise during disturbance = more unusual sounds

    base_noise = (cluster_labels[baseline_mask] == -1).sum() / baseline_mask.sum()
    sim_noise = (cluster_labels[sim_mask] == -1).sum() / sim_mask.sum()

    print(f"\n=== Noise Fraction ===")
    print(f"Baseline noise:   {100*base_noise:.1f}%")
    print(f"Simulation noise: {100*sim_noise:.1f}%")
    print(f"Difference:       {100*(sim_noise-base_noise):.1f}%")

    # --- Metric 3: Average distance from cluster center ---
    # How far are segments from their cluster centers?
    # Larger distance = more scattered = more chaotic

    def avg_distance_from_center(X_2d, cluster_labels, mask):
        distances = []
        for c in set(cluster_labels[mask]):
            if c == -1:
                continue
            cluster_mask = (cluster_labels == c) & mask
            if cluster_mask.sum() < 2:
                continue
            center = X_2d[cluster_mask].mean(axis=0)
            dists = np.linalg.norm(X_2d[cluster_mask] - center, axis=1)
            distances.extend(dists.tolist())
        return np.mean(distances) if distances else 0

    base_scatter = avg_distance_from_center(X_2d, cluster_labels, baseline_mask)
    sim_scatter = avg_distance_from_center(X_2d, cluster_labels, sim_mask)

    print(f"\n=== Cluster Scatter (avg distance from center) ===")
    print(f"Baseline scatter:   {base_scatter:.4f}")
    print(f"Simulation scatter: {sim_scatter:.4f}")
    print(f"Difference:         {sim_scatter - base_scatter:.4f}")
    print(f"Interpretation: {'MORE scattered during simulation' if sim_scatter > base_scatter else 'LESS scattered during simulation'}")

    # --- Metric 4: Number of unique clusters used ---
    base_clusters = len(set(cluster_labels[baseline_mask]) - {-1})
    sim_clusters = len(set(cluster_labels[sim_mask]) - {-1})

    print(f"\n=== Unique Clusters Used ===")
    print(f"Baseline:   {base_clusters} clusters")
    print(f"Simulation: {sim_clusters} clusters")

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Entropy comparison
    axes[0].bar(['Baseline', 'Simulation'], [base_entropy, sim_entropy],
                color=['steelblue', 'red'], alpha=0.8)
    axes[0].set_title("Cluster Entropy\n(higher = more chaotic)")
    axes[0].set_ylabel("Entropy")

    # Plot 2: Noise fraction
    axes[1].bar(['Baseline', 'Simulation'], [100*base_noise, 100*sim_noise],
                color=['steelblue', 'red'], alpha=0.8)
    axes[1].set_title("Noise Fraction %\n(higher = more unusual sounds)")
    axes[1].set_ylabel("% noise segments")

    # Plot 3: Scatter
    axes[2].bar(['Baseline', 'Simulation'], [base_scatter, sim_scatter],
                color=['steelblue', 'red'], alpha=0.8)
    axes[2].set_title("Cluster Scatter\n(higher = more dispersed)")
    axes[2].set_ylabel("Avg distance from center")

    plt.suptitle("Acoustic Entropy Metrics — AudioMoth 4 (Time-Controlled)", fontsize=13)
    plt.tight_layout()
    plt.savefig("acoustic_entropy_plot.png")
    plt.show()

    # --- Summary ---
    print(f"\n=== SUMMARY ===")
    print(f"Entropy increase:  {sim_entropy - base_entropy:.4f}")
    print(f"Noise increase:    {100*(sim_noise - base_noise):.1f}%")
    print(f"Scatter increase:  {sim_scatter - base_scatter:.4f}")
    print(f"Extra clusters:    {sim_clusters - base_clusters}")