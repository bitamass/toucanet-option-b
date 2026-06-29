# ============================================================
# Toucanet Option B — Step 3: Cluster embeddings → tokens
# Run this as many times as you want, it's fast!
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import hdbscan
import umap

if __name__ == '__main__':

    # --- Load embeddings from step 2 ---
    embeddings = np.load("embeddings.npy")
    print(f"Loaded embeddings shape: {embeddings.shape}")

    # --- Flatten to 2D ---
    n = embeddings.shape[0]
    X = embeddings.reshape(n, 1536)
    print(f"Embedding matrix: {X.shape}")

    # --- UMAP dimensionality reduction ---
    print("Running UMAP...")
    reducer = umap.UMAP(n_components=5, random_state=42)
    X_reduced = reducer.fit_transform(X)

    # --- HDBSCAN clustering ---
    print("Running HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
    labels = clusterer.fit_predict(X_reduced)

    print(f"\nCluster labels: {labels}")
    print(f"Clusters found: {len(set(labels)) - (1 if -1 in labels else 0)}")
    print(f"Noise points: {list(labels).count(-1)}")

    # --- Visualize ---
    reducer_2d = umap.UMAP(n_components=2, random_state=42)
    X_2d = reducer_2d.fit_transform(X)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='tab10', s=100)
    plt.colorbar(scatter, label='Cluster')
    plt.title("Bird call clusters — each dot is one call")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    for i, (x, y) in enumerate(X_2d):
        plt.annotate(f"call {i+1}", (x, y), textcoords="offset points", xytext=(5, 5))
    plt.tight_layout()
    plt.show()