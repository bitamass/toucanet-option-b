# ============================================================
# Toucanet Option B — Short timescale acoustic tokenization
# 01: Explore a bird-confirmed clip
# ============================================================

import librosa
import numpy as np
import matplotlib.pyplot as plt
import hdbscan
import umap
from birdnet import load_perch_v2
import os

if __name__ == '__main__':

    # --- Load the clip ---
    CLIP_PATH = "Audio_Moth_1_20250317_093654 (2).wav"
    y, sr = librosa.load(CLIP_PATH, sr=None, mono=True)
    print(f"Sample rate: {sr} Hz")
    print(f"Duration: {len(y)/sr:.2f} seconds")

    # --- Onset detection ---
    onsets = librosa.onset.onset_detect(
        y=y, sr=sr,
        units='time',
        delta=0.1,
        wait=15
    )
    print(f"Events found: {len(onsets)}")

    # --- Plot waveform with onsets ---
    plt.figure(figsize=(12, 3))
    librosa.display.waveshow(y, sr=sr, alpha=0.6)
    for onset in onsets:
        plt.axvline(x=onset, color='red', linewidth=1.5, alpha=0.8)
    plt.title(f"Bird clip — {len(onsets)} events found")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

    # --- Cut out each individual call ---
    WINDOW = 0.2
    segments = []
    for onset in onsets:
        start = int(onset * sr)
        end = int((onset + WINDOW) * sr)
        if end < len(y):
            segments.append(y[start:end])
    print(f"Cut out {len(segments)} segments")

    # --- Embed each segment with Perch v2 ---
    EMBEDDINGS_PATH = "embeddings.npy"

    if os.path.exists(EMBEDDINGS_PATH):
        # Load saved embeddings — skip recomputing
        print("Loading saved embeddings from disk...")
        embeddings = np.load(EMBEDDINGS_PATH)
        print(f"Loaded embeddings shape: {embeddings.shape}")
    else:
        # Compute embeddings for the first time
        print("Loading Perch v2 model...")
        model = load_perch_v2(device="CPU")
        print("Model loaded! Embedding segments...")

        embeddings = []
        for i, segment in enumerate(segments):
            seg = segment.astype(np.float32)
            result = model.encode_arrays((seg, sr))
            embedding = result.embeddings
            embeddings.append(embedding)
            print(f"Segment {i+1}: embedding shape = {np.array(embedding).shape}")

        embeddings = np.array(embeddings)
        np.save(EMBEDDINGS_PATH, embeddings)
        print(f"Embeddings saved to {EMBEDDINGS_PATH}!")

    print(f"All embeddings shape: {embeddings.shape}")

    # --- Step 4: Cluster the embeddings to find token types ---
    # Flatten embeddings from (12, 1, 1, 1536) to (12, 1536)
    X = embeddings.reshape(len(segments), 1536)
    print(f"Embedding matrix shape: {X.shape}")

    # Reduce dimensions with UMAP
    print("Running UMAP...")
    reducer = umap.UMAP(n_components=5, random_state=42)
    X_reduced = reducer.fit_transform(X)
    print(f"Reduced shape: {X_reduced.shape}")

    # Cluster with HDBSCAN
    print("Running HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
    labels = clusterer.fit_predict(X_reduced)

    print(f"\nCluster labels: {labels}")
    print(f"Number of clusters found: {len(set(labels)) - (1 if -1 in labels else 0)}")
    print(f"Noise points (label=-1): {list(labels).count(-1)}")

    # Visualize in 2D
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