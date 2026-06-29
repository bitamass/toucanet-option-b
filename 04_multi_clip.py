# ============================================================
# Toucanet Option B — Step 4: Multi-clip cross-site clustering
# ============================================================

import librosa
import numpy as np
import matplotlib.pyplot as plt
import hdbscan
import umap
from birdnet import load_perch_v2
import os

CLIPS = [
    "Audio_Moth_1_20250317_093654 (2).wav",
    "Audio_Moth_1_20250317_094136.wav",
    "Audio_Moth_2_20250317_101833.wav",
    "Audio_Moth_3_20250319_080434.wav",
    "Audio_Moth_4_20250317_124445.wav",
    "Audio_Moth_5_20250317_115130.wav",
    "Audio_Moth_6_20250317_115638.wav"
]

EMBEDDINGS_SAVE = "multi_embeddings.npy"
LABELS_SAVE = "multi_labels.npy"

if __name__ == '__main__':

    if os.path.exists(EMBEDDINGS_SAVE):
        print("Loading saved embeddings...")
        all_embeddings = np.load(EMBEDDINGS_SAVE)
        all_labels = np.load(LABELS_SAVE)
        print(f"Loaded {len(all_embeddings)} segments")
    else:
        print("Loading Perch v2 model...")
        model = load_perch_v2(device="CPU")
        print("Model loaded!")

        all_embeddings = []
        all_labels = []

        for clip_idx, clip_path in enumerate(CLIPS):
            print(f"\nProcessing clip {clip_idx+1}/{len(CLIPS)}: {clip_path}")
            y, sr = librosa.load(clip_path, sr=None, mono=True)

            onsets = librosa.onset.onset_detect(
                y=y, sr=sr, units='time', delta=0.1, wait=15
            )
            print(f"  Found {len(onsets)} events")

            WINDOW = 0.2
            for onset in onsets:
                start = int(onset * sr)
                end = int((onset + WINDOW) * sr)
                if end < len(y):
                    seg = y[start:end].astype(np.float32)
                    result = model.encode_arrays((seg, sr))
                    all_embeddings.append(result.embeddings)
                    all_labels.append(clip_idx)

        all_embeddings = np.array(all_embeddings)
        all_labels = np.array(all_labels)
        np.save(EMBEDDINGS_SAVE, all_embeddings)
        np.save(LABELS_SAVE, all_labels)
        print(f"\nSaved {len(all_embeddings)} embeddings")

    # --- Cluster ---
    X = all_embeddings.reshape(len(all_embeddings), 1536)
    print(f"Total segments: {X.shape[0]}")

    print("Running UMAP + HDBSCAN...")
    reducer = umap.UMAP(n_components=2, random_state=42)
    X_2d = reducer.fit_transform(X)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=3)
    cluster_labels = clusterer.fit_predict(X_2d)

    print(f"Clusters found: {len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)}")
    print(f"Noise points: {list(cluster_labels).count(-1)}")

    # --- Plot ---
    markers = ['o', 's', '^', 'D', 'P', '*', 'X']
    plt.figure(figsize=(10, 7))
    for clip_idx, clip_path in enumerate(CLIPS):
        mask = all_labels == clip_idx
        recorder_name = clip_path.split("_")[2]  # gets recorder number
        plt.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=cluster_labels[mask],
            cmap='tab10', vmin=-1, vmax=8,
            marker=markers[clip_idx],
            s=120, label=f"AudioMoth {recorder_name}",
            edgecolors='black', linewidths=0.5
        )

    plt.legend()
    plt.title("Calls from 7 clips — shape=recorder, color=cluster type")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.tight_layout()
    plt.show()