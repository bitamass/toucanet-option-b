# ============================================================
# Toucanet Option B — Step 6: Within-Site Stability Test
# All clips from AudioMoth 1 only
# Question: do the same embedding clusters appear across
# multiple clips from the same recorder?
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
    "Audio_Moth_1_20250317_100900.wav",
    "Audio_Moth_1_20250317_100903.wav",
    "Audio_Moth_1_20250317_100909.wav",
    "Audio_Moth_1_20250317_100912.wav",
    "Audio_Moth_1_20250317_100954.wav",
    "Audio_Moth_1_20250317_100957.wav",
    "Audio_Moth_1_20250317_101003.wav",
    "Audio_Moth_1_20250317_101006.wav",
    "Audio_Moth_1_20250317_101012.wav",
    "Audio_Moth_1_20250317_101015.wav",
]

SAVE_PATH = "within_site_embeddings.npy"
LABELS_PATH = "within_site_labels.npy"

if __name__ == '__main__':

    if os.path.exists(SAVE_PATH):
        print("Loading saved embeddings...")
        all_embeddings = np.load(SAVE_PATH)
        all_labels = np.load(LABELS_PATH)
        print(f"Loaded {len(all_embeddings)} segments")
    else:
        print("Loading Perch v2 model...")
        model = load_perch_v2(device="CPU")
        print("Model loaded!")

        all_embeddings = []
        all_labels = []

        for clip_idx, clip_path in enumerate(CLIPS):
            if not os.path.exists(clip_path):
                print(f"  SKIPPING {clip_path} — not found")
                continue

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
        np.save(SAVE_PATH, all_embeddings)
        np.save(LABELS_PATH, all_labels)
        print(f"\nSaved {len(all_embeddings)} embeddings")

    # --- Cluster ---
    X = all_embeddings.reshape(len(all_embeddings), 1536)
    print(f"\nTotal segments: {X.shape[0]}")

    print("Running UMAP...")
    reducer = umap.UMAP(n_components=2, random_state=42)
    X_2d = reducer.fit_transform(X)

    print("Running HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=3)
    cluster_labels = clusterer.fit_predict(X_2d)

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    print(f"Candidate acoustic clusters found: {n_clusters}")
    print(f"Noise points: {n_noise} ({100*n_noise/len(cluster_labels):.1f}%)")

    # --- Plot: color=cluster, each clip is labeled ---
    markers = ['o','s','^','D','P','*','X','v','<','>','h']
    plt.figure(figsize=(10, 7))
    for clip_idx in range(len(CLIPS)):
        mask = all_labels == clip_idx
        if mask.sum() == 0:
            continue
        clip_time = CLIPS[clip_idx].split("_")[3].replace(".wav","")
        plt.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=cluster_labels[mask],
            cmap='tab10', vmin=-1, vmax=8,
            marker=markers[clip_idx % len(markers)],
            s=120, label=f"clip {clip_time}",
            edgecolors='black', linewidths=0.5
        )

    plt.legend(fontsize=8, loc='upper left')
    plt.title(f"AudioMoth 1 only — {n_clusters} candidate acoustic clusters\n"
              f"shape=clip, color=cluster type")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.tight_layout()
    plt.show()