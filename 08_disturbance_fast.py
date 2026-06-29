# ============================================================
# Toucanet Option B — Step 8: Disturbance Detection (Fixed)
# ============================================================

import librosa
import numpy as np
import matplotlib.pyplot as plt
import hdbscan
import umap
import pandas as pd
from birdnet import load_perch_v2
import os

CLIP_FOLDER = "filtered_clips_4"
METADATA    = "cleaned_df (1).csv"
SAVE_EMB    = "am4_fast_embeddings.npy"
SAVE_LABELS = "am4_fast_labels.npy"

if __name__ == '__main__':

    print("Loading metadata...")
    df = pd.read_csv(METADATA)
    am4 = df[df['Recorder'] == 'Audio_Moth_4'].copy()
    am4['is_sim'] = am4['Sim Type'] != '[]'

    sim_clips = am4[am4['is_sim']]['clip_name'].tolist()
    baseline_clips = am4[~am4['is_sim']]['clip_name'].sample(
        n=500, random_state=42).tolist()

    clips_to_process = [(c, 1) for c in sim_clips] + \
                       [(c, 0) for c in baseline_clips]

    print(f"Simulation clips: {len(sim_clips)}")
    print(f"Baseline clips: {len(baseline_clips)}")

    if os.path.exists(SAVE_EMB):
        print("Loading saved embeddings...")
        all_embeddings = np.load(SAVE_EMB)
        all_labels = np.load(SAVE_LABELS)
        print(f"Loaded {len(all_embeddings)} segments")
    else:
        print("Loading Perch v2 model...")
        model = load_perch_v2(device="CPU")
        print("Model loaded! Collecting segments...")

        all_segments = []
        all_labels = []

        for i, (clip_name, label) in enumerate(clips_to_process):
            wav_path = os.path.join(CLIP_FOLDER, clip_name)
            if not os.path.exists(wav_path):
                continue
            if (i+1) % 20 == 0:
                print(f"  Collecting {i+1}/{len(clips_to_process)}")
            try:
                y, sr = librosa.load(wav_path, sr=None, mono=True)
                onsets = librosa.onset.onset_detect(
                    y=y, sr=sr, units='time', delta=0.1, wait=15)
                WINDOW = 0.2
                for onset in onsets:
                    start = int(onset * sr)
                    end = int((onset + WINDOW) * sr)
                    if end < len(y):
                        all_segments.append((y[start:end].astype(np.float32), sr))
                        all_labels.append(label)
            except Exception as e:
                continue

        print(f"Collected {len(all_segments)} segments total")
        print("Embedding all segments in one batch...")

        results = model.encode_arrays(
            iter(all_segments),
            n_producers=1,
            batch_size=32
        )

        results_list = list(results)

        # Inspect first result to find field names
        first = results_list[0]
        print(f"First result dtype: {first.dtype}")
        print(f"First result dtype names: {first.dtype.names}")

        # Extract using the correct field name
        field_names = first.dtype.names
        print(f"Available fields: {field_names}")

        # Use the first field that looks like embeddings
        emb_field = None
        for name in field_names:
            arr = first[name]
            if hasattr(arr, 'shape') and arr.size > 100:
                emb_field = name
                print(f"Using field: {name} with shape {arr.shape}")
                break

        all_embeddings = np.array([r[emb_field].flatten() for r in results_list])
        all_labels = np.array(all_labels)

        np.save(SAVE_EMB, all_embeddings)
        np.save(SAVE_LABELS, all_labels)
        print(f"Saved {len(all_embeddings)} embeddings with shape {all_embeddings.shape}")

    # --- Cluster and compare ---
    print(f"\nEmbeddings shape: {all_embeddings.shape}")
    n_segments = all_embeddings.shape[0]
    emb_dim = all_embeddings.shape[1] if all_embeddings.ndim > 1 else all_embeddings.reshape(n_segments, -1).shape[1]
    X = all_embeddings.reshape(n_segments, -1)
    print(f"Total segments: {X.shape[0]}, Embedding dim: {X.shape[1]}")
    print(f"Simulation: {all_labels.sum()}, Baseline: {(all_labels==0).sum()}")

    print("Running UMAP...")
    X_2d = umap.UMAP(n_components=2, random_state=42).fit_transform(X)

    print("Running HDBSCAN...")
    cluster_labels = hdbscan.HDBSCAN(min_cluster_size=5).fit_predict(X_2d)
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    print(f"Candidate acoustic clusters: {n_clusters}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].scatter(X_2d[all_labels==0, 0], X_2d[all_labels==0, 1],
                    c='steelblue', alpha=0.4, s=30, label='Baseline')
    axes[0].scatter(X_2d[all_labels==1, 0], X_2d[all_labels==1, 1],
                    c='red', alpha=0.6, s=30, label='Simulation')
    axes[0].set_title("Baseline vs Simulation in embedding space")
    axes[0].legend()
    axes[0].set_xlabel("UMAP 1"); axes[0].set_ylabel("UMAP 2")

    scatter = axes[1].scatter(X_2d[:, 0], X_2d[:, 1],
                               c=cluster_labels, cmap='tab10', alpha=0.5, s=30)
    axes[1].set_title(f"{n_clusters} candidate acoustic clusters")
    axes[1].set_xlabel("UMAP 1"); axes[1].set_ylabel("UMAP 2")
    plt.colorbar(scatter, ax=axes[1])
    plt.tight_layout()
    plt.savefig("disturbance_plot.png")
    plt.show()

    # --- Cluster frequency table ---
    print("\n=== Cluster frequency: baseline vs simulation ===")
    results_out = []
    total_b = (all_labels==0).sum()
    total_s = all_labels.sum()
    for c in sorted(set(cluster_labels)):
        if c == -1: continue
        b = ((cluster_labels==c) & (all_labels==0)).sum()
        s = ((cluster_labels==c) & (all_labels==1)).sum()
        results_out.append({
            'cluster': c,
            'baseline_%': round(100*b/total_b, 2),
            'simulation_%': round(100*s/total_s, 2),
            'difference': round(100*s/total_s - 100*b/total_b, 2)
        })
    out = pd.DataFrame(results_out).sort_values('difference', ascending=False)
    print(out.to_string(index=False))
    print("\nPositive = more common during simulation")
    print("Negative = more common during baseline")