# ============================================================
# Toucanet Option B — Step 16: AudioMoth 2 disturbance analysis
# 453 simulation clips including CHAINSAW data
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hdbscan
import umap
import librosa
import os
from birdnet import load_perch_v2

METADATA    = "cleaned_df (1).csv"
CLIP_FOLDER = "filtered_clips_2"
SAVE_EMB    = "am2_embeddings.npy"
SAVE_LABELS = "am2_labels.npy"

if __name__ == '__main__':

    print("Loading metadata...")
    df = pd.read_csv(METADATA)
    am2 = df[df['Recorder'] == 'Audio_Moth_2'].copy()
    am2['is_sim'] = am2['Sim Type'] != '[]'

    # Only use clips that exist in folder
    am2['exists'] = am2['clip_name'].apply(
        lambda x: os.path.exists(os.path.join(CLIP_FOLDER, x))
    )
    available = am2[am2['exists']]
    print(f"Total AM2 clips in metadata: {len(am2)}")
    print(f"Clips available in folder:   {len(available)}")
    print(f"Available simulation clips:  {available['is_sim'].sum()}")
    print(f"Available baseline clips:    {(~available['is_sim']).sum()}")

    sim_clips = available[available['is_sim']]['clip_name'].tolist()
    baseline_clips = available[~available['is_sim']]['clip_name'].sample(
        n=min(500, int((~available['is_sim']).sum())), random_state=42).tolist()

    clips_to_process = [(c, 1) for c in sim_clips] + \
                       [(c, 0) for c in baseline_clips]

    print(f"\nSimulation clips to process: {len(sim_clips)}")
    print(f"Baseline clips to process:   {len(baseline_clips)}")
    print(f"\nSimulation types available:")
    print(available[available['is_sim']]['Sim Type'].value_counts().to_string())
    print(f"\nTop species in simulation clips:")
    print(available[available['is_sim']]['species'].value_counts().head(5).to_string())

    if os.path.exists(SAVE_EMB):
        print("\nLoading saved embeddings...")
        all_embeddings = np.load(SAVE_EMB)
        all_labels = np.load(SAVE_LABELS)
        print(f"Loaded {len(all_embeddings)} segments")
    else:
        print("\nLoading Perch v2 model...")
        model = load_perch_v2(device="CPU")
        print("Model loaded! Collecting segments...")

        all_segments = []
        all_labels = []

        for i, (clip_name, label) in enumerate(clips_to_process):
            wav_path = os.path.join(CLIP_FOLDER, clip_name)
            if not os.path.exists(wav_path):
                continue
            if (i+1) % 50 == 0:
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
            except:
                continue

        print(f"Collected {len(all_segments)} segments")
        print("Embedding in one batch...")

        results = model.encode_arrays(iter(all_segments), n_producers=1, batch_size=32)
        results_list = list(results)
        all_embeddings = np.array([r['embedding'].flatten() for r in results_list])
        all_labels = np.array(all_labels)

        np.save(SAVE_EMB, all_embeddings)
        np.save(SAVE_LABELS, all_labels)
        print(f"Saved {len(all_embeddings)} embeddings!")

    # --- Cluster and compare ---
    X = all_embeddings.reshape(len(all_embeddings), 1536)
    print(f"\nTotal segments: {X.shape[0]}")
    print(f"Simulation: {all_labels.sum()}, Baseline: {(all_labels==0).sum()}")

    print("Running UMAP...")
    X_2d = umap.UMAP(n_components=2, random_state=42).fit_transform(X)

    print("Running HDBSCAN...")
    cluster_labels = hdbscan.HDBSCAN(min_cluster_size=5).fit_predict(X_2d)
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    print(f"Candidate clusters: {n_clusters}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].scatter(X_2d[all_labels==0, 0], X_2d[all_labels==0, 1],
                    c='steelblue', alpha=0.4, s=20, label='Baseline')
    axes[0].scatter(X_2d[all_labels==1, 0], X_2d[all_labels==1, 1],
                    c='red', alpha=0.6, s=20, label='Simulation')
    axes[0].set_title("AudioMoth 2 — Baseline vs Simulation")
    axes[0].legend()
    axes[0].set_xlabel("UMAP 1"); axes[0].set_ylabel("UMAP 2")

    scatter = axes[1].scatter(X_2d[:, 0], X_2d[:, 1],
                               c=cluster_labels, cmap='tab10', alpha=0.5, s=20)
    axes[1].set_title(f"AudioMoth 2 — {n_clusters} candidate clusters")
    axes[1].set_xlabel("UMAP 1"); axes[1].set_ylabel("UMAP 2")
    plt.colorbar(scatter, ax=axes[1])
    plt.tight_layout()
    plt.savefig("am2_disturbance_plot.png")
    plt.show()

    # --- Cluster frequency table ---
    print("\n=== Top cluster frequency shifts ===")
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
    print(out.head(15).to_string(index=False))