# ============================================================
# Toucanet Option B — Step 20: Time-controlled disturbance analysis
# Compare baseline vs simulation at the SAME hours only
# AudioMoth 4
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hdbscan
import umap
import librosa
import os
from birdnet import load_perch_v2

CLIP_FOLDER = "filtered_clips_4"
METADATA    = "cleaned_df (1).csv"
SAVE_EMB    = "am4_time_controlled_emb.npy"
SAVE_LABELS = "am4_time_controlled_labels.npy"

if __name__ == '__main__':

    print("Loading metadata...")
    df = pd.read_csv(METADATA)
    am4 = df[df['Recorder'] == 'Audio_Moth_4'].copy()
    am4['is_sim'] = am4['Sim Type'] != '[]'
    am4['hour'] = am4['clip_name'].str[22:24].astype(int)

    # --- Find simulation hours ---
    sim_hours = am4[am4['is_sim']]['hour'].unique()
    print(f"Simulation hours: {sorted(sim_hours)}")

    # --- Get simulation clips ---
    sim_clips = am4[am4['is_sim']][['clip_name', 'hour']].copy()
    print(f"Simulation clips: {len(sim_clips)}")

    # --- Get baseline clips from SAME hours only ---
    baseline_clips = []
    for hour in sorted(sim_hours):
        hour_sim_count = len(am4[(am4['is_sim']) & (am4['hour'] == hour)])
        hour_baseline = am4[(~am4['is_sim']) & (am4['hour'] == hour)]
        # Sample same number of baseline as simulation for that hour
        n_sample = min(hour_sim_count * 5, len(hour_baseline))
        if n_sample > 0:
            sampled = hour_baseline.sample(n=n_sample, random_state=42)
            baseline_clips.append(sampled[['clip_name', 'hour']])
            print(f"Hour {hour}: {hour_sim_count} sim clips, {n_sample} baseline clips sampled")

    baseline_df = pd.concat(baseline_clips)
    print(f"\nTotal baseline clips (time-controlled): {len(baseline_df)}")
    print(f"Total simulation clips: {len(sim_clips)}")

    clips_to_process = [(c, 1) for c in sim_clips['clip_name'].tolist()] + \
                       [(c, 0) for c in baseline_df['clip_name'].tolist()]

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
                    c='steelblue', alpha=0.4, s=25, label='Baseline (same hour)')
    axes[0].scatter(X_2d[all_labels==1, 0], X_2d[all_labels==1, 1],
                    c='red', alpha=0.6, s=25, label='Simulation')
    axes[0].set_title("AudioMoth 4 — Time-Controlled\nBaseline vs Simulation (same hours only)")
    axes[0].legend()
    axes[0].set_xlabel("UMAP 1"); axes[0].set_ylabel("UMAP 2")

    scatter = axes[1].scatter(X_2d[:, 0], X_2d[:, 1],
                               c=cluster_labels, cmap='tab10', alpha=0.5, s=25)
    axes[1].set_title(f"Time-Controlled — {n_clusters} candidate clusters")
    axes[1].set_xlabel("UMAP 1"); axes[1].set_ylabel("UMAP 2")
    plt.colorbar(scatter, ax=axes[1])
    plt.tight_layout()
    plt.savefig("am4_time_controlled_plot.png")
    plt.show()

    # --- Cluster frequency table ---
    print("\n=== Cluster frequency: time-controlled baseline vs simulation ===")
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
    print("\nPositive = more common during simulation")
    print("Negative = more common during baseline")