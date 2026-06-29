# ============================================================
# Toucanet Option B — Step 10: Identify species in Cluster 29
# Cross-reference cluster 29 segments with metadata
# ============================================================

import librosa
import numpy as np
import pandas as pd
import os
import hdbscan
import umap

CLIP_FOLDER = "filtered_clips_4"
METADATA    = "cleaned_df (1).csv"
SAVE_EMB    = "am4_fast_embeddings.npy"
SAVE_LABELS = "am4_fast_labels.npy"

if __name__ == '__main__':

    # --- Load saved embeddings ---
    print("Loading embeddings...")
    all_embeddings = np.load(SAVE_EMB)
    all_labels = np.load(SAVE_LABELS)
    X = all_embeddings.reshape(len(all_embeddings), 1536)

    # --- Re-run clustering ---
    print("Running UMAP + HDBSCAN...")
    X_2d = umap.UMAP(n_components=2, random_state=42).fit_transform(X)
    cluster_labels = hdbscan.HDBSCAN(min_cluster_size=5).fit_predict(X_2d)

    # --- Rebuild segment index ---
    print("Loading metadata...")
    df = pd.read_csv(METADATA)
    am4 = df[df['Recorder'] == 'Audio_Moth_4'].copy()
    am4['is_sim'] = am4['Sim Type'] != '[]'
    sim_clips = am4[am4['is_sim']]['clip_name'].tolist()
    baseline_clips = am4[~am4['is_sim']]['clip_name'].sample(
        n=200, random_state=42).tolist()
    clips_to_process = [(c, 1) for c in sim_clips] + \
                       [(c, 0) for c in baseline_clips]

    # Rebuild segment → clip mapping
    segment_info = []
    for clip_name, label in clips_to_process:
        wav_path = os.path.join(CLIP_FOLDER, clip_name)
        if not os.path.exists(wav_path):
            continue
        try:
            y, sr = librosa.load(wav_path, sr=None, mono=True)
            onsets = librosa.onset.onset_detect(
                y=y, sr=sr, units='time', delta=0.1, wait=15)
            WINDOW = 0.2
            for onset in onsets:
                start = int(onset * sr)
                end = int((onset + WINDOW) * sr)
                if end < len(y):
                    segment_info.append({
                        'clip_name': clip_name,
                        'onset': onset,
                        'label': label,
                    })
        except:
            continue

    print(f"Rebuilt {len(segment_info)} segment records")

    # --- Find cluster 29 segments ---
    cluster29_indices = np.where(cluster_labels == 29)[0]
    print(f"\nCluster 29 has {len(cluster29_indices)} segments")

    # --- Get clip names for cluster 29 ---
    cluster29_clips = [segment_info[i]['clip_name'] for i in cluster29_indices]
    cluster29_labels = [segment_info[i]['label'] for i in cluster29_indices]

    # --- Look up species in metadata ---
    cluster29_df = pd.DataFrame({
        'clip_name': cluster29_clips,
        'is_sim': cluster29_labels
    })

    # Merge with metadata to get species
    merged = cluster29_df.merge(
        am4[['clip_name', 'species', 'confidence', 'Sim Type']],
        on='clip_name', how='left'
    )

    print("\n=== Species in Cluster 29 (all segments) ===")
    print(merged['species'].value_counts().head(10).to_string())

    print("\n=== Species in Cluster 29 — SIMULATION only ===")
    sim_only = merged[merged['is_sim'] == 1]
    print(sim_only['species'].value_counts().head(10).to_string())

    print("\n=== Species in Cluster 29 — BASELINE only ===")
    base_only = merged[merged['is_sim'] == 0]
    print(base_only['species'].value_counts().head(10).to_string())

    print("\n=== Simulation types in Cluster 29 ===")
    print(merged['Sim Type'].value_counts().to_string())
