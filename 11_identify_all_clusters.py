# ============================================================
# Toucanet Option B — Step 11: Identify species in all clusters
# Focus on clusters 44 and 16 which also increase during simulation
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

CLUSTERS_OF_INTEREST = [29, 44, 16]

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
        n=500, random_state=42).tolist()
    clips_to_process = [(c, 1) for c in sim_clips] + \
                       [(c, 0) for c in baseline_clips]

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

    # --- Analyze each cluster of interest ---
    for cluster_id in CLUSTERS_OF_INTEREST:

        indices = np.where(cluster_labels == cluster_id)[0]
        print(f"\n{'='*50}")
        print(f"CLUSTER {cluster_id} — {len(indices)} segments")
        print(f"{'='*50}")

        clip_names = [segment_info[i]['clip_name'] for i in indices]
        labels = [segment_info[i]['label'] for i in indices]

        cluster_df = pd.DataFrame({
            'clip_name': clip_names,
            'is_sim': labels
        })

        merged = cluster_df.merge(
            am4[['clip_name', 'species', 'confidence', 'Sim Type']],
            on='clip_name', how='left'
        )

        sim_count = sum(labels)
        base_count = len(labels) - sim_count
        print(f"Simulation segments: {sim_count}")
        print(f"Baseline segments:   {base_count}")

        print(f"\nTop species (all segments):")
        print(merged['species'].value_counts().head(5).to_string())

        print(f"\nTop species (simulation only):")
        sim_only = merged[merged['is_sim'] == 1]
        if len(sim_only) > 0:
            print(sim_only['species'].value_counts().head(5).to_string())
        else:
            print("No simulation segments")

        print(f"\nTop species (baseline only):")
        base_only = merged[merged['is_sim'] == 0]
        if len(base_only) > 0:
            print(base_only['species'].value_counts().head(5).to_string())
        else:
            print("No baseline segments")

        print(f"\nSimulation types:")
        print(merged['Sim Type'].value_counts().to_string())
