# ============================================================
# Toucanet Option B — Step 13: Compare disturbance types
# With species identification per disturbance type
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hdbscan
import umap
import librosa
import os

CLIP_FOLDER = "filtered_clips_4"
METADATA    = "cleaned_df (1).csv"
SAVE_EMB    = "am4_fast_embeddings.npy"

if __name__ == '__main__':

    # --- Load embeddings ---
    print("Loading embeddings...")
    all_embeddings = np.load(SAVE_EMB)
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
    clips_to_process = [(c, 'simulation') for c in sim_clips] + \
                       [(c, 'baseline') for c in baseline_clips]

    sim_type_lookup = dict(zip(am4['clip_name'], am4['Sim Type']))

    segment_info = []
    for clip_name, condition in clips_to_process:
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
                    sim_type = sim_type_lookup.get(clip_name, '[]')
                    segment_info.append({
                        'clip_name': clip_name,
                        'condition': condition,
                        'sim_type': sim_type,
                    })
        except:
            continue

    print(f"Rebuilt {len(segment_info)} segment records")

    # --- Assign disturbance type ---
    def get_type(sim_type):
        if sim_type == '[]':
            return 'baseline'
        elif 'Gunshot' in sim_type:
            return 'gunshot'
        elif 'Vehicle' in sim_type:
            return 'human_vehicle'
        else:
            return 'human_trail'

    types = [get_type(s['sim_type']) for s in segment_info]
    type_array = np.array(types)
    clip_names = [s['clip_name'] for s in segment_info]

    # --- Merge with species metadata ---
    seg_df = pd.DataFrame({
        'clip_name': clip_names,
        'dist_type': types,
        'cluster': cluster_labels,
    })
    seg_df = seg_df.merge(
        am4[['clip_name', 'species', 'confidence']],
        on='clip_name', how='left'
    )

    # --- Species breakdown per disturbance type ---
    for dist_type in ['baseline', 'human_trail', 'human_vehicle', 'gunshot']:
        subset = seg_df[seg_df['dist_type'] == dist_type]
        print(f"\n{'='*50}")
        print(f"DISTURBANCE TYPE: {dist_type} ({len(subset)} segments)")
        print(f"{'='*50}")
        print(f"Top species:")
        print(subset['species'].value_counts().head(5).to_string())
        print(f"\nCluster 6 segments: {(subset['cluster']==6).sum()} ({100*(subset['cluster']==6).sum()/len(subset):.1f}%)")
        print(f"Cluster 6 species:")
        c6 = subset[subset['cluster'] == 6]
        print(c6['species'].value_counts().head(5).to_string())

    # --- Plot colored by disturbance type ---
    colors = {
        'baseline':      'steelblue',
        'human_trail':   'red',
        'human_vehicle': 'orange',
        'gunshot':       'purple',
    }
    plt.figure(figsize=(10, 7))
    for t, color in colors.items():
        mask = type_array == t
        if mask.sum() > 0:
            plt.scatter(X_2d[mask, 0], X_2d[mask, 1],
                       c=color, alpha=0.5, s=25,
                       label=f"{t} (n={mask.sum()})")
    plt.title("Disturbance Type Comparison\ncolor = when recorded, NOT species")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig("disturbance_types_plot.png")
    plt.show()