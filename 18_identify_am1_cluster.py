import numpy as np
import pandas as pd
import librosa
import os
import hdbscan
import umap

CLIP_FOLDER = "filtered_clips_1"
METADATA    = "cleaned_df (1).csv"
SAVE_EMB    = "am1_embeddings.npy"
SAVE_LABELS = "am1_labels.npy"

if __name__ == '__main__':

    print("Loading embeddings...")
    all_embeddings = np.load(SAVE_EMB)
    all_labels = np.load(SAVE_LABELS)
    X = all_embeddings.reshape(len(all_embeddings), 1536)

    print("Running UMAP + HDBSCAN...")
    X_2d = umap.UMAP(n_components=2, random_state=42).fit_transform(X)
    cluster_labels = hdbscan.HDBSCAN(min_cluster_size=5).fit_predict(X_2d)

    print("Loading metadata...")
    df = pd.read_csv(METADATA)
    am1 = df[df['Recorder'] == 'Audio_Moth_1'].copy()
    am1['is_sim'] = am1['Sim Type'] != '[]'
    am1['exists'] = am1['clip_name'].apply(
        lambda x: os.path.exists(os.path.join(CLIP_FOLDER, x)))
    available = am1[am1['exists']]

    sim_clips = available[available['is_sim']]['clip_name'].tolist()
    baseline_clips = available[~available['is_sim']]['clip_name'].sample(
        n=min(500, int((~available['is_sim']).sum())), random_state=42).tolist()
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
                        'label': label,
                    })
        except:
            continue

    print(f"Rebuilt {len(segment_info)} segment records")

    # --- Identify species in cluster 1 ---
    for cluster_id in [1, 40]:
        indices = np.where(cluster_labels == cluster_id)[0]
        clip_names = [segment_info[i]['clip_name'] for i in indices]
        labels = [segment_info[i]['label'] for i in indices]

        cluster_df = pd.DataFrame({
            'clip_name': clip_names,
            'is_sim': labels
        })
        merged = cluster_df.merge(
            available[['clip_name', 'species', 'confidence', 'Sim Type']],
            on='clip_name', how='left'
        )

        print(f"\n{'='*50}")
        print(f"CLUSTER {cluster_id} — {len(indices)} segments")
        print(f"Simulation: {sum(labels)}, Baseline: {len(labels)-sum(labels)}")
        print(f"\nTop species:")
        print(merged['species'].value_counts().head(5).to_string())
        print(f"\nSimulation types:")
        print(merged['Sim Type'].value_counts().to_string())