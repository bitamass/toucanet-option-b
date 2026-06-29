# ============================================================
# Toucanet Option B — Step 9: Listen to Cluster 29
# Extract audio segments from cluster 29 and save as wav files
# ============================================================

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import os
import hdbscan
import umap

CLIP_FOLDER = "filtered_clips_4"
METADATA    = "cleaned_df (1).csv"
SAVE_EMB    = "am4_fast_embeddings.npy"
SAVE_LABELS = "am4_fast_labels.npy"
OUTPUT_DIR  = "cluster29_audio"

if __name__ == '__main__':

    # --- Load saved embeddings ---
    print("Loading embeddings...")
    all_embeddings = np.load(SAVE_EMB)
    all_labels = np.load(SAVE_LABELS)
    X = all_embeddings.reshape(len(all_embeddings), 1536)

    # --- Re-run clustering to get cluster labels ---
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
                        'sr': sr
                    })
        except:
            continue

    print(f"Rebuilt {len(segment_info)} segment records")

    # --- Find cluster 29 segments ---
    cluster29_mask = cluster_labels == 29
    cluster29_indices = np.where(cluster29_mask)[0]
    print(f"Cluster 29 has {len(cluster29_indices)} segments")
    print(f"  Simulation: {sum(1 for i in cluster29_indices if segment_info[i]['label']==1)}")
    print(f"  Baseline:   {sum(1 for i in cluster29_indices if segment_info[i]['label']==0)}")

    # --- Save 10 simulation and 10 baseline examples ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    sim_saved = 0
    base_saved = 0

    for idx in cluster29_indices:
        if sim_saved >= 10 and base_saved >= 10:
            break

        info = segment_info[idx]
        wav_path = os.path.join(CLIP_FOLDER, info['clip_name'])

        try:
            y, sr = librosa.load(wav_path, sr=None, mono=True)
            start = int(info['onset'] * sr)
            end = int((info['onset'] + 0.2) * sr)
            segment = y[start:end]

            if info['label'] == 1 and sim_saved < 10:
                fname = f"{OUTPUT_DIR}/sim_{sim_saved+1:02d}_{info['clip_name']}.wav"
                sf.write(fname, segment, sr)
                sim_saved += 1
                print(f"Saved simulation example {sim_saved}: {fname}")

            elif info['label'] == 0 and base_saved < 10:
                fname = f"{OUTPUT_DIR}/base_{base_saved+1:02d}_{info['clip_name']}.wav"
                sf.write(fname, segment, sr)
                base_saved += 1
                print(f"Saved baseline example {base_saved}: {fname}")
        except:
            continue

    print(f"\nDone! Saved to '{OUTPUT_DIR}' folder")
    print(f"Simulation examples: {sim_saved}")
    print(f"Baseline examples: {base_saved}")
    print("\nOpen the folder and listen to the wav files!")