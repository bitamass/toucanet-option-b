# ============================================================
# Toucanet Option B — Step 1: Load clip and segment
# ============================================================

import librosa
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    CLIP_PATH = "Audio_Moth_1_20250317_093654 (2).wav"

    # --- Load the clip ---
    y, sr = librosa.load(CLIP_PATH, sr=None, mono=True)
    print(f"Sample rate: {sr} Hz")
    print(f"Duration: {len(y)/sr:.2f} seconds")

    # --- Detect onsets ---
    onsets = librosa.onset.onset_detect(
        y=y, sr=sr,
        units='time',
        delta=0.1,
        wait=15
    )
    print(f"Events found: {len(onsets)}")

    # --- Save plot instead of showing it ---
    plt.figure(figsize=(12, 3))
    librosa.display.waveshow(y, sr=sr, alpha=0.6)
    for onset in onsets:
        plt.axvline(x=onset, color='red', linewidth=1.5, alpha=0.8)
    plt.title(f"Bird clip — {len(onsets)} events found")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig("onset_plot.png")
    plt.close()
    print("Plot saved to onset_plot.png")

    # --- Cut out segments ---
    WINDOW = 0.2
    segments = []
    for onset in onsets:
        start = int(onset * sr)
        end = int((onset + WINDOW) * sr)
        if end < len(y):
            segments.append(y[start:end])
    print(f"Cut out {len(segments)} segments")

    # --- Save segments for next step ---
    np.save("segments.npy", np.array(segments, dtype=object))
    np.save("sr.npy", np.array(sr))
    print("Segments saved to segments.npy!")