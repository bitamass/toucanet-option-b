# ============================================================
# Toucanet Option B — Step 2: Embed segments with Perch v2
# Run this ONCE — saves embeddings to disk
# ============================================================

import numpy as np
from birdnet import load_perch_v2

if __name__ == '__main__':

    # --- Load segments from step 1 ---
    segments = np.load("segments.npy", allow_pickle=True)
    sr = int(np.load("sr.npy"))
    print(f"Loaded {len(segments)} segments at {sr} Hz")

    # --- Load Perch v2 ---
    print("Loading Perch v2 model...")
    model = load_perch_v2(device="CPU")
    print("Model loaded! Embedding segments...")

    # --- Embed each segment ---
    embeddings = []
    for i, segment in enumerate(segments):
        seg = segment.astype(np.float32)
        result = model.encode_arrays((seg, sr))
        embedding = result.embeddings
        embeddings.append(embedding)
        print(f"Segment {i+1}/{len(segments)}: done")

    embeddings = np.array(embeddings)
    np.save("embeddings.npy", embeddings)
    print(f"\nAll embeddings shape: {embeddings.shape}")
    print("Embeddings saved to embeddings.npy!")