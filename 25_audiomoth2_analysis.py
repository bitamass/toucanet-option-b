"""
24_audiomoth2_analysis.py
--------------------------
AudioMoth 2 disturbance analysis — time-controlled from the start.

This script applies everything learned from AM4:
  1. Time-control first — only compare baseline clips from the SAME hours
     as simulations. Do NOT repeat the naive comparison mistake from AM4.
  2. Embed with Perch v2 in one batch (sims first, then baseline).
  3. Build the same 106-feature matrix as script 22:
       - Raw spectral features (MFCCs, centroid, entropy, etc.)
       - Hour-normalized z-scores (removes time-of-day at feature level)
       - Perch PCA top 20 components
  4. Train logistic regression with threshold tuning + clip-level voting.
  5. Compare results against AM4 to test cross-site generalization.

AM2 specifics:
  - 453 simulation clips: Human trail, Vehicle, CHAINSAW (new type)
  - Simulation dates: March 18 (93 clips) and March 19 (360 clips)
  - Recorder column in metadata: 'Audio_Moth_2'

Inputs:
  cleaned_df (1).csv
  filtered_clips_2/   (must be fully downloaded)

Outputs:
  am2_time_controlled_emb.npy
  am2_time_controlled_labels.npy
  am2_feature_matrix.csv
  am2_classifier_results.txt
  am2_disturbance_plot.png
  am2_precision_recall_curve.png
"""

import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from scipy.stats import entropy as scipy_entropy
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (classification_report,
                              precision_recall_fscore_support,
                              precision_recall_curve)

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = r"C:\Users\BitaMassoudi\KasmirWorldProjects\toucanet-option-b"
CLIPS_DIR  = os.path.join(BASE_DIR, "filtered_clips_2")
META_CSV   = os.path.join(BASE_DIR, "cleaned_df (1).csv")
SAVE_EMB   = os.path.join(BASE_DIR, "am2_time_controlled_emb.npy")
SAVE_LAB   = os.path.join(BASE_DIR, "am2_time_controlled_labels.npy")
OUT_CSV    = os.path.join(BASE_DIR, "am2_feature_matrix.csv")
OUT_REPORT = os.path.join(BASE_DIR, "am2_classifier_results.txt")
OUT_PLOT   = os.path.join(BASE_DIR, "am2_disturbance_plot.png")
OUT_CURVE  = os.path.join(BASE_DIR, "am2_precision_recall_curve.png")

SR          = None    # keep native sample rate
ONSET_DELTA = 0.1
ONSET_WAIT  = 15
WINDOW      = 0.2
RANDOM_SEED = 42
VOTE_FRAC   = 0.5

SPECTRAL_KEYS = (
    [f"mfcc_mean_{i}" for i in range(13)] +
    [f"mfcc_std_{i}"  for i in range(13)] +
    ["centroid_mean", "centroid_std",
     "rolloff_mean",  "rolloff_std",
     "bandwidth_mean","bandwidth_std",
     "zcr_mean",      "zcr_std",
     "rms_mean",      "rms_std",
     "silence_fraction",
     "spec_entropy_mean", "spec_entropy_std",
     "temporal_entropy",  "onset_count"]
)

if __name__ == '__main__':

    # ── 1. load metadata ──────────────────────────────────────────────────────
    print("Loading metadata...")
    df  = pd.read_csv(META_CSV)
    am2 = df[df['Recorder'] == 'Audio_Moth_2'].copy()
    am2['is_sim'] = am2['Sim Type'] != '[]'
    am2['hour']   = am2['clip_name'].str[22:24].astype(int)

    # Only keep clips that actually exist in the folder
    am2['exists'] = am2['clip_name'].apply(
        lambda x: os.path.exists(os.path.join(CLIPS_DIR, x)))
    am2 = am2[am2['exists']].copy()

    print(f"  AM2 clips available in folder: {len(am2)}")
    print(f"  Simulation clips available:    {am2['is_sim'].sum()}")
    print(f"  Baseline clips available:      {(~am2['is_sim']).sum()}")

    # Show simulation types
    print("\nSimulation types in AM2:")
    print(am2[am2['is_sim']]['Sim Type'].value_counts().to_string())

    # ── 2. time-controlled clip selection ─────────────────────────────────────
    # Critical: find which hours the simulations happened, then
    # only use baseline clips from those SAME hours.
    # This is the fix we learned from AM4 — apply it from the start here.
    print("\nApplying time-control from the start...")

    sim_clips = am2[am2['is_sim']][['clip_name', 'hour', 'Sim Type']].copy()
    sim_hours = sim_clips['hour'].unique()

    print(f"  Simulation hours: {sorted(sim_hours)}")
    print(f"  Total simulation clips: {len(sim_clips)}")

    # Check simulation types per hour so we can interpret results properly
    print("\nSimulation type breakdown by hour:")
    for hour in sorted(sim_hours):
        hour_sims = sim_clips[sim_clips['hour'] == hour]
        print(f"  Hour {hour}: {len(hour_sims)} clips")
        print(f"    Types: {hour_sims['Sim Type'].value_counts().to_dict()}")

    # Sample baseline clips from same hours (5x simulation count per hour)
    baseline_parts = []
    for hour in sorted(sim_hours):
        hour_sim_count = len(am2[(am2['is_sim']) & (am2['hour'] == hour)])
        hour_baseline  = am2[(~am2['is_sim']) & (am2['hour'] == hour)]
        n_sample       = min(hour_sim_count * 5, len(hour_baseline))
        if n_sample > 0:
            sampled = hour_baseline.sample(n=n_sample, random_state=RANDOM_SEED)
            baseline_parts.append(sampled[['clip_name', 'hour']])
            print(f"  Hour {hour}: {hour_sim_count} sim, {n_sample} baseline sampled "
                  f"(of {len(hour_baseline)} available)")
        else:
            print(f"  Hour {hour}: {hour_sim_count} sim, NO baseline available at this hour!")

    if not baseline_parts:
        raise ValueError("No baseline clips found at simulation hours. "
                         "Check that filtered_clips_2 is fully downloaded.")

    baseline_df = pd.concat(baseline_parts)
    print(f"\nTotal baseline clips (time-controlled): {len(baseline_df)}")
    print(f"Total simulation clips: {len(sim_clips)}")

    # Order: sims first (label=1), then baseline (label=0)
    # This matches the convention from script 20/22
    clips_ordered = (
        [(c, 1, h) for c, h in zip(sim_clips['clip_name'], sim_clips['hour'])] +
        [(c, 0, h) for c, h in zip(baseline_df['clip_name'], baseline_df['hour'])]
    )

    # ── 3. embed with Perch v2 ────────────────────────────────────────────────
    if os.path.exists(SAVE_EMB):
        print("\nLoading saved AM2 embeddings...")
        embeddings = np.load(SAVE_EMB)
        labels     = np.load(SAVE_LAB)
        print(f"  Loaded {len(embeddings)} segments  |  "
              f"{int(labels.sum())} sim  |  {int((labels==0).sum())} baseline")
    else:
        print("\nLoading Perch v2 model...")
        from birdnet import load_perch_v2
        model = load_perch_v2(device="CPU")
        print("Model loaded. Collecting segments...")

        all_segments = []
        all_labels   = []

        for i, (clip_name, label, hour) in enumerate(clips_ordered):
            wav_path = os.path.join(CLIPS_DIR, clip_name)
            if not os.path.exists(wav_path):
                continue
            if (i + 1) % 100 == 0:
                print(f"  Collecting {i+1}/{len(clips_ordered)}")
            try:
                y, sr = librosa.load(wav_path, sr=None, mono=True)
                onsets = librosa.onset.onset_detect(
                    y=y, sr=sr, units='time', delta=ONSET_DELTA, wait=ONSET_WAIT)
                for onset in onsets:
                    start = int(onset * sr)
                    end   = int((onset + WINDOW) * sr)
                    if end < len(y):
                        all_segments.append((y[start:end].astype(np.float32), sr))
                        all_labels.append(label)
            except Exception as e:
                print(f"  WARNING: could not load {clip_name}: {e}")
                continue

        print(f"Collected {len(all_segments)} segments. Embedding in one batch...")
        results        = model.encode_arrays(
            iter(all_segments), n_producers=1, batch_size=32)
        results_list   = list(results)
        embeddings     = np.array([r['embedding'].flatten() for r in results_list])
        labels         = np.array(all_labels)

        np.save(SAVE_EMB, embeddings)
        np.save(SAVE_LAB, labels)
        print(f"Saved {len(embeddings)} embeddings to disk.")

    # ── 4. UMAP + HDBSCAN (exploratory — to mirror AM4 workflow) ─────────────
    print("\nRunning UMAP + HDBSCAN for exploratory visualization...")
    import umap
    import hdbscan

    X = embeddings.reshape(len(embeddings), 1536)
    print(f"  Total segments: {X.shape[0]}")

    X_2d = umap.UMAP(n_components=2, random_state=RANDOM_SEED).fit_transform(X)

    cluster_ids = hdbscan.HDBSCAN(min_cluster_size=5).fit_predict(X_2d)
    n_clusters  = len(set(cluster_ids)) - (1 if -1 in cluster_ids else 0)
    print(f"  Candidate clusters found: {n_clusters}")

    # Count clusters per group (same-space comparison, not separate HDBSCAN)
    base_clusters = set(cluster_ids[labels == 0]) - {-1}
    sim_clusters  = set(cluster_ids[labels == 1]) - {-1}
    print(f"  Clusters in baseline: {len(base_clusters)}")
    print(f"  Clusters in simulation: {len(sim_clusters)}")
    print(f"  Drop: {len(base_clusters) - len(sim_clusters)} fewer during disturbance")

    # Cluster frequency table — top shifts
    print("\nTop cluster frequency shifts (AM2 time-controlled):")
    total_b = (labels == 0).sum()
    total_s = labels.sum()
    freq_rows = []
    for c in sorted(set(cluster_ids)):
        if c == -1:
            continue
        b = ((cluster_ids == c) & (labels == 0)).sum()
        s = ((cluster_ids == c) & (labels == 1)).sum()
        freq_rows.append({
            'cluster':      c,
            'baseline_%':   round(100 * b / total_b, 2),
            'simulation_%': round(100 * s / total_s, 2),
            'difference':   round(100 * s / total_s - 100 * b / total_b, 2)
        })
    freq_df = pd.DataFrame(freq_rows).sort_values('difference', ascending=False)
    print(freq_df.head(10).to_string(index=False))
    print("\nNegative = more common during baseline:")
    print(freq_df.tail(5).to_string(index=False))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#2C5F2D")
    for ax in axes:
        ax.set_facecolor("#2C5F2D")

    axes[0].scatter(X_2d[labels==0, 0], X_2d[labels==0, 1],
                    c='steelblue', alpha=0.4, s=15, label='Baseline (same hour)')
    axes[0].scatter(X_2d[labels==1, 0], X_2d[labels==1, 1],
                    c='red', alpha=0.6, s=15, label='Simulation')
    axes[0].set_title("AudioMoth 2 — Time-Controlled\nBaseline vs Simulation",
                      color='white')
    axes[0].legend(facecolor='#1F451F', labelcolor='white')
    axes[0].tick_params(colors='white')
    axes[0].set_xlabel("UMAP 1", color='white')
    axes[0].set_ylabel("UMAP 2", color='white')

    sc = axes[1].scatter(X_2d[:, 0], X_2d[:, 1],
                          c=cluster_ids, cmap='tab10', alpha=0.5, s=15)
    axes[1].set_title(f"AudioMoth 2 — {n_clusters} candidate clusters",
                      color='white')
    axes[1].tick_params(colors='white')
    axes[1].set_xlabel("UMAP 1", color='white')
    axes[1].set_ylabel("UMAP 2", color='white')
    plt.colorbar(sc, ax=axes[1])

    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=150, bbox_inches='tight', facecolor='#2C5F2D')
    plt.show()
    print(f"Plot saved -> {OUT_PLOT}")

    # ── 5. rebuild clip list for feature extraction ───────────────────────────
    # We need the clip list to match the saved embedding order exactly.
    # Same convention as script 22: sims first (label=1), baseline second (label=0).
    print("\nExtracting spectral features from wav files...")

    rows    = []
    missing = 0

    for clip_idx, (clip_name, clip_label, clip_hour) in enumerate(clips_ordered):
        if (clip_idx + 1) % 200 == 0:
            print(f"  {clip_idx+1}/{len(clips_ordered)} clips processed...")

        wav_path = os.path.join(CLIPS_DIR, clip_name)
        if not os.path.exists(wav_path):
            missing += 1
            continue

        try:
            y, sr = librosa.load(wav_path, sr=SR, mono=True)
        except Exception:
            missing += 1
            continue

        onsets       = librosa.onset.onset_detect(
            y=y, sr=sr, units='time', delta=ONSET_DELTA, wait=ONSET_WAIT)
        valid_onsets = [o for o in onsets if int((o + WINDOW) * sr) < len(y)]
        n_segments   = len(valid_onsets)
        if n_segments == 0:
            continue

        mfcc      = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        centroid  = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        rolloff   = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        zcr       = librosa.feature.zero_crossing_rate(y)[0]
        rms       = librosa.feature.rms(y=y)[0]
        S         = np.abs(librosa.stft(y)) ** 2
        S_norm    = S / (S.sum(axis=0, keepdims=True) + 1e-10)
        frame_ent = -np.sum(S_norm * np.log(S_norm + 1e-10), axis=0)
        rms_norm  = rms / (rms.sum() + 1e-10)

        feat = {
            **{f"mfcc_mean_{i}": mfcc.mean(axis=1)[i] for i in range(13)},
            **{f"mfcc_std_{i}":  mfcc.std(axis=1)[i]  for i in range(13)},
            "centroid_mean":     centroid.mean(),
            "centroid_std":      centroid.std(),
            "rolloff_mean":      rolloff.mean(),
            "rolloff_std":       rolloff.std(),
            "bandwidth_mean":    bandwidth.mean(),
            "bandwidth_std":     bandwidth.std(),
            "zcr_mean":          zcr.mean(),
            "zcr_std":           zcr.std(),
            "rms_mean":          rms.mean(),
            "rms_std":           rms.std(),
            "silence_fraction":  (rms < 0.01).mean(),
            "spec_entropy_mean": frame_ent.mean(),
            "spec_entropy_std":  frame_ent.std(),
            "temporal_entropy":  scipy_entropy(rms_norm + 1e-10),
            "onset_count":       n_segments,
            "clip_name":         clip_name,
            "clip_label":        clip_label,
            "clip_hour":         clip_hour,
            "sim_type":          am2.loc[am2['clip_name']==clip_name, 'Sim Type'].values[0]
                                 if clip_label == 1 else 'baseline',
            "n_segments":        n_segments,
        }
        for _ in range(n_segments):
            rows.append(feat)

    if missing:
        print(f"  WARNING: {missing} wav files not found / unreadable")

    feat_df = pd.DataFrame(rows)
    print(f"  Built {len(feat_df)} feature rows from {len(clips_ordered)-missing} clips")

    # ── 6. alignment check ────────────────────────────────────────────────────
    print(f"\nEmbedding rows : {len(embeddings)}")
    print(f"Feature rows   : {len(feat_df)}")

    if len(feat_df) != len(embeddings):
        print("WARNING: counts differ -- trimming to shorter length.")
        n          = min(len(feat_df), len(embeddings))
        feat_df    = feat_df.iloc[:n].reset_index(drop=True)
        embeddings = embeddings[:n]
        labels     = labels[:n]
    else:
        print("Row counts match -- alignment confirmed.")

    label_match = (feat_df['clip_label'].values == labels).mean()
    print(f"Label agreement: {label_match*100:.1f}%")
    if label_match < 0.95:
        print("WARNING: labels don't match well -- check clip ordering.")

    # ── 7. hour-normalized z-scores ───────────────────────────────────────────
    print("\nComputing hour-normalized z-scores...")
    norm_rows = []

    for hour in feat_df['clip_hour'].unique():
        base_mask = (feat_df['clip_hour'] == hour) & (feat_df['clip_label'] == 0)
        base_rows = feat_df.loc[base_mask, SPECTRAL_KEYS]

        if len(base_rows) == 0:
            print(f"  WARNING: No baseline clips at hour {hour} -- cannot z-score")
            all_mask = feat_df['clip_hour'] == hour
            zero_df  = feat_df.loc[all_mask, SPECTRAL_KEYS].copy() * 0
            zero_df.columns = [f"z_{c}" for c in zero_df.columns]
            norm_rows.append(zero_df)
            continue

        hour_mean = base_rows.mean()
        hour_std  = base_rows.std().replace(0, 1e-6)
        all_mask  = feat_df['clip_hour'] == hour
        all_rows  = feat_df.loc[all_mask, SPECTRAL_KEYS]
        z_scored  = (all_rows - hour_mean) / hour_std
        z_scored.columns = [f"z_{c}" for c in z_scored.columns]
        norm_rows.append(z_scored)

    norm_df = pd.concat(norm_rows).sort_index()
    feat_df = pd.concat([feat_df.reset_index(drop=True),
                         norm_df.reset_index(drop=True)], axis=1)
    print(f"  Added {len(norm_df.columns)} hour-normalized features")

    # ── 8. Perch PCA features ─────────────────────────────────────────────────
    print("Adding Perch PCA features (top 20 components)...")
    pca    = PCA(n_components=20, random_state=RANDOM_SEED)
    emb_pc = pca.fit_transform(embeddings)
    emb_df = pd.DataFrame(emb_pc, columns=[f"emb_pc{i}" for i in range(20)])
    feat_df = pd.concat([feat_df.reset_index(drop=True), emb_df], axis=1)

    # ── 9. save feature matrix ────────────────────────────────────────────────
    feat_df.to_csv(OUT_CSV, index=False)
    print(f"Saved -> {OUT_CSV}  shape={feat_df.shape}")

    # ── 10. classify ──────────────────────────────────────────────────────────
    meta_cols    = {"clip_name", "clip_label", "clip_hour",
                    "n_segments", "sim_type"}
    feature_cols = [c for c in feat_df.columns if c not in meta_cols]
    X_feat = feat_df[feature_cols].values.astype(np.float32)
    y      = labels.astype(int)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_feat)

    cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced",
                              random_state=RANDOM_SEED)

    print("\nRunning 5-fold stratified CV...")
    all_preds = np.zeros_like(y)
    all_probs = np.zeros(len(y), dtype=np.float32)

    for fold, (tr, va) in enumerate(cv.split(X_scaled, y)):
        clf.fit(X_scaled[tr], y[tr])
        all_preds[va] = clf.predict(X_scaled[va])
        all_probs[va] = clf.predict_proba(X_scaled[va])[:, 1]
        p, r, f, _ = precision_recall_fscore_support(
            y[va], all_preds[va], average="binary", zero_division=0)
        print(f"  Fold {fold+1}: precision={p:.3f}  recall={r:.3f}  F1={f:.3f}")

    # ── 11. segment-level results (t=0.50) ────────────────────────────────────
    report_base = classification_report(
        y, all_preds, target_names=["baseline", "simulation"], zero_division=0)
    p0, r0, f0, _ = precision_recall_fscore_support(
        y, all_preds, average="binary", zero_division=0)
    print("\nSegment-level results (threshold=0.50)")
    print(report_base)
    print(f"Precision {p0:.3f}  Recall {r0:.3f}  F1 {f0:.3f}")

    # ── 12. threshold tuning ──────────────────────────────────────────────────
    print("\nThreshold tuning")
    precisions, recalls, thresholds = precision_recall_curve(y, all_probs)
    valid = np.where((precisions[:-1] >= 0.70) & (recalls[:-1] >= 0.70))[0]

    if len(valid) > 0:
        best_idx    = valid[np.argmax(precisions[valid] + recalls[valid])]
        best_thresh = float(thresholds[best_idx])
        preds_tuned = (all_probs >= best_thresh).astype(int)
        p1, r1, f1, _ = precision_recall_fscore_support(
            y, preds_tuned, average="binary", zero_division=0)
        beat_thresh = True
        print(f"Best threshold: {best_thresh:.3f}")
        print(f"Precision {p1:.3f}  Recall {r1:.3f}  F1 {f1:.3f}")
        print("BEAT baseline")
    else:
        best_idx    = np.argmax(precisions[:-1] + recalls[:-1])
        best_thresh = float(thresholds[best_idx])
        preds_tuned = (all_probs >= best_thresh).astype(int)
        p1, r1, f1, _ = precision_recall_fscore_support(
            y, preds_tuned, average="binary", zero_division=0)
        beat_thresh = False
        print(f"No threshold achieves both >= 0.70")
        print(f"Best available (t={best_thresh:.3f}): "
              f"precision={p1:.3f}  recall={r1:.3f}  F1={f1:.3f}")

    # ── 13. clip-level voting ─────────────────────────────────────────────────
    print(f"\nClip-level voting (vote_frac={VOTE_FRAC})")
    clip_results = []
    for clip_name in feat_df['clip_name'].unique():
        mask       = feat_df['clip_name'].values == clip_name
        true_label = feat_df['clip_label'].values[mask][0]
        sim_type   = feat_df['sim_type'].values[mask][0]
        seg_probs  = all_probs[mask]
        frac_above = (seg_probs >= best_thresh).mean()
        pred_label = int(frac_above >= VOTE_FRAC)
        clip_results.append({
            'clip_name':  clip_name,
            'true':       true_label,
            'pred':       pred_label,
            'sim_type':   sim_type,
            'n_segments': int(mask.sum()),
            'frac_sim':   round(float(frac_above), 3),
        })

    clip_df = pd.DataFrame(clip_results)
    p2, r2, f2, _ = precision_recall_fscore_support(
        clip_df['true'], clip_df['pred'], average="binary", zero_division=0)
    report_clip = classification_report(
        clip_df['true'], clip_df['pred'],
        target_names=["baseline", "simulation"], zero_division=0)
    beat_clip = (p2 >= 0.70 and r2 >= 0.70)
    print(report_clip)
    print(f"Precision {p2:.3f}  Recall {r2:.3f}  F1 {f2:.3f}  ({len(clip_df)} clips)")
    print("BEAT baseline" if beat_clip else "Not yet beating baseline")

    # ── 14. breakdown by simulation type ─────────────────────────────────────
    # This is important for AM2 because it has chainsaw — a new disturbance type
    print("\nPerformance breakdown by simulation type:")
    sim_clip_df = clip_df[clip_df['true'] == 1].copy()
    for sim_type in sim_clip_df['sim_type'].unique():
        mask     = sim_clip_df['sim_type'] == sim_type
        n_total  = mask.sum()
        n_found  = sim_clip_df.loc[mask, 'pred'].sum()
        recall_t = n_found / n_total if n_total > 0 else 0
        print(f"  {sim_type:<40s}  detected {n_found}/{n_total}  ({recall_t:.1%})")

    # ── 15. summary ───────────────────────────────────────────────────────────
    print("\nSummary")
    print(f"{'Method':<38} {'Precision':>10} {'Recall':>8} {'F1':>6}")
    print("-" * 65)
    print(f"{'Segment default (t=0.50)':<38} {p0:>10.3f} {r0:>8.3f} {f0:>6.3f}")
    print(f"{'Segment tuned  (t={:.2f})'.format(best_thresh):<38} {p1:>10.3f} {r1:>8.3f} {f1:>6.3f}")
    print(f"{'Clip voting    (frac={})'.format(VOTE_FRAC):<38} {p2:>10.3f} {r2:>8.3f} {f2:>6.3f}")
    print(f"{'Baseline to beat':<38} {'0.700':>10} {'0.700':>8} {'-':>6}")
    print("\nCompare with AM4 results:")
    print(f"  AM4 segment tuned:  precision=0.704  recall=0.851")
    print(f"  AM2 segment tuned:  precision={p1:.3f}  recall={r1:.3f}")
    diff_p = p1 - 0.704
    diff_r = r1 - 0.851
    print(f"  Difference:         precision{diff_p:+.3f}  recall{diff_r:+.3f}")
    if abs(diff_p) < 0.05 and abs(diff_r) < 0.05:
        print("  --> Results are similar to AM4. Features appear to GENERALIZE.")
    else:
        print("  --> Results differ from AM4. Features may need site-specific tuning.")

    # ── 16. top features ──────────────────────────────────────────────────────
    clf.fit(X_scaled, y)
    importance = np.abs(clf.coef_[0])
    top_idx    = np.argsort(importance)[::-1][:20]
    print("\nTop 20 features:")
    for rank, i in enumerate(top_idx, 1):
        print(f"  {rank:2d}. {feature_cols[i]:<35s}  coef={clf.coef_[0][i]:+.4f}")

    # ── 17. precision-recall curve ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("#2C5F2D")
    ax.set_facecolor("#2C5F2D")
    ax.plot(recalls, precisions, color="#97BC62", linewidth=2.5, label="AM2 classifier")
    ax.axhline(0.70, color="white", linestyle="--", linewidth=1.2,
               alpha=0.6, label="Target precision 0.70")
    ax.axvline(0.70, color="white", linestyle=":", linewidth=1.2,
               alpha=0.6, label="Target recall 0.70")
    if len(valid) > 0:
        idx = np.argmin(np.abs(thresholds - best_thresh))
        ax.scatter(recalls[idx], precisions[idx], color="white", s=80, zorder=5)
        ax.annotate(f"t={best_thresh:.2f}\nP={p1:.3f} R={r1:.3f}",
                    xy=(recalls[idx], precisions[idx]),
                    xytext=(recalls[idx]-0.2, precisions[idx]-0.08),
                    color="white", fontsize=9,
                    arrowprops=dict(arrowstyle="->", color="white", lw=1))
    ax.set_facecolor("#2C5F2D")
    ax.tick_params(colors="white")
    ax.set_xlabel("Recall", color="white", fontsize=12)
    ax.set_ylabel("Precision", color="white", fontsize=12)
    ax.set_title("Precision-Recall Curve -- AM2 Classifier", color="white", fontsize=13)
    ax.legend(facecolor="#1F451F", labelcolor="white", framealpha=0.8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#97BC62")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(OUT_CURVE, dpi=150, bbox_inches="tight", facecolor="#2C5F2D")
    plt.show()
    print(f"Precision-recall curve saved -> {OUT_CURVE}")

    # ── 18. save report ───────────────────────────────────────────────────────
    with open(OUT_REPORT, "w", encoding="utf-8") as fh:
        fh.write("AM2 Classifier Results -- Time-Controlled\n")
        fh.write("=" * 50 + "\n\n")
        fh.write(f"n_segments:   {len(feat_df)}\n")
        fh.write(f"n_clips:      {len(clip_df)}\n")
        fh.write(f"n_features:   {len(feature_cols)}\n")
        fh.write(f"sim segs:     {int(y.sum())}\n")
        fh.write(f"base segs:    {int((y==0).sum())}\n")
        fh.write(f"sim hours:    {sorted(sim_hours)}\n\n")
        fh.write("Segment level (t=0.50)\n")
        fh.write(report_base)
        fh.write(f"\nPrecision {p0:.3f}  Recall {r0:.3f}  F1 {f0:.3f}\n\n")
        fh.write(f"Threshold tuned (t={best_thresh:.3f})\n")
        fh.write(f"Precision {p1:.3f}  Recall {r1:.3f}  F1 {f1:.3f}\n")
        fh.write("BEAT baseline\n\n" if beat_thresh else "Not yet beating baseline\n\n")
        fh.write(f"Clip voting (frac={VOTE_FRAC})\n")
        fh.write(report_clip)
        fh.write(f"\nPrecision {p2:.3f}  Recall {r2:.3f}  F1 {f2:.3f}\n")
        fh.write("BEAT baseline\n\n" if beat_clip else "Not yet beating baseline\n\n")
        fh.write("Breakdown by simulation type:\n")
        for sim_type in sim_clip_df['sim_type'].unique():
            mask     = sim_clip_df['sim_type'] == sim_type
            n_total  = mask.sum()
            n_found  = sim_clip_df.loc[mask, 'pred'].sum()
            recall_t = n_found / n_total if n_total > 0 else 0
            fh.write(f"  {sim_type:<40s}  {n_found}/{n_total}  ({recall_t:.1%})\n")
        fh.write("\nComparison with AM4:\n")
        fh.write(f"  AM4: precision=0.704  recall=0.851\n")
        fh.write(f"  AM2: precision={p1:.3f}  recall={r1:.3f}\n")
        fh.write(f"  Diff: precision{diff_p:+.3f}  recall{diff_r:+.3f}\n\n")
        fh.write("Top 20 features:\n")
        for rank, i in enumerate(top_idx, 1):
            fh.write(f"  {rank:2d}. {feature_cols[i]:<35s}  coef={clf.coef_[0][i]:+.4f}\n")

    print(f"\nReport saved -> {OUT_REPORT}")
    print("\nDone. Check am2_classifier_results.txt for the full summary.")