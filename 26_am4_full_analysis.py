"""
26_am4_full_analysis.py
------------------------
Re-runs the AM4 analysis using ALL available baseline clips from
simulation hours — not just the 5x sampled subset used in script 20/22.

Why: Script 20 capped baseline at 5x simulation count per hour (428 clips).
AM4 has 7,407 total clips. Using all available same-hour baseline clips
gives the classifier much more to learn from and produces more reliable
precision/recall estimates.

Key differences from script 22:
  - Uses ALL baseline clips from simulation hours (no 5x cap)
  - Re-embeds everything from scratch (new .npy files)
  - Everything else identical: same features, same z-scores, same classifier

Inputs:
  cleaned_df (1).csv
  filtered_clips_4/

Outputs:
  am4_full_emb.npy
  am4_full_labels.npy
  am4_full_feature_matrix.csv
  am4_full_classifier_results.txt
  am4_full_precision_recall_curve.png
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
CLIPS_DIR  = os.path.join(BASE_DIR, "filtered_clips_4")
META_CSV   = os.path.join(BASE_DIR, "cleaned_df (1).csv")
SAVE_EMB   = os.path.join(BASE_DIR, "am4_full_emb.npy")
SAVE_LAB   = os.path.join(BASE_DIR, "am4_full_labels.npy")
OUT_CSV    = os.path.join(BASE_DIR, "am4_full_feature_matrix.csv")
OUT_REPORT = os.path.join(BASE_DIR, "am4_full_classifier_results.txt")
OUT_CURVE  = os.path.join(BASE_DIR, "am4_full_precision_recall_curve.png")

SR          = None
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
    am4 = df[df['Recorder'] == 'Audio_Moth_4'].copy()
    am4['is_sim'] = am4['Sim Type'] != '[]'
    am4['hour']   = am4['clip_name'].str[22:24].astype(int)

    # Only keep clips that exist in folder
    am4['exists'] = am4['clip_name'].apply(
        lambda x: os.path.exists(os.path.join(CLIPS_DIR, x)))
    am4 = am4[am4['exists']].copy()

    print(f"  AM4 clips available: {len(am4)}")
    print(f"  Simulation clips:    {am4['is_sim'].sum()}")
    print(f"  Baseline clips:      {(~am4['is_sim']).sum()}")

    # ── 2. time-controlled clip selection — ALL baseline, no cap ─────────────
    print("\nSelecting clips (time-controlled, ALL baseline at sim hours)...")

    sim_clips = am4[am4['is_sim']][['clip_name', 'hour']].copy()
    sim_hours = sim_clips['hour'].unique()
    print(f"  Simulation hours: {sorted(sim_hours)}")
    print(f"  Total simulation clips: {len(sim_clips)}")

    baseline_parts = []
    for hour in sorted(sim_hours):
        hour_sim_count  = len(am4[(am4['is_sim']) & (am4['hour'] == hour)])
        hour_baseline   = am4[(~am4['is_sim']) & (am4['hour'] == hour)]
        n_available     = len(hour_baseline)

        # KEY DIFFERENCE FROM SCRIPT 22: use ALL available baseline clips
        # at simulation hours — no 5x cap
        baseline_parts.append(hour_baseline[['clip_name', 'hour']])
        print(f"  Hour {hour}: {hour_sim_count} sim clips, "
              f"{n_available} baseline clips (ALL used)")

    baseline_df = pd.concat(baseline_parts)
    print(f"\nTotal baseline clips: {len(baseline_df)}")
    print(f"Total simulation clips: {len(sim_clips)}")
    print(f"Ratio baseline:sim = {len(baseline_df)/len(sim_clips):.1f}:1")
    print(f"\nCompare with script 22: 428 baseline, 91 sim (4.7:1)")

    # Order: sims first (label=1), then baseline (label=0)
    clips_ordered = (
        [(c, 1, h) for c, h in zip(sim_clips['clip_name'], sim_clips['hour'])] +
        [(c, 0, h) for c, h in zip(baseline_df['clip_name'], baseline_df['hour'])]
    )
    print(f"Total clips to process: {len(clips_ordered)}")

    # ── 3. embed with Perch v2 ────────────────────────────────────────────────
    if os.path.exists(SAVE_EMB):
        print("\nLoading saved AM4 full embeddings...")
        embeddings = np.load(SAVE_EMB)
        labels     = np.load(SAVE_LAB)
        print(f"  Loaded {len(embeddings)} segments | "
              f"{int(labels.sum())} sim | {int((labels==0).sum())} baseline")
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
                    y=y, sr=sr, units='time',
                    delta=ONSET_DELTA, wait=ONSET_WAIT)
                for onset in onsets:
                    start = int(onset * sr)
                    end   = int((onset + WINDOW) * sr)
                    if end < len(y):
                        all_segments.append(
                            (y[start:end].astype(np.float32), sr))
                        all_labels.append(label)
            except Exception as e:
                print(f"  WARNING: could not load {clip_name}: {e}")
                continue

        print(f"Collected {len(all_segments)} segments. "
              f"Embedding in one batch...")
        results      = model.encode_arrays(
            iter(all_segments), n_producers=1, batch_size=32)
        results_list = list(results)
        embeddings   = np.array(
            [r['embedding'].flatten() for r in results_list])
        labels       = np.array(all_labels)

        np.save(SAVE_EMB, embeddings)
        np.save(SAVE_LAB, labels)
        print(f"Saved {len(embeddings)} embeddings.")

    # ── 4. extract spectral features ──────────────────────────────────────────
    print("\nExtracting spectral features...")
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
            "n_segments":        n_segments,
        }
        for _ in range(n_segments):
            rows.append(feat)

    if missing:
        print(f"  WARNING: {missing} wav files not found")

    feat_df = pd.DataFrame(rows)
    print(f"  Built {len(feat_df)} feature rows from "
          f"{len(clips_ordered)-missing} clips")

    # ── 5. alignment check ────────────────────────────────────────────────────
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
        print("WARNING: labels don't match well.")

    # ── 6. hour-normalized z-scores ───────────────────────────────────────────
    print("\nComputing hour-normalized z-scores...")
    norm_rows = []

    for hour in feat_df['clip_hour'].unique():
        base_mask = ((feat_df['clip_hour'] == hour) &
                     (feat_df['clip_label'] == 0))
        base_rows = feat_df.loc[base_mask, SPECTRAL_KEYS]
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

    # ── 7. Perch PCA features ─────────────────────────────────────────────────
    print("Adding Perch PCA features (top 20 components)...")
    pca    = PCA(n_components=20, random_state=RANDOM_SEED)
    emb_pc = pca.fit_transform(embeddings)
    emb_df = pd.DataFrame(emb_pc, columns=[f"emb_pc{i}" for i in range(20)])
    feat_df = pd.concat([feat_df.reset_index(drop=True), emb_df], axis=1)

    # ── 8. save feature matrix ────────────────────────────────────────────────
    feat_df.to_csv(OUT_CSV, index=False)
    print(f"Saved -> {OUT_CSV}  shape={feat_df.shape}")

    # ── 9. classify ───────────────────────────────────────────────────────────
    meta_cols    = {"clip_name", "clip_label", "clip_hour", "n_segments"}
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

    # ── 10. segment-level results ─────────────────────────────────────────────
    report_base = classification_report(
        y, all_preds,
        target_names=["baseline", "simulation"], zero_division=0)
    p0, r0, f0, _ = precision_recall_fscore_support(
        y, all_preds, average="binary", zero_division=0)
    print("\nSegment-level results (threshold=0.50)")
    print(report_base)
    print(f"Precision {p0:.3f}  Recall {r0:.3f}  F1 {f0:.3f}")

    # ── 11. threshold tuning ──────────────────────────────────────────────────
    print("\nThreshold tuning")
    precisions, recalls, thresholds = precision_recall_curve(y, all_probs)
    valid = np.where(
        (precisions[:-1] >= 0.70) & (recalls[:-1] >= 0.70))[0]

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

    # ── 12. clip-level voting ─────────────────────────────────────────────────
    print(f"\nClip-level voting (vote_frac={VOTE_FRAC})")
    clip_results = []
    for clip_name in feat_df['clip_name'].unique():
        mask       = feat_df['clip_name'].values == clip_name
        true_label = feat_df['clip_label'].values[mask][0]
        seg_probs  = all_probs[mask]
        frac_above = (seg_probs >= best_thresh).mean()
        pred_label = int(frac_above >= VOTE_FRAC)
        clip_results.append({
            'clip_name':  clip_name,
            'true':       true_label,
            'pred':       pred_label,
            'n_segments': int(mask.sum()),
            'frac_sim':   round(float(frac_above), 3),
        })

    clip_df = pd.DataFrame(clip_results)
    p2, r2, f2, _ = precision_recall_fscore_support(
        clip_df['true'], clip_df['pred'],
        average="binary", zero_division=0)
    report_clip = classification_report(
        clip_df['true'], clip_df['pred'],
        target_names=["baseline", "simulation"], zero_division=0)
    beat_clip = (p2 >= 0.70 and r2 >= 0.70)
    print(report_clip)
    print(f"Precision {p2:.3f}  Recall {r2:.3f}  F1 {f2:.3f}  "
          f"({len(clip_df)} clips)")
    print("BEAT baseline" if beat_clip else "Not yet beating baseline")

    # ── 13. summary with direct comparison to script 22 ──────────────────────
    print("\nSummary")
    print(f"{'Method':<40} {'Precision':>10} {'Recall':>8} {'F1':>6}")
    print("-" * 67)
    print(f"{'Segment default (t=0.50)':<40} {p0:>10.3f} {r0:>8.3f} {f0:>6.3f}")
    print(f"{'Segment tuned  (t={:.2f})'.format(best_thresh):<40} "
          f"{p1:>10.3f} {r1:>8.3f} {f1:>6.3f}")
    print(f"{'Clip voting    (frac={})'.format(VOTE_FRAC):<40} "
          f"{p2:>10.3f} {r2:>8.3f} {f2:>6.3f}")
    print(f"{'Baseline to beat':<40} {'0.700':>10} {'0.700':>8} {'-':>6}")
    print("\nComparison with script 22 (428 baseline clips):")
    print(f"  Script 22 (subset):   precision=0.704  recall=0.851")
    print(f"  Script 26 (full):     precision={p1:.3f}  recall={r1:.3f}")
    diff_p = p1 - 0.704
    diff_r = r1 - 0.851
    print(f"  Difference:           precision{diff_p:+.3f}  recall{diff_r:+.3f}")
    if diff_p > 0.01:
        print("  --> More baseline data IMPROVED precision")
    elif diff_p < -0.01:
        print("  --> More baseline data reduced precision "
              "(model sees more diversity, harder to separate)")
    else:
        print("  --> Results similar -- 5x sampling was sufficient for AM4")

    # ── 14. top features ──────────────────────────────────────────────────────
    clf.fit(X_scaled, y)
    importance = np.abs(clf.coef_[0])
    top_idx    = np.argsort(importance)[::-1][:20]
    print("\nTop 20 features:")
    for rank, i in enumerate(top_idx, 1):
        print(f"  {rank:2d}. {feature_cols[i]:<35s}  "
              f"coef={clf.coef_[0][i]:+.4f}")

    # ── 15. precision-recall curve ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("#2C5F2D")
    ax.set_facecolor("#2C5F2D")
    ax.plot(recalls, precisions, color="#97BC62",
            linewidth=2.5, label="AM4 full classifier")
    ax.axhline(0.70, color="white", linestyle="--",
               linewidth=1.2, alpha=0.6, label="Target precision 0.70")
    ax.axvline(0.70, color="white", linestyle=":",
               linewidth=1.2, alpha=0.6, label="Target recall 0.70")
    if len(valid) > 0:
        idx = np.argmin(np.abs(thresholds - best_thresh))
        ax.scatter(recalls[idx], precisions[idx],
                   color="white", s=80, zorder=5)
        ax.annotate(f"t={best_thresh:.2f}\nP={p1:.3f} R={r1:.3f}",
                    xy=(recalls[idx], precisions[idx]),
                    xytext=(recalls[idx]-0.2, precisions[idx]-0.08),
                    color="white", fontsize=9,
                    arrowprops=dict(arrowstyle="->", color="white", lw=1))
    ax.tick_params(colors="white")
    ax.set_xlabel("Recall", color="white", fontsize=12)
    ax.set_ylabel("Precision", color="white", fontsize=12)
    ax.set_title("Precision-Recall Curve -- AM4 Full Classifier",
                 color="white", fontsize=13)
    ax.legend(facecolor="#1F451F", labelcolor="white", framealpha=0.8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#97BC62")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(OUT_CURVE, dpi=150, bbox_inches="tight",
                facecolor="#2C5F2D")
    plt.show()
    print(f"Curve saved -> {OUT_CURVE}")

    # ── 16. save report ───────────────────────────────────────────────────────
    with open(OUT_REPORT, "w", encoding="utf-8") as fh:
        fh.write("AM4 Full Classifier Results\n")
        fh.write("=" * 50 + "\n\n")
        fh.write(f"n_clips total:    {len(clips_ordered)}\n")
        fh.write(f"n_sim clips:      {len(sim_clips)}\n")
        fh.write(f"n_base clips:     {len(baseline_df)}\n")
        fh.write(f"n_segments:       {len(feat_df)}\n")
        fh.write(f"n_features:       {len(feature_cols)}\n")
        fh.write(f"sim hours:        {sorted(sim_hours)}\n\n")
        fh.write("Segment level (t=0.50)\n")
        fh.write(report_base)
        fh.write(f"\nPrecision {p0:.3f}  Recall {r0:.3f}  F1 {f0:.3f}\n\n")
        fh.write(f"Threshold tuned (t={best_thresh:.3f})\n")
        fh.write(f"Precision {p1:.3f}  Recall {r1:.3f}  F1 {f1:.3f}\n")
        fh.write("BEAT baseline\n\n" if beat_thresh
                 else "Not yet beating baseline\n\n")
        fh.write(f"Clip voting (frac={VOTE_FRAC})\n")
        fh.write(report_clip)
        fh.write(f"\nPrecision {p2:.3f}  Recall {r2:.3f}  F1 {f2:.3f}\n")
        fh.write("BEAT baseline\n\n" if beat_clip
                 else "Not yet beating baseline\n\n")
        fh.write("Comparison with script 22 (subset):\n")
        fh.write(f"  Script 22: precision=0.704  recall=0.851\n")
        fh.write(f"  Script 26: precision={p1:.3f}  recall={r1:.3f}\n")
        fh.write(f"  Diff:      precision{diff_p:+.3f}  recall{diff_r:+.3f}\n\n")
        fh.write("Top 20 features:\n")
        for rank, i in enumerate(top_idx, 1):
            fh.write(f"  {rank:2d}. {feature_cols[i]:<35s}  "
                     f"coef={clf.coef_[0][i]:+.4f}\n")

    print(f"\nReport saved -> {OUT_REPORT}")
    print("Done.")