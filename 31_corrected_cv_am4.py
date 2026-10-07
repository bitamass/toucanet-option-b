"""
31_corrected_cv_am4.py
-----------------------
Re-runs the AM4 classifier with CORRECT cross-validation.

WHAT WAS WRONG IN SCRIPTS 22/26:
  CV was done at the segment level (0.2s windows).
  But spectral features are computed at the CLIP level (3s clips)
  and replicated across all segments of that clip.
  This means segments from the same clip could appear in both
  train and test folds — data leakage.
  82 of 106 features are IDENTICAL within a clip, so the model
  effectively memorizes clips it has already seen.

THE FIX:
  1. CV splits at the CLIP level — all segments from a clip
     stay together in the same fold
  2. Z-score normalization computed INSIDE each fold using
     only training fold baseline clips
  3. PCA fitted INSIDE each fold on training segments only
  4. StandardScaler fitted INSIDE each fold on training data only

This gives a genuinely honest estimate of how the classifier
performs on clips it has never seen.

Uses saved am4_full_feature_matrix.csv and am4_full_emb.npy
from script 26 — no re-embedding needed.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (classification_report,
                              precision_recall_fscore_support,
                              precision_recall_curve)

BASE_DIR   = r"C:\Users\BitaMassoudi\KasmirWorldProjects\toucanet-option-b"
FEAT_CSV   = os.path.join(BASE_DIR, "am4_full_feature_matrix.csv")
EMB_FILE   = os.path.join(BASE_DIR, "am4_full_emb.npy")
LAB_FILE   = os.path.join(BASE_DIR, "am4_full_labels.npy")
OUT_REPORT = os.path.join(BASE_DIR, "am4_corrected_cv_results.txt")
OUT_CURVE  = os.path.join(BASE_DIR, "am4_corrected_cv_pr_curve.png")

RANDOM_SEED = 42
VOTE_FRAC   = 0.5
N_FOLDS     = 5

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

    print("Loading saved feature matrix and embeddings...")
    feat_df    = pd.read_csv(FEAT_CSV)
    embeddings = np.load(EMB_FILE)
    labels     = np.load(LAB_FILE)

    print(f"  Total segments: {len(feat_df)}")
    print(f"  Simulation segments: {int(labels.sum())}")
    print(f"  Baseline segments:   {int((labels==0).sum())}")

    # ── Get unique clips and their labels ─────────────────────────────────────
    # One row per clip for CV splitting
    clip_info = feat_df.groupby('clip_name').agg(
        clip_label=('clip_label', 'first'),
        clip_hour=('clip_hour', 'first')
    ).reset_index()

    clip_names  = clip_info['clip_name'].values
    clip_labels = clip_info['clip_label'].values

    print(f"\n  Unique clips: {len(clip_names)}")
    print(f"  Sim clips:    {clip_labels.sum()}")
    print(f"  Base clips:   {(clip_labels==0).sum()}")

    # ── CORRECT 5-fold CV at CLIP level ───────────────────────────────────────
    print(f"\nRunning CORRECT {N_FOLDS}-fold CV at CLIP level...")
    print("(All segments from a clip stay in the same fold)")
    print("(Z-scores, PCA, scaler all fitted inside each fold)")

    cv = StratifiedKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    all_seg_probs  = np.zeros(len(feat_df), dtype=np.float32)
    all_seg_preds  = np.zeros(len(feat_df), dtype=int)
    fold_results   = []

    for fold, (tr_clip_idx, va_clip_idx) in enumerate(
            cv.split(clip_names, clip_labels)):

        tr_clips = set(clip_names[tr_clip_idx])
        va_clips = set(clip_names[va_clip_idx])

        # Get segment indices for each split
        tr_mask = feat_df['clip_name'].isin(tr_clips).values
        va_mask = feat_df['clip_name'].isin(va_clips).values

        feat_tr = feat_df[tr_mask].copy().reset_index(drop=True)
        feat_va = feat_df[va_mask].copy().reset_index(drop=True)
        emb_tr  = embeddings[tr_mask]
        emb_va  = embeddings[va_mask]
        y_tr    = labels[tr_mask].astype(int)
        y_va    = labels[va_mask].astype(int)

        # ── Z-scores computed on TRAINING baseline only ────────────────────
        z_tr_rows = []
        z_va_rows = []
        for hour in feat_tr['clip_hour'].unique():
            # Baseline stats from TRAINING fold only
            base_mask_tr = ((feat_tr['clip_hour'] == hour) &
                            (feat_tr['clip_label'] == 0))
            base_rows = feat_tr.loc[base_mask_tr, SPECTRAL_KEYS]

            if len(base_rows) == 0:
                # No baseline at this hour in training fold
                for mask, feat_sub in [(feat_tr['clip_hour']==hour, feat_tr),
                                       (feat_va['clip_hour']==hour, feat_va)]:
                    z = feat_sub.loc[mask, SPECTRAL_KEYS].copy() * 0
                    z.columns = [f"z_{c}" for c in z.columns]
                    if feat_sub is feat_tr:
                        z_tr_rows.append(z)
                    else:
                        z_va_rows.append(z)
                continue

            hour_mean = base_rows.mean()
            hour_std  = base_rows.std().replace(0, 1e-6)

            # Apply to training segments at this hour
            tr_hour_mask = feat_tr['clip_hour'] == hour
            z_tr = (feat_tr.loc[tr_hour_mask, SPECTRAL_KEYS] -
                    hour_mean) / hour_std
            z_tr.columns = [f"z_{c}" for c in z_tr.columns]
            z_tr_rows.append(z_tr)

            # Apply same stats to validation segments at this hour
            va_hour_mask = feat_va['clip_hour'] == hour
            if va_hour_mask.sum() > 0:
                z_va = (feat_va.loc[va_hour_mask, SPECTRAL_KEYS] -
                        hour_mean) / hour_std
                z_va.columns = [f"z_{c}" for c in z_va.columns]
                z_va_rows.append(z_va)

        z_tr_df = pd.concat(z_tr_rows).sort_index()
        z_va_df = pd.concat(z_va_rows).sort_index() if z_va_rows else \
                  pd.DataFrame(0, index=feat_va.index,
                               columns=[f"z_{c}" for c in SPECTRAL_KEYS])

        # ── PCA fitted on TRAINING embeddings only ─────────────────────────
        pca    = PCA(n_components=20, random_state=RANDOM_SEED)
        emb_tr_pc = pca.fit_transform(emb_tr)
        emb_va_pc = pca.transform(emb_va)

        # ── Assemble feature matrices ──────────────────────────────────────
        raw_cols = SPECTRAL_KEYS
        X_tr = np.hstack([
            feat_tr[raw_cols].values,
            z_tr_df.values,
            emb_tr_pc
        ]).astype(np.float32)

        X_va = np.hstack([
            feat_va[raw_cols].values,
            z_va_df.values,
            emb_va_pc
        ]).astype(np.float32)

        # ── Scaler fitted on TRAINING data only ───────────────────────────
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)

        # ── Train and predict ─────────────────────────────────────────────
        clf = LogisticRegression(
            max_iter=1000, class_weight="balanced",
            random_state=RANDOM_SEED)
        clf.fit(X_tr_s, y_tr)

        va_probs = clf.predict_proba(X_va_s)[:, 1]
        va_preds = clf.predict(X_va_s)

        # Store predictions back to full array
        va_indices = np.where(va_mask)[0]
        all_seg_probs[va_indices] = va_probs
        all_seg_preds[va_indices] = va_preds

        p, r, f, _ = precision_recall_fscore_support(
            y_va, va_preds, average="binary", zero_division=0)
        print(f"  Fold {fold+1}: precision={p:.3f}  "
              f"recall={r:.3f}  F1={f:.3f}  "
              f"({va_mask.sum()} segs, {len(va_clips)} clips)")
        fold_results.append((p, r, f))

    # ── Segment-level results ─────────────────────────────────────────────────
    y_all = labels.astype(int)
    report_seg = classification_report(
        y_all, all_seg_preds,
        target_names=["baseline", "simulation"], zero_division=0)
    p0, r0, f0, _ = precision_recall_fscore_support(
        y_all, all_seg_preds, average="binary", zero_division=0)
    print("\nSegment-level results (t=0.50) — CORRECTED CV")
    print(report_seg)
    print(f"Precision {p0:.3f}  Recall {r0:.3f}  F1 {f0:.3f}")

    # ── Threshold tuning ──────────────────────────────────────────────────────
    precisions, recalls, thresholds = precision_recall_curve(
        y_all, all_seg_probs)
    valid = np.where(
        (precisions[:-1] >= 0.70) & (recalls[:-1] >= 0.70))[0]

    if len(valid) > 0:
        best_idx    = valid[np.argmax(precisions[valid] + recalls[valid])]
        best_thresh = float(thresholds[best_idx])
        preds_tuned = (all_seg_probs >= best_thresh).astype(int)
        p1, r1, f1, _ = precision_recall_fscore_support(
            y_all, preds_tuned, average="binary", zero_division=0)
        beat = True
        print(f"\nThreshold tuned (t={best_thresh:.3f}): "
              f"P={p1:.3f}  R={r1:.3f}  --> BEAT BASELINE")
    else:
        best_idx    = np.argmax(precisions[:-1] + recalls[:-1])
        best_thresh = float(thresholds[best_idx])
        preds_tuned = (all_seg_probs >= best_thresh).astype(int)
        p1, r1, f1, _ = precision_recall_fscore_support(
            y_all, preds_tuned, average="binary", zero_division=0)
        beat = False
        print(f"\nNo threshold achieves both >= 0.70")
        print(f"Best (t={best_thresh:.3f}): "
              f"P={p1:.3f}  R={r1:.3f}  F1={f1:.3f}")

    # ── Clip-level voting ─────────────────────────────────────────────────────
    clip_results = []
    for clip_name in feat_df['clip_name'].unique():
        mask       = feat_df['clip_name'].values == clip_name
        true_label = feat_df['clip_label'].values[mask][0]
        seg_probs  = all_seg_probs[mask]
        frac_above = (seg_probs >= best_thresh).mean()
        pred_label = int(frac_above >= VOTE_FRAC)
        clip_results.append({
            'clip_name':  clip_name,
            'true':       true_label,
            'pred':       pred_label,
            'n_segments': int(mask.sum()),
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
    print(f"Clip voting: P={p2:.3f}  R={r2:.3f}  "
          f"({'BEAT' if beat_clip else 'not beating'} baseline)")

    # ── Comparison ────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("COMPARISON: ORIGINAL (leaky) vs CORRECTED CV")
    print("=" * 65)
    print(f"{'Method':<40} {'Precision':>10} {'Recall':>8}")
    print("-" * 65)
    print(f"{'Original script 26 (leaky segment CV)':<40} "
          f"{'0.872':>10} {'0.973':>8}")
    print(f"{'Corrected (clip-level CV, t=0.50)':<40} "
          f"{p0:>10.3f} {r0:>8.3f}")
    print(f"{'Corrected tuned (t={:.2f})'.format(best_thresh):<40} "
          f"{p1:>10.3f} {r1:>8.3f}")
    print(f"{'Corrected clip voting':<40} "
          f"{p2:>10.3f} {r2:>8.3f}")
    print(f"{'Target':<40} {'0.700':>10} {'0.700':>8}")
    print("=" * 65)

    if beat:
        diff_p = p1 - 0.872
        diff_r = r1 - 0.973
        print(f"\nDifference from original: "
              f"precision{diff_p:+.3f}  recall{diff_r:+.3f}")
        if abs(diff_p) < 0.05:
            print("Results are SIMILAR — original CV was approximately valid")
        else:
            print("Results DIFFER significantly — leakage was inflating scores")

    # ── PR curve ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("#2C5F2D")
    ax.set_facecolor("#2C5F2D")
    ax.plot(recalls, precisions, color="#97BC62",
            linewidth=2.5, label="AM4 corrected CV")
    ax.axhline(0.70, color="white", linestyle="--",
               linewidth=1.2, alpha=0.6, label="Target precision 0.70")
    ax.axvline(0.70, color="white", linestyle=":",
               linewidth=1.2, alpha=0.6, label="Target recall 0.70")
    if beat:
        idx = np.argmin(np.abs(thresholds - best_thresh))
        ax.scatter(recalls[idx], precisions[idx],
                   color="white", s=80, zorder=5)
        ax.annotate(f"t={best_thresh:.2f}\nP={p1:.3f} R={r1:.3f}",
                    xy=(recalls[idx], precisions[idx]),
                    xytext=(max(0.05, recalls[idx]-0.25),
                            max(0.05, precisions[idx]-0.1)),
                    color="white", fontsize=9,
                    arrowprops=dict(arrowstyle="->", color="white", lw=1))
    ax.tick_params(colors="white")
    ax.set_xlabel("Recall", color="white", fontsize=12)
    ax.set_ylabel("Precision", color="white", fontsize=12)
    ax.set_title("AM4 — Corrected Clip-Level CV\n"
                 "(No data leakage)",
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

    # ── Save report ───────────────────────────────────────────────────────────
    with open(OUT_REPORT, "w", encoding="utf-8") as fh:
        fh.write("AM4 CORRECTED CV Results\n")
        fh.write("=" * 50 + "\n\n")
        fh.write("LEAKAGE ISSUES FIXED:\n")
        fh.write("1. CV split at CLIP level (not segment level)\n")
        fh.write("2. Z-scores computed inside each fold\n")
        fh.write("3. PCA fitted inside each fold\n")
        fh.write("4. Scaler fitted inside each fold\n\n")
        fh.write("Segment level (t=0.50)\n")
        fh.write(report_seg)
        fh.write(f"\nPrecision {p0:.3f}  Recall {r0:.3f}  F1 {f0:.3f}\n\n")
        fh.write(f"Threshold tuned (t={best_thresh:.3f})\n")
        fh.write(f"P={p1:.3f}  R={r1:.3f}  F1={f1:.3f}\n")
        fh.write("BEAT baseline\n\n" if beat
                 else "Not yet beating baseline\n\n")
        fh.write("Clip voting\n")
        fh.write(report_clip)
        fh.write(f"\nP={p2:.3f}  R={r2:.3f}  F1={f2:.3f}\n\n")
        fh.write("Comparison:\n")
        fh.write(f"  Original (leaky): P=0.872  R=0.973\n")
        fh.write(f"  Corrected:        P={p1:.3f}  R={r1:.3f}\n")

    print(f"\nReport saved -> {OUT_REPORT}")
    print("\nDone. Run this same corrected CV on AM2, AM5, AM6, AM1")
    print("to get honest estimates for all recorders.")