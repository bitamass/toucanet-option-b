"""
27_am2_balance_experiment.py
-----------------------------
Tests whether the precision problem on AM2 is caused by class imbalance.

AM2 currently has 2,263 baseline vs 453 simulation clips (5:1 ratio).
This script tries 3 different baseline sampling ratios:
  - 1x (453 baseline : 453 simulation)
  - 2x (906 baseline : 453 simulation)
  - 3x (1,359 baseline : 453 simulation)
  - 5x (2,265 baseline : 453 simulation) -- current, for comparison

For each ratio it trains the same logistic regression classifier and
reports precision, recall, F1. This tells us whether more baseline
data is hurting precision or whether the problem is deeper.

Reuses saved embeddings from script 25 -- no re-embedding needed.
Reuses saved feature matrix from script 25 -- no re-extraction needed.

Inputs (already saved by script 25):
  am2_time_controlled_emb.npy
  am2_time_controlled_labels.npy
  am2_feature_matrix.csv

Outputs:
  am2_balance_experiment_results.txt
  am2_balance_experiment_plot.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (precision_recall_fscore_support,
                              precision_recall_curve,
                              classification_report)

BASE_DIR   = r"C:\Users\BitaMassoudi\KasmirWorldProjects\toucanet-option-b"
FEAT_CSV   = os.path.join(BASE_DIR, "am2_feature_matrix.csv")
EMB_FILE   = os.path.join(BASE_DIR, "am2_time_controlled_emb.npy")
LAB_FILE   = os.path.join(BASE_DIR, "am2_time_controlled_labels.npy")
OUT_REPORT = os.path.join(BASE_DIR, "am2_balance_experiment_results.txt")
OUT_PLOT   = os.path.join(BASE_DIR, "am2_balance_experiment_plot.png")

RANDOM_SEED = 42
VOTE_FRAC   = 0.5
RATIOS      = [1, 2, 3, 5]   # baseline:simulation ratios to test

if __name__ == '__main__':

    # ── 1. load saved feature matrix and embeddings ───────────────────────────
    print("Loading saved feature matrix...")
    feat_df    = pd.read_csv(FEAT_CSV)
    embeddings = np.load(EMB_FILE)
    labels     = np.load(LAB_FILE)

    print(f"  Total segments: {len(feat_df)}")
    print(f"  Simulation segs: {int(labels.sum())}")
    print(f"  Baseline segs:   {int((labels==0).sum())}")

    meta_cols    = {"clip_name", "clip_label", "clip_hour",
                    "n_segments", "sim_type"}
    feature_cols = [c for c in feat_df.columns if c not in meta_cols]

    # ── 2. identify simulation and baseline clip names ────────────────────────
    sim_clips  = feat_df[feat_df['clip_label'] == 1]['clip_name'].unique()
    base_clips = feat_df[feat_df['clip_label'] == 0]['clip_name'].unique()

    n_sim  = len(sim_clips)
    n_base = len(base_clips)
    print(f"\n  Simulation clips: {n_sim}")
    print(f"  Baseline clips:   {n_base}")
    print(f"  Current ratio:    {n_base/n_sim:.1f}:1")

    # ── 3. run experiment for each ratio ──────────────────────────────────────
    results = []

    for ratio in RATIOS:
        n_base_sample = min(ratio * n_sim, n_base)
        print(f"\n{'='*55}")
        print(f"Testing ratio {ratio}:1  "
              f"({n_sim} sim : {n_base_sample} baseline clips)")
        print(f"{'='*55}")

        # Sample baseline clips
        np.random.seed(RANDOM_SEED)
        sampled_base = np.random.choice(
            base_clips, size=n_base_sample, replace=False)

        # Build mask for this experiment's rows
        keep_clips = set(sim_clips) | set(sampled_base)
        mask       = feat_df['clip_name'].isin(keep_clips)

        feat_sub = feat_df[mask].reset_index(drop=True)
        # Match labels to the subset — use feat_df clip_label not saved labels
        # (saved labels are segment-level and already aligned)
        seg_mask  = np.isin(
            feat_df['clip_name'].values, list(keep_clips))
        y_sub     = labels[seg_mask].astype(int)
        emb_sub   = embeddings[seg_mask]

        print(f"  Segments: {len(feat_sub)} total  |  "
              f"{int(y_sub.sum())} sim  |  {int((y_sub==0).sum())} baseline")

        # Features
        X_sub    = feat_sub[feature_cols].values.astype(np.float32)
        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X_sub)

        # 5-fold CV
        cv        = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        clf       = LogisticRegression(
            max_iter=1000, class_weight="balanced",
            random_state=RANDOM_SEED)
        all_preds = np.zeros_like(y_sub)
        all_probs = np.zeros(len(y_sub), dtype=np.float32)

        for fold, (tr, va) in enumerate(cv.split(X_scaled, y_sub)):
            clf.fit(X_scaled[tr], y_sub[tr])
            all_preds[va] = clf.predict(X_scaled[va])
            all_probs[va] = clf.predict_proba(X_scaled[va])[:, 1]
            p, r, f, _ = precision_recall_fscore_support(
                y_sub[va], all_preds[va],
                average="binary", zero_division=0)
            print(f"  Fold {fold+1}: precision={p:.3f}  "
                  f"recall={r:.3f}  F1={f:.3f}")

        # Default threshold
        p0, r0, f0, _ = precision_recall_fscore_support(
            y_sub, all_preds, average="binary", zero_division=0)

        # Threshold tuning
        precisions, recalls, thresholds = precision_recall_curve(
            y_sub, all_probs)
        valid = np.where(
            (precisions[:-1] >= 0.70) & (recalls[:-1] >= 0.70))[0]

        if len(valid) > 0:
            best_idx    = valid[np.argmax(
                precisions[valid] + recalls[valid])]
            best_thresh = float(thresholds[best_idx])
            preds_tuned = (all_probs >= best_thresh).astype(int)
            p1, r1, f1, _ = precision_recall_fscore_support(
                y_sub, preds_tuned, average="binary", zero_division=0)
            beat = True
            print(f"\n  Threshold tuned (t={best_thresh:.3f}): "
                  f"P={p1:.3f}  R={r1:.3f}  F1={f1:.3f}  --> BEAT BASELINE")
        else:
            best_idx    = np.argmax(precisions[:-1] + recalls[:-1])
            best_thresh = float(thresholds[best_idx])
            preds_tuned = (all_probs >= best_thresh).astype(int)
            p1, r1, f1, _ = precision_recall_fscore_support(
                y_sub, preds_tuned, average="binary", zero_division=0)
            beat = False
            print(f"\n  Best available (t={best_thresh:.3f}): "
                  f"P={p1:.3f}  R={r1:.3f}  F1={f1:.3f}  "
                  f"--> not yet beating baseline")

        # Clip-level voting
        clip_results = []
        for clip_name in feat_sub['clip_name'].unique():
            cmask      = feat_sub['clip_name'].values == clip_name
            true_label = feat_sub['clip_label'].values[cmask][0]
            seg_probs  = all_probs[cmask]
            frac_above = (seg_probs >= best_thresh).mean()
            pred_label = int(frac_above >= VOTE_FRAC)
            clip_results.append({
                'true': true_label,
                'pred': pred_label
            })
        clip_df = pd.DataFrame(clip_results)
        p2, r2, f2, _ = precision_recall_fscore_support(
            clip_df['true'], clip_df['pred'],
            average="binary", zero_division=0)
        print(f"  Clip voting:          "
              f"P={p2:.3f}  R={r2:.3f}  F1={f2:.3f}")

        results.append({
            'ratio':        ratio,
            'n_base_clips': n_base_sample,
            'n_sim_clips':  n_sim,
            'p_default':    p0,
            'r_default':    r0,
            'p_tuned':      p1,
            'r_tuned':      r1,
            'f1_tuned':     f1,
            'p_vote':       p2,
            'r_vote':       r2,
            'best_thresh':  best_thresh,
            'beat_baseline': beat,
            'precisions':   precisions,
            'recalls':      recalls,
        })

    # ── 4. summary table ──────────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("BALANCE EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"{'Ratio':<8} {'Base clips':<12} {'Thresh':<8} "
          f"{'Precision':>10} {'Recall':>8} {'F1':>6} {'Beat?':>8}")
    print("-" * 70)
    for r in results:
        beat_str = "YES" if r['beat_baseline'] else "no"
        print(f"{str(r['ratio'])+':1':<8} {r['n_base_clips']:<12} "
              f"{r['best_thresh']:<8.3f} {r['p_tuned']:>10.3f} "
              f"{r['r_tuned']:>8.3f} {r['f1_tuned']:>6.3f} "
              f"{beat_str:>8}")
    print(f"{'Target':<8} {'-':<12} {'-':<8} {'0.700':>10} "
          f"{'0.700':>8} {'-':>6}")
    print("=" * 70)

    # ── 5. plot precision-recall curves for all ratios ────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#2C5F2D")
    ax.set_facecolor("#2C5F2D")

    colours = ["#97BC62", "#F5A623", "#E05C5C", "#4A90D9"]
    for res, colour in zip(results, colours):
        label = f"ratio {res['ratio']}:1  (P={res['p_tuned']:.3f} R={res['r_tuned']:.3f})"
        ax.plot(res['recalls'], res['precisions'],
                color=colour, linewidth=2, label=label)
        # Mark operating point
        if res['beat_baseline']:
            idx = np.argmin(
                np.abs(np.array(
                    [t for t in np.linspace(0,1,len(res['recalls']))]) -
                    res['best_thresh']))
        ax.scatter(res['r_tuned'], res['p_tuned'],
                   color=colour, s=60, zorder=5)

    ax.axhline(0.70, color="white", linestyle="--",
               linewidth=1.2, alpha=0.7, label="Target precision 0.70")
    ax.axvline(0.70, color="white", linestyle=":",
               linewidth=1.2, alpha=0.7, label="Target recall 0.70")

    ax.tick_params(colors="white")
    ax.set_xlabel("Recall", color="white", fontsize=12)
    ax.set_ylabel("Precision", color="white", fontsize=12)
    ax.set_title("AM2 Balance Experiment\nPrecision-Recall Curves by Baseline:Simulation Ratio",
                 color="white", fontsize=12)
    ax.legend(facecolor="#1F451F", labelcolor="white",
              framealpha=0.8, fontsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#97BC62")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=150, bbox_inches="tight",
                facecolor="#2C5F2D")
    plt.show()
    print(f"\nPlot saved -> {OUT_PLOT}")

    # ── 6. interpretation ─────────────────────────────────────────────────────
    print("\nInterpretation:")
    best_result = max(results, key=lambda x: x['p_tuned'] + x['r_tuned'])
    print(f"  Best ratio: {best_result['ratio']}:1  "
          f"(P={best_result['p_tuned']:.3f}  R={best_result['r_tuned']:.3f})")

    if any(r['beat_baseline'] for r in results):
        winning = [r for r in results if r['beat_baseline']]
        print(f"  Ratios that beat baseline: "
              f"{[r['ratio'] for r in winning]}")
        print("  --> Class balance WAS the problem. "
              "Use the winning ratio for AM2.")
    else:
        print("  No ratio beats baseline.")
        print("  --> Class balance alone is NOT the problem.")
        print("  --> The features need to be adapted for AM2's acoustic character.")
        print("  --> Consider: site-specific training, cross-site pooling,")
        print("      or adding AM2-specific features.")

    # ── 7. save report ────────────────────────────────────────────────────────
    with open(OUT_REPORT, "w", encoding="utf-8") as fh:
        fh.write("AM2 Balance Experiment Results\n")
        fh.write("=" * 50 + "\n\n")
        fh.write(f"Simulation clips: {n_sim}\n")
        fh.write(f"Available baseline clips: {n_base}\n\n")
        fh.write(f"{'Ratio':<8} {'Base clips':<12} {'Thresh':<8} "
                 f"{'Precision':>10} {'Recall':>8} {'F1':>6} {'Beat?':>8}\n")
        fh.write("-" * 60 + "\n")
        for r in results:
            beat_str = "YES" if r['beat_baseline'] else "no"
            fh.write(f"{str(r['ratio'])+':1':<8} {r['n_base_clips']:<12} "
                     f"{r['best_thresh']:<8.3f} {r['p_tuned']:>10.3f} "
                     f"{r['r_tuned']:>8.3f} {r['f1_tuned']:>6.3f} "
                     f"{beat_str:>8}\n")
        fh.write("\n")
        if any(r['beat_baseline'] for r in results):
            fh.write("Class balance WAS the problem.\n")
        else:
            fh.write("Class balance alone is NOT the problem.\n")
            fh.write("Features need site-specific adaptation.\n")

    print(f"Report saved -> {OUT_REPORT}")
    print("\nDone.")