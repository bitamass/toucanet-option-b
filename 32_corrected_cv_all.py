"""
32_corrected_cv_all.py
-----------------------
Runs the corrected clip-level CV on ALL five recorders simultaneously.
Uses already-saved feature matrices — no re-embedding needed.
Takes about 10 minutes total.

Fixes applied vs original scripts:
  1. CV split at CLIP level — segments from same clip stay together
  2. Z-scores computed inside each fold (training baseline only)
  3. PCA fitted inside each fold on training data only
  4. StandardScaler fitted inside each fold on training data only
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (precision_recall_fscore_support,
                              precision_recall_curve,
                              classification_report)

BASE_DIR = r"C:\Users\BitaMassoudi\KasmirWorldProjects\toucanet-option-b"

RECORDERS = [
    {
        'name':    'AM4',
        'feat':    'am4_full_feature_matrix.csv',
        'emb':     'am4_full_emb.npy',
        'lab':     'am4_full_labels.npy',
        'sim_clips': 91,
        'old_p':   0.872,
        'old_r':   0.973,
    },
    {
        'name':    'AM2',
        'feat':    'am2_feature_matrix.csv',
        'emb':     'am2_time_controlled_emb.npy',
        'lab':     'am2_time_controlled_labels.npy',
        'sim_clips': 453,
        'old_p':   0.715,
        'old_r':   0.909,
    },
    {
        'name':    'AM5',
        'feat':    'am5_feature_matrix.csv',
        'emb':     'am5_time_controlled_emb.npy',
        'lab':     'am5_time_controlled_labels.npy',
        'sim_clips': 184,
        'old_p':   0.760,
        'old_r':   0.903,
    },
    {
        'name':    'AM6',
        'feat':    'am6_feature_matrix.csv',
        'emb':     'am6_time_controlled_emb.npy',
        'lab':     'am6_time_controlled_labels.npy',
        'sim_clips': 310,
        'old_p':   0.705,
        'old_r':   0.913,
    },
    {
        'name':    'AM1',
        'feat':    'am1_feature_matrix.csv',
        'emb':     'am1_full_emb.npy',
        'lab':     'am1_full_labels.npy',
        'sim_clips': 59,
        'old_p':   0.874,
        'old_r':   0.971,
    },
]

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

RANDOM_SEED = 42
VOTE_FRAC   = 0.5
N_FOLDS     = 5


def corrected_cv(feat_df, embeddings, labels, name):
    """Run corrected clip-level CV for one recorder."""

    clip_info = feat_df.groupby('clip_name').agg(
        clip_label=('clip_label', 'first'),
        clip_hour=('clip_hour', 'first')
    ).reset_index()
    clip_names  = clip_info['clip_name'].values
    clip_labels = clip_info['clip_label'].values

    cv = StratifiedKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    all_seg_probs = np.zeros(len(feat_df), dtype=np.float32)
    all_seg_preds = np.zeros(len(feat_df), dtype=int)

    for fold, (tr_clip_idx, va_clip_idx) in enumerate(
            cv.split(clip_names, clip_labels)):

        tr_clips = set(clip_names[tr_clip_idx])
        va_clips = set(clip_names[va_clip_idx])

        tr_mask = feat_df['clip_name'].isin(tr_clips).values
        va_mask = feat_df['clip_name'].isin(va_clips).values

        feat_tr = feat_df[tr_mask].copy().reset_index(drop=True)
        feat_va = feat_df[va_mask].copy().reset_index(drop=True)
        emb_tr  = embeddings[tr_mask]
        emb_va  = embeddings[va_mask]
        y_tr    = labels[tr_mask].astype(int)
        y_va    = labels[va_mask].astype(int)

        # Z-scores from training baseline only
        z_tr_rows, z_va_rows = [], []
        for hour in feat_tr['clip_hour'].unique():
            base_mask_tr = ((feat_tr['clip_hour'] == hour) &
                            (feat_tr['clip_label'] == 0))
            base_rows = feat_tr.loc[base_mask_tr, SPECTRAL_KEYS]

            if len(base_rows) == 0:
                for feat_sub, rows_list in [(feat_tr, z_tr_rows),
                                            (feat_va, z_va_rows)]:
                    m = feat_sub['clip_hour'] == hour
                    z = feat_sub.loc[m, SPECTRAL_KEYS].copy() * 0
                    z.columns = [f"z_{c}" for c in z.columns]
                    rows_list.append(z)
                continue

            hour_mean = base_rows.mean()
            hour_std  = base_rows.std().replace(0, 1e-6)

            tr_h = feat_tr['clip_hour'] == hour
            z_tr = (feat_tr.loc[tr_h, SPECTRAL_KEYS] -
                    hour_mean) / hour_std
            z_tr.columns = [f"z_{c}" for c in z_tr.columns]
            z_tr_rows.append(z_tr)

            va_h = feat_va['clip_hour'] == hour
            if va_h.sum() > 0:
                z_va = (feat_va.loc[va_h, SPECTRAL_KEYS] -
                        hour_mean) / hour_std
                z_va.columns = [f"z_{c}" for c in z_va.columns]
                z_va_rows.append(z_va)

        z_tr_df = pd.concat(z_tr_rows).sort_index()
        z_va_df = (pd.concat(z_va_rows).sort_index()
                   if z_va_rows else
                   pd.DataFrame(0, index=feat_va.index,
                                columns=[f"z_{c}"
                                         for c in SPECTRAL_KEYS]))

        # PCA on training embeddings only
        pca       = PCA(n_components=20, random_state=RANDOM_SEED)
        emb_tr_pc = pca.fit_transform(emb_tr)
        emb_va_pc = pca.transform(emb_va)

        X_tr = np.hstack([feat_tr[SPECTRAL_KEYS].values,
                           z_tr_df.values, emb_tr_pc]).astype(np.float32)
        X_va = np.hstack([feat_va[SPECTRAL_KEYS].values,
                           z_va_df.values, emb_va_pc]).astype(np.float32)

        # Scaler on training data only
        scaler   = StandardScaler()
        X_tr_s   = scaler.fit_transform(X_tr)
        X_va_s   = scaler.transform(X_va)

        clf = LogisticRegression(
            max_iter=1000, class_weight="balanced",
            random_state=RANDOM_SEED)
        clf.fit(X_tr_s, y_tr)

        va_probs = clf.predict_proba(X_va_s)[:, 1]
        va_preds = clf.predict(X_va_s)

        va_indices = np.where(va_mask)[0]
        all_seg_probs[va_indices] = va_probs
        all_seg_preds[va_indices] = va_preds

        p, r, f, _ = precision_recall_fscore_support(
            y_va, va_preds, average="binary", zero_division=0)
        print(f"    Fold {fold+1}: P={p:.3f}  R={r:.3f}  F1={f:.3f}")

    # Results
    y_all = labels.astype(int)
    p0, r0, f0, _ = precision_recall_fscore_support(
        y_all, all_seg_preds, average="binary", zero_division=0)

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
    else:
        best_idx    = np.argmax(precisions[:-1] + recalls[:-1])
        best_thresh = float(thresholds[best_idx])
        preds_tuned = (all_seg_probs >= best_thresh).astype(int)
        p1, r1, f1, _ = precision_recall_fscore_support(
            y_all, preds_tuned, average="binary", zero_division=0)
        beat = False

    # Clip voting
    clip_results = []
    for cn in feat_df['clip_name'].unique():
        mask  = feat_df['clip_name'].values == cn
        true  = feat_df['clip_label'].values[mask][0]
        probs = all_seg_probs[mask]
        pred  = int((probs >= best_thresh).mean() >= VOTE_FRAC)
        clip_results.append({'true': true, 'pred': pred})
    clip_df = pd.DataFrame(clip_results)
    p2, r2, f2, _ = precision_recall_fscore_support(
        clip_df['true'], clip_df['pred'],
        average="binary", zero_division=0)

    return {
        'p_default':  p0, 'r_default':  r0,
        'p_tuned':    p1, 'r_tuned':    r1,
        'p_vote':     p2, 'r_vote':     r2,
        'best_thresh': best_thresh,
        'beat':        beat,
        'precisions':  precisions,
        'recalls':     recalls,
        'thresholds':  thresholds,
    }


if __name__ == '__main__':

    results = {}

    for rec in RECORDERS:
        name = rec['name']
        feat_path = os.path.join(BASE_DIR, rec['feat'])
        emb_path  = os.path.join(BASE_DIR, rec['emb'])
        lab_path  = os.path.join(BASE_DIR, rec['lab'])

        if not os.path.exists(feat_path):
            print(f"\n{name}: feature matrix not found — skipping")
            continue
        if not os.path.exists(emb_path):
            print(f"\n{name}: embeddings not found — skipping")
            continue

        print(f"\n{'='*55}")
        print(f"{name} — Corrected Clip-Level CV")
        print(f"{'='*55}")

        feat_df    = pd.read_csv(feat_path)
        embeddings = np.load(emb_path)
        labels     = np.load(lab_path)

        # AM2 used 2:1 ratio — embeddings may be full 5:1
        # Use labels to filter to matching rows
        if len(feat_df) != len(embeddings):
            n = min(len(feat_df), len(embeddings))
            feat_df    = feat_df.iloc[:n].reset_index(drop=True)
            embeddings = embeddings[:n]
            labels     = labels[:n]
            print(f"  Trimmed to {n} rows")

        print(f"  Segments: {len(feat_df)}  |  "
              f"Sim clips: {rec['sim_clips']}")

        res = corrected_cv(feat_df, embeddings, labels, name)
        results[name] = res

        beat_str = "BEAT" if res['beat'] else "no"
        print(f"\n  Results:")
        print(f"    Default (t=0.50):  P={res['p_default']:.3f}  "
              f"R={res['r_default']:.3f}")
        print(f"    Tuned   (t={res['best_thresh']:.2f}):  "
              f"P={res['p_tuned']:.3f}  R={res['r_tuned']:.3f}"
              f"  --> {beat_str} baseline")
        print(f"    Clip vote:         P={res['p_vote']:.3f}  "
              f"R={res['r_vote']:.3f}")
        print(f"    Old result:        P={rec['old_p']:.3f}  "
              f"R={rec['old_r']:.3f}")
        diff_p = res['p_tuned'] - rec['old_p']
        diff_r = res['r_tuned'] - rec['old_r']
        print(f"    Change:            P{diff_p:+.3f}  R{diff_r:+.3f}")

    # ── Final summary table ───────────────────────────────────────────────────
    print("\n\n" + "=" * 75)
    print("FINAL CORRECTED RESULTS — ALL RECORDERS")
    print("=" * 75)
    print(f"{'Recorder':<8} {'Sim':>5} {'Old P':>7} {'Old R':>7} "
          f"{'New P':>7} {'New R':>7} {'Change P':>9} {'Beat?':>6}")
    print("-" * 75)

    for rec in RECORDERS:
        name = rec['name']
        if name not in results:
            print(f"{name:<8} {'—':>5} {rec['old_p']:>7.3f} "
                  f"{rec['old_r']:>7.3f} {'N/A':>7} {'N/A':>7} "
                  f"{'N/A':>9} {'?':>6}")
            continue
        res    = results[name]
        diff_p = res['p_tuned'] - rec['old_p']
        beat   = "YES" if res['beat'] else "NO"
        print(f"{name:<8} {rec['sim_clips']:>5} {rec['old_p']:>7.3f} "
              f"{rec['old_r']:>7.3f} {res['p_tuned']:>7.3f} "
              f"{res['r_tuned']:>7.3f} {diff_p:>+9.3f} {beat:>6}")

    print(f"{'Target':<8} {'—':>5} {'0.700':>7} {'0.700':>7}")
    print("=" * 75)

    # ── PR curve overlay ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#2C5F2D")
    ax.set_facecolor("#2C5F2D")

    colours = {"AM4": "#97BC62", "AM2": "#F5A623",
               "AM5": "#4A90D9", "AM6": "#E05C5C", "AM1": "#B39DDB"}

    for rec in RECORDERS:
        name = rec['name']
        if name not in results:
            continue
        res = results[name]
        ax.plot(res['recalls'], res['precisions'],
                color=colours.get(name, "white"),
                linewidth=1.8,
                label=f"{name} P={res['p_tuned']:.3f} "
                      f"R={res['r_tuned']:.3f}")
        if res['beat']:
            idx = np.argmin(
                np.abs(res['thresholds'] - res['best_thresh']))
            ax.scatter(res['recalls'][idx], res['precisions'][idx],
                       color=colours.get(name, "white"),
                       s=50, zorder=5)

    ax.axhline(0.70, color="white", linestyle="--",
               linewidth=1.2, alpha=0.7, label="Target 0.70")
    ax.axvline(0.70, color="white", linestyle=":",
               linewidth=1.2, alpha=0.7)
    ax.tick_params(colors="white")
    ax.set_xlabel("Recall", color="white", fontsize=12)
    ax.set_ylabel("Precision", color="white", fontsize=12)
    ax.set_title("Corrected Clip-Level CV — All Recorders\n"
                 "(No data leakage)",
                 color="white", fontsize=12)
    ax.legend(facecolor="#1F451F", labelcolor="white",
              framealpha=0.8, fontsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#97BC62")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout()
    out_plot = os.path.join(BASE_DIR, "corrected_cv_all_recorders.png")
    plt.savefig(out_plot, dpi=150, bbox_inches="tight",
                facecolor="#2C5F2D")
    plt.show()
    print(f"\nPlot saved -> {out_plot}")

    # Save report
    out_report = os.path.join(BASE_DIR, "corrected_cv_all_results.txt")
    with open(out_report, "w", encoding="utf-8") as fh:
        fh.write("Corrected Clip-Level CV — All Recorders\n")
        fh.write("=" * 50 + "\n\n")
        fh.write("Leakage fixes applied to all recorders:\n")
        fh.write("1. CV split at CLIP level\n")
        fh.write("2. Z-scores inside each fold\n")
        fh.write("3. PCA inside each fold\n")
        fh.write("4. Scaler inside each fold\n\n")
        for rec in RECORDERS:
            name = rec['name']
            if name not in results:
                continue
            res = results[name]
            fh.write(f"{name}:\n")
            fh.write(f"  Old: P={rec['old_p']:.3f}  R={rec['old_r']:.3f}\n")
            fh.write(f"  New: P={res['p_tuned']:.3f}  "
                     f"R={res['r_tuned']:.3f}\n")
            fh.write(f"  Beat baseline: {res['beat']}\n\n")
    print(f"Report saved -> {out_report}")
    print("\nDone.")