"""
34_clip_level_features.py
--------------------------
Clip-level feature aggregation — the fundamental fix for AM5.

WHY THIS IS DIFFERENT FROM PREVIOUS SCRIPTS:
  Previous approach:
    - Compute spectral features from 3-sec clip → replicate across all segments
    - One row per 0.2-sec segment → 8-10 identical rows per clip
    - This caused data leakage (82/106 features identical within a clip)

  This approach:
    - Aggregate Perch embeddings across ALL segments of a clip
    - Mean + std + max of 1536-dim vector = 4608 features per clip
    - Add clip-level spectral features (no replication needed)
    - ONE ROW PER CLIP — leakage impossible by design
    - CV splits at clip level naturally

EXPECTED BENEFIT:
  - No leakage by design
  - Richer representation (aggregated embedding captures full clip acoustics)
  - Cleaner dataset — one row per clip makes class balance straightforward
  - Should help AM5 most — where ratio tuning failed completely

Runs on all 5 recorders using saved .npy embedding files.
No re-embedding needed. Should finish in ~15 minutes.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (precision_recall_fscore_support,
                              precision_recall_curve,
                              classification_report)

BASE_DIR = r"C:\Users\BitaMassoudi\KasmirWorldProjects\toucanet-option-b"

RECORDERS = [
    {
        'name':     'AM4',
        'feat':     'am4_full_feature_matrix.csv',
        'emb':      'am4_full_emb.npy',
        'lab':      'am4_full_labels.npy',
        'old_p':    0.724, 'old_r': 0.903,
        'sim_clips': 91,
    },
    {
        'name':     'AM2',
        'feat':     'am2_feature_matrix.csv',
        'emb':      'am2_time_controlled_emb.npy',
        'lab':      'am2_time_controlled_labels.npy',
        'old_p':    0.750, 'old_r': 0.946,
        'sim_clips': 453,
    },
    {
        'name':     'AM5',
        'feat':     'am5_feature_matrix.csv',
        'emb':      'am5_time_controlled_emb.npy',
        'lab':      'am5_time_controlled_labels.npy',
        'old_p':    0.574, 'old_r': 0.964,
        'sim_clips': 184,
    },
    {
        'name':     'AM6',
        'feat':     'am6_feature_matrix.csv',
        'emb':      'am6_time_controlled_emb.npy',
        'lab':      'am6_time_controlled_labels.npy',
        'old_p':    0.700, 'old_r': 0.846,
        'sim_clips': 310,
    },
    {
        'name':     'AM1',
        'feat':     'am1_feature_matrix.csv',
        'emb':      'am1_full_emb.npy',
        'lab':      'am1_full_labels.npy',
        'old_p':    0.717, 'old_r': 0.723,
        'sim_clips': 59,
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
N_FOLDS     = 5


def build_clip_features(feat_df, embeddings, labels):
    """
    Aggregate segment-level embeddings to clip level.
    Returns clip_df with one row per clip containing:
      - Aggregated Perch embedding (mean, std, max across segments)
      - Clip-level spectral features (already at clip level)
      - clip_label, clip_hour
    """
    clip_rows = []

    for clip_name in feat_df['clip_name'].unique():
        mask = feat_df['clip_name'].values == clip_name
        clip_label = feat_df['clip_label'].values[mask][0]
        clip_hour  = feat_df['clip_hour'].values[mask][0]

        # Aggregate embeddings across all segments of this clip
        clip_embs = embeddings[mask]  # shape: (n_segments, 1536)

        emb_mean = clip_embs.mean(axis=0)   # 1536 features
        emb_std  = clip_embs.std(axis=0)    # 1536 features
        emb_max  = clip_embs.max(axis=0)    # 1536 features

        # Spectral features are identical across segments — just take first
        spectral = feat_df.loc[mask, SPECTRAL_KEYS].values[0]

        row = {
            'clip_name':  clip_name,
            'clip_label': clip_label,
            'clip_hour':  clip_hour,
            'n_segments': int(mask.sum()),
        }
        # Add spectral features
        for k, v in zip(SPECTRAL_KEYS, spectral):
            row[k] = v
        # Add aggregated embeddings
        for i, v in enumerate(emb_mean):
            row[f'emb_mean_{i}'] = v
        for i, v in enumerate(emb_std):
            row[f'emb_std_{i}'] = v
        for i, v in enumerate(emb_max):
            row[f'emb_max_{i}'] = v

        clip_rows.append(row)

    return pd.DataFrame(clip_rows)


def run_corrected_cv_clip_level(clip_df, name):
    """
    Run corrected CV on clip-level features.
    Each row is one clip — no leakage possible.
    Z-scores, PCA, scaler all fitted inside each fold.
    Tests both logistic regression and gradient boosting.
    """
    clip_names  = clip_df['clip_name'].values
    clip_labels = clip_df['clip_label'].values

    # Feature columns: spectral + aggregated embeddings
    meta_cols    = {'clip_name', 'clip_label', 'clip_hour', 'n_segments'}
    feature_cols = [c for c in clip_df.columns if c not in meta_cols]

    print(f"  Clips: {len(clip_df)}  |  "
          f"Sim: {clip_labels.sum()}  |  "
          f"Base: {(clip_labels==0).sum()}")
    print(f"  Features per clip: {len(feature_cols)}")

    cv = StratifiedKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    # Storage for both classifiers
    results = {}
    for clf_name in ['logistic', 'gradient_boost']:
        results[clf_name] = {
            'probs': np.zeros(len(clip_df), dtype=np.float32),
            'preds': np.zeros(len(clip_df), dtype=int),
        }

    for fold, (tr_idx, va_idx) in enumerate(
            cv.split(clip_names, clip_labels)):

        clip_tr = clip_df.iloc[tr_idx].copy().reset_index(drop=True)
        clip_va = clip_df.iloc[va_idx].copy().reset_index(drop=True)
        y_tr    = clip_labels[tr_idx].astype(int)
        y_va    = clip_labels[va_idx].astype(int)

        # Z-score normalization on spectral features
        # Using training baseline clips only
        z_tr_rows, z_va_rows = [], []
        for hour in clip_tr['clip_hour'].unique():
            base_mask = ((clip_tr['clip_hour'] == hour) &
                         (clip_tr['clip_label'] == 0))
            base_rows = clip_tr.loc[base_mask, SPECTRAL_KEYS]

            if len(base_rows) == 0:
                for df_sub, rows_list in [(clip_tr, z_tr_rows),
                                          (clip_va, z_va_rows)]:
                    m = df_sub['clip_hour'] == hour
                    if m.sum() > 0:
                        z = df_sub.loc[m, SPECTRAL_KEYS].copy() * 0
                        z.columns = [f"z_{c}" for c in z.columns]
                        rows_list.append(z)
                continue

            hour_mean = base_rows.mean()
            hour_std  = base_rows.std().replace(0, 1e-6)

            tr_h = clip_tr['clip_hour'] == hour
            if tr_h.sum() > 0:
                z_tr = (clip_tr.loc[tr_h, SPECTRAL_KEYS] -
                        hour_mean) / hour_std
                z_tr.columns = [f"z_{c}" for c in z_tr.columns]
                z_tr_rows.append(z_tr)

            va_h = clip_va['clip_hour'] == hour
            if va_h.sum() > 0:
                z_va = (clip_va.loc[va_h, SPECTRAL_KEYS] -
                        hour_mean) / hour_std
                z_va.columns = [f"z_{c}" for c in z_va.columns]
                z_va_rows.append(z_va)

        z_tr_df = (pd.concat(z_tr_rows).sort_index()
                   if z_tr_rows else
                   pd.DataFrame(0, index=clip_tr.index,
                                columns=[f"z_{c}" for c in SPECTRAL_KEYS]))
        z_va_df = (pd.concat(z_va_rows).sort_index()
                   if z_va_rows else
                   pd.DataFrame(0, index=clip_va.index,
                                columns=[f"z_{c}" for c in SPECTRAL_KEYS]))

        # PCA on aggregated embeddings — training only
        emb_cols = [c for c in feature_cols if c.startswith('emb_')]
        pca      = PCA(n_components=min(50, len(emb_cols)//10),
                       random_state=RANDOM_SEED)
        emb_tr_pc = pca.fit_transform(clip_tr[emb_cols].values)
        emb_va_pc = pca.transform(clip_va[emb_cols].values)

        # Assemble feature matrices
        X_tr = np.hstack([
            clip_tr[SPECTRAL_KEYS].values,
            z_tr_df.values,
            emb_tr_pc
        ]).astype(np.float32)

        X_va = np.hstack([
            clip_va[SPECTRAL_KEYS].values,
            z_va_df.values,
            emb_va_pc
        ]).astype(np.float32)

        # Scaler on training only
        scaler   = StandardScaler()
        X_tr_s   = scaler.fit_transform(X_tr)
        X_va_s   = scaler.transform(X_va)

        # Logistic regression
        lr = LogisticRegression(
            max_iter=2000, class_weight="balanced",
            random_state=RANDOM_SEED)
        lr.fit(X_tr_s, y_tr)
        results['logistic']['probs'][va_idx] = (
            lr.predict_proba(X_va_s)[:, 1])
        results['logistic']['preds'][va_idx] = lr.predict(X_va_s)

        # Gradient boosting
        gb = GradientBoostingClassifier(
            n_estimators=100, max_depth=3,
            learning_rate=0.1, random_state=RANDOM_SEED,
            subsample=0.8)
        # For imbalanced data, use sample weights
        n_pos = y_tr.sum()
        n_neg = (y_tr == 0).sum()
        w = np.where(y_tr == 1, n_neg / max(n_pos, 1), 1.0)
        gb.fit(X_tr_s, y_tr, sample_weight=w)
        results['gradient_boost']['probs'][va_idx] = (
            gb.predict_proba(X_va_s)[:, 1])
        results['gradient_boost']['preds'][va_idx] = gb.predict(X_va_s)

        p_lr, r_lr, _, _ = precision_recall_fscore_support(
            y_va, results['logistic']['preds'][va_idx],
            average="binary", zero_division=0)
        p_gb, r_gb, _, _ = precision_recall_fscore_support(
            y_va, results['gradient_boost']['preds'][va_idx],
            average="binary", zero_division=0)
        print(f"    Fold {fold+1}: "
              f"LR P={p_lr:.3f} R={r_lr:.3f}  |  "
              f"GB P={p_gb:.3f} R={r_gb:.3f}")

    y_all = clip_labels.astype(int)
    out   = {}

    for clf_name, res in results.items():
        prec, rec, thresh = precision_recall_curve(y_all, res['probs'])
        valid = np.where((prec[:-1] >= 0.70) & (rec[:-1] >= 0.70))[0]

        if len(valid) > 0:
            best_idx    = valid[np.argmax(prec[valid] + rec[valid])]
            best_thresh = float(thresh[best_idx])
            preds_t     = (res['probs'] >= best_thresh).astype(int)
            p1, r1, _, _ = precision_recall_fscore_support(
                y_all, preds_t, average="binary", zero_division=0)
            beat = True
        else:
            best_idx    = np.argmax(prec[:-1] + rec[:-1])
            best_thresh = float(thresh[best_idx])
            preds_t     = (res['probs'] >= best_thresh).astype(int)
            p1, r1, _, _ = precision_recall_fscore_support(
                y_all, preds_t, average="binary", zero_division=0)
            beat = False

        out[clf_name] = {
            'p': p1, 'r': r1, 'thresh': best_thresh,
            'beat': beat, 'prec': prec, 'rec': rec, 'thresh_arr': thresh
        }

    return out


if __name__ == '__main__':

    all_results = {}
    OUT_REPORT  = os.path.join(BASE_DIR, "clip_level_results.txt")

    for rec in RECORDERS:
        name = rec['name']
        print(f"\n{'='*60}")
        print(f"{name} — Clip-Level Feature Aggregation")
        print(f"{'='*60}")

        feat_path = os.path.join(BASE_DIR, rec['feat'])
        emb_path  = os.path.join(BASE_DIR, rec['emb'])
        lab_path  = os.path.join(BASE_DIR, rec['lab'])

        if not os.path.exists(feat_path):
            print(f"  Skipping — feature matrix not found")
            continue
        if not os.path.exists(emb_path):
            print(f"  Skipping — embeddings not found")
            continue

        feat_df    = pd.read_csv(feat_path)
        embeddings = np.load(emb_path)
        labels     = np.load(lab_path)

        if len(feat_df) != len(embeddings):
            n = min(len(feat_df), len(embeddings))
            feat_df    = feat_df.iloc[:n].reset_index(drop=True)
            embeddings = embeddings[:n]
            labels     = labels[:n]

        print(f"  Building clip-level features...")
        clip_df = build_clip_features(feat_df, embeddings, labels)
        print(f"  Built {len(clip_df)} clip-level rows")

        out = run_corrected_cv_clip_level(clip_df, name)
        all_results[name] = {'out': out, 'rec': rec}

        print(f"\n  Results:")
        for clf_name, res in out.items():
            beat_str = "BEAT" if res['beat'] else "miss"
            print(f"    {clf_name:<20}  "
                  f"P={res['p']:.3f}  R={res['r']:.3f}  "
                  f"t={res['thresh']:.3f}  {beat_str}")
        print(f"    Previous (segment):   "
              f"P={rec['old_p']:.3f}  R={rec['old_r']:.3f}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n\n" + "=" * 75)
    print("CLIP-LEVEL AGGREGATION — SUMMARY")
    print("=" * 75)
    print(f"{'Recorder':<8} {'Old P':>7} {'LR P':>7} "
          f"{'LR R':>7} {'LR Beat':>8} {'GB P':>7} "
          f"{'GB R':>7} {'GB Beat':>8}")
    print("-" * 75)

    for rec in RECORDERS:
        name = rec['name']
        if name not in all_results:
            continue
        res = all_results[name]
        lr  = res['out']['logistic']
        gb  = res['out']['gradient_boost']
        print(f"{name:<8} {rec['old_p']:>7.3f} "
              f"{lr['p']:>7.3f} {lr['r']:>7.3f} "
              f"{'YES' if lr['beat'] else 'no':>8} "
              f"{gb['p']:>7.3f} {gb['r']:>7.3f} "
              f"{'YES' if gb['beat'] else 'no':>8}")

    print(f"{'Target':<8} {'0.700':>7} {'0.700':>7} {'0.700':>7}")
    print("=" * 75)

    # ── PR curves ─────────────────────────────────────────────────────────────
    n   = len(all_results)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
    fig.patch.set_facecolor("#2C5F2D")
    if n == 1:
        axes = [axes]

    cols = {'logistic': "#97BC62", 'gradient_boost': "#F5A623"}

    for ax, (name, res_dict) in zip(axes, all_results.items()):
        ax.set_facecolor("#2C5F2D")
        for clf_name, res in res_dict['out'].items():
            lbl = (f"{'LR' if clf_name=='logistic' else 'GB'}  "
                   f"P={res['p']:.3f} R={res['r']:.3f}")
            ax.plot(res['rec'], res['prec'],
                    color=cols[clf_name], linewidth=1.8, label=lbl)
            if res['beat']:
                idx = np.argmin(
                    np.abs(res['thresh_arr'] - res['thresh']))
                ax.scatter(res['rec'][idx], res['prec'][idx],
                           color=cols[clf_name], s=60, zorder=5)

        ax.axhline(0.70, color="white", linestyle="--",
                   linewidth=1.2, alpha=0.7)
        ax.axvline(0.70, color="white", linestyle=":",
                   linewidth=1.2, alpha=0.7)
        ax.tick_params(colors="white")
        ax.set_xlabel("Recall", color="white", fontsize=11)
        ax.set_ylabel("Precision", color="white", fontsize=11)
        ax.set_title(f"{name} — Clip-Level Features",
                     color="white", fontsize=11)
        ax.legend(facecolor="#1F451F", labelcolor="white",
                  framealpha=0.8, fontsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#97BC62")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    plt.tight_layout()
    out_plot = os.path.join(BASE_DIR, "clip_level_pr_curves.png")
    plt.savefig(out_plot, dpi=150, bbox_inches="tight",
                facecolor="#2C5F2D")
    print(f"\nPlot saved -> {out_plot}")

    # ── Save report ───────────────────────────────────────────────────────────
    with open(OUT_REPORT, "w", encoding="utf-8") as fh:
        fh.write("Clip-Level Feature Aggregation Results\n")
        fh.write("=" * 50 + "\n\n")
        fh.write("Method: aggregate Perch embeddings across all segments\n")
        fh.write("per clip (mean, std, max of 1536-dim vector).\n")
        fh.write("One row per clip — leakage impossible by design.\n\n")
        for rec in RECORDERS:
            name = rec['name']
            if name not in all_results:
                continue
            res = all_results[name]
            fh.write(f"{name}:\n")
            fh.write(f"  Previous (segment-level): "
                     f"P={rec['old_p']:.3f}  R={rec['old_r']:.3f}\n")
            for clf_name, r in res['out'].items():
                fh.write(f"  {clf_name:<20}: "
                         f"P={r['p']:.3f}  R={r['r']:.3f}  "
                         f"{'BEAT' if r['beat'] else 'miss'}\n")
            fh.write("\n")
    print(f"Report saved -> {OUT_REPORT}")
    print("\nDone.")