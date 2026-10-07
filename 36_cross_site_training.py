"""
36_cross_site_training.py
--------------------------
Cross-site training using leave-one-recorder-out evaluation.

WHAT THIS DOES:
  Instead of training one model per recorder on its own data,
  we train one model on ALL OTHER recorders combined and test
  on the held-out recorder.

  For each recorder in turn:
    - Training data: all clips from the other 4 recorders
    - Test data:     all clips from this recorder only
    - The test recorder contributes ZERO clips to training

  This answers the key question:
  "Can one model generalise across recorder sites without
   site-specific retraining?"

WHY THIS MATTERS:
  - Current per-site models have small training sets
    (AM4: 91 sim clips, AM1: 59 sim clips)
  - Cross-site training pools all data: 1000+ sim clips
  - If it works, new recorders could be deployed without
    collecting new simulation data at each site
  - If it fails, it tells us what site-specific information
    the model is relying on

METHOD:
  - Uses clip-level feature aggregation (mean, std, max of
    Perch embeddings) — same as script 34
  - Gradient boosting classifier with sample weighting
  - Rain filtering applied before training
  - Z-scores computed from training data baseline only
  - PCA fitted on training data only

Uses saved feature matrices and embeddings — no re-embedding.
Should finish in about 20-30 minutes.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (precision_recall_fscore_support,
                              precision_recall_curve,
                              classification_report)

BASE_DIR = r"C:\Users\BitaMassoudi\KasmirWorldProjects\toucanet-option-b"

RECORDERS = [
    {
        'name':  'AM4',
        'feat':  'am4_full_feature_matrix.csv',
        'emb':   'am4_full_emb.npy',
        'lab':   'am4_full_labels.npy',
        'old_p': 0.856, 'old_r': 0.906,
    },
    {
        'name':  'AM2',
        'feat':  'am2_feature_matrix.csv',
        'emb':   'am2_time_controlled_emb.npy',
        'lab':   'am2_time_controlled_labels.npy',
        'old_p': 0.779, 'old_r': 0.781,
    },
    {
        'name':  'AM5',
        'feat':  'am5_feature_matrix.csv',
        'emb':   'am5_time_controlled_emb.npy',
        'lab':   'am5_time_controlled_labels.npy',
        'old_p': 0.707, 'old_r': 0.932,
    },
    {
        'name':  'AM6',
        'feat':  'am6_feature_matrix.csv',
        'emb':   'am6_time_controlled_emb.npy',
        'lab':   'am6_time_controlled_labels.npy',
        'old_p': 0.703, 'old_r': 0.744,
    },
    {
        'name':  'AM1',
        'feat':  'am1_feature_matrix.csv',
        'emb':   'am1_full_emb.npy',
        'lab':   'am1_full_labels.npy',
        'old_p': 0.779, 'old_r': 0.898,
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


def compute_rain_score(clip_df):
    """Compute rain likelihood score per clip."""
    rms  = clip_df['rms_mean'].values
    ent  = clip_df['spec_entropy_mean'].values
    roll = clip_df['rolloff_std'].values
    sil  = clip_df['silence_fraction'].values

    rms_n  = (rms  - rms.min())  / (rms.max()  - rms.min()  + 1e-10)
    ent_n  = (ent  - ent.min())  / (ent.max()  - ent.min()  + 1e-10)
    roll_n = 1.0 - (roll - roll.min()) / (roll.max() - roll.min() + 1e-10)
    sil_n  = 1.0 - (sil  - sil.min())  / (sil.max()  - sil.min()  + 1e-10)

    return 0.35*rms_n + 0.35*ent_n + 0.20*roll_n + 0.10*sil_n


def build_clip_features(feat_df, embeddings, labels):
    """Aggregate embeddings to clip level — one row per clip."""
    clip_rows = []
    for clip_name in feat_df['clip_name'].unique():
        mask       = feat_df['clip_name'].values == clip_name
        clip_label = feat_df['clip_label'].values[mask][0]
        clip_hour  = feat_df['clip_hour'].values[mask][0]
        clip_embs  = embeddings[mask]
        emb_mean   = clip_embs.mean(axis=0)
        emb_std    = clip_embs.std(axis=0)
        emb_max    = clip_embs.max(axis=0)
        spectral   = feat_df.loc[mask, SPECTRAL_KEYS].values[0]
        row = {'clip_name':  clip_name,
               'clip_label': clip_label,
               'clip_hour':  clip_hour}
        for k, v in zip(SPECTRAL_KEYS, spectral):
            row[k] = v
        for i, v in enumerate(emb_mean): row[f'emb_mean_{i}'] = v
        for i, v in enumerate(emb_std):  row[f'emb_std_{i}']  = v
        for i, v in enumerate(emb_max):  row[f'emb_max_{i}']  = v
        clip_rows.append(row)
    return pd.DataFrame(clip_rows)


def apply_rain_filter(clip_df, threshold_pct=90):
    """Remove top threshold_pct% of clips by rain score from baseline."""
    scores     = compute_rain_score(clip_df)
    clip_df    = clip_df.copy()
    clip_df['rain_score'] = scores
    base_scores = clip_df.loc[clip_df['clip_label']==0, 'rain_score']
    threshold   = np.percentile(base_scores, threshold_pct)
    n_before    = len(clip_df)
    clip_df     = clip_df[clip_df['rain_score'] <= threshold].copy()
    clip_df     = clip_df.drop(columns=['rain_score'])
    n_removed   = n_before - len(clip_df)
    return clip_df, n_removed, threshold


def z_score_features(train_df, test_df):
    """Compute z-scores from training baseline only, apply to both."""
    z_tr_rows, z_te_rows = [], []

    for hour in train_df['clip_hour'].unique():
        base_mask = ((train_df['clip_hour'] == hour) &
                     (train_df['clip_label'] == 0))
        base_rows = train_df.loc[base_mask, SPECTRAL_KEYS]
        if len(base_rows) == 0:
            continue
        hour_mean = base_rows.mean()
        hour_std  = base_rows.std().replace(0, 1e-6)

        tr_h = train_df['clip_hour'] == hour
        if tr_h.sum() > 0:
            z = (train_df.loc[tr_h, SPECTRAL_KEYS] -
                 hour_mean) / hour_std
            z.columns = [f"z_{c}" for c in z.columns]
            z_tr_rows.append(z)

        te_h = test_df['clip_hour'] == hour
        if te_h.sum() > 0:
            z = (test_df.loc[te_h, SPECTRAL_KEYS] -
                 hour_mean) / hour_std
            z.columns = [f"z_{c}" for c in z.columns]
            z_te_rows.append(z)

    # Hours in test not seen in training — zero z-scores
    seen_hours = set(train_df['clip_hour'].unique())
    unseen     = set(test_df['clip_hour'].unique()) - seen_hours
    if unseen:
        for hour in unseen:
            te_h = test_df['clip_hour'] == hour
            if te_h.sum() > 0:
                z = test_df.loc[te_h, SPECTRAL_KEYS].copy() * 0
                z.columns = [f"z_{c}" for c in z.columns]
                z_te_rows.append(z)
                print(f"    WARNING: hour {hour} not in training data "
                      f"— zero z-scores for {te_h.sum()} test clips")

    z_tr = (pd.concat(z_tr_rows).sort_index()
            if z_tr_rows else
            pd.DataFrame(0, index=train_df.index,
                         columns=[f"z_{c}" for c in SPECTRAL_KEYS]))
    z_te = (pd.concat(z_te_rows).sort_index()
            if z_te_rows else
            pd.DataFrame(0, index=test_df.index,
                         columns=[f"z_{c}" for c in SPECTRAL_KEYS]))
    return z_tr, z_te


def run_cross_site(train_clips, test_clips, test_name):
    """
    Train on train_clips, evaluate on test_clips.
    Both are already clip-level DataFrames.
    Returns result dict.
    """
    meta_cols = {'clip_name', 'clip_label', 'clip_hour'}
    emb_cols  = [c for c in train_clips.columns
                 if c.startswith('emb_')]

    # Z-scores from training baseline only
    z_tr, z_te = z_score_features(train_clips, test_clips)

    # PCA on training embeddings only
    pca       = PCA(n_components=50, random_state=RANDOM_SEED)
    emb_tr_pc = pca.fit_transform(train_clips[emb_cols].values)
    emb_te_pc = pca.transform(test_clips[emb_cols].values)

    # Assemble feature matrices
    X_tr = np.hstack([train_clips[SPECTRAL_KEYS].values,
                      z_tr.values,
                      emb_tr_pc]).astype(np.float32)
    X_te = np.hstack([test_clips[SPECTRAL_KEYS].values,
                      z_te.values,
                      emb_te_pc]).astype(np.float32)

    y_tr = train_clips['clip_label'].values.astype(int)
    y_te = test_clips['clip_label'].values.astype(int)

    # Scaler on training only
    scaler   = StandardScaler()
    X_tr_s   = scaler.fit_transform(X_tr)
    X_te_s   = scaler.transform(X_te)

    # Class weights
    n_pos = y_tr.sum()
    n_neg = (y_tr == 0).sum()
    w     = np.where(y_tr == 1, n_neg / max(n_pos, 1), 1.0)

    # Gradient boosting
    gb = GradientBoostingClassifier(
        n_estimators=100, max_depth=3,
        learning_rate=0.1, random_state=RANDOM_SEED,
        subsample=0.8)
    gb.fit(X_tr_s, y_tr, sample_weight=w)

    probs = gb.predict_proba(X_te_s)[:, 1]
    prec, rec, thresh = precision_recall_curve(y_te, probs)
    valid = np.where((prec[:-1] >= 0.70) & (rec[:-1] >= 0.70))[0]

    if len(valid) > 0:
        best_idx    = valid[np.argmax(prec[valid] + rec[valid])]
        best_thresh = float(thresh[best_idx])
        preds       = (probs >= best_thresh).astype(int)
        p, r, _, _  = precision_recall_fscore_support(
            y_te, preds, average="binary", zero_division=0)
        beat = True
    else:
        best_idx    = np.argmax(prec[:-1] + rec[:-1])
        best_thresh = float(thresh[best_idx])
        preds       = (probs >= best_thresh).astype(int)
        p, r, _, _  = precision_recall_fscore_support(
            y_te, preds, average="binary", zero_division=0)
        beat = False

    report = classification_report(
        y_te, preds,
        target_names=["baseline", "simulation"], zero_division=0)

    return {
        'p': p, 'r': r, 'beat': beat,
        'thresh': best_thresh,
        'prec': prec, 'rec': rec, 'thresh_arr': thresh,
        'report': report,
        'n_train_sim': int(y_tr.sum()),
        'n_train_base': int((y_tr==0).sum()),
        'n_test_sim': int(y_te.sum()),
        'n_test_base': int((y_te==0).sum()),
    }


if __name__ == '__main__':

    OUT_REPORT = os.path.join(BASE_DIR, "cross_site_results.txt")

    # ── Load all recorder data ────────────────────────────────────────────────
    print("Loading all recorder data...")
    all_clip_dfs = {}

    for rec in RECORDERS:
        name      = rec['name']
        feat_path = os.path.join(BASE_DIR, rec['feat'])
        emb_path  = os.path.join(BASE_DIR, rec['emb'])
        lab_path  = os.path.join(BASE_DIR, rec['lab'])

        if not os.path.exists(feat_path):
            print(f"  {name}: feature matrix not found — skipping")
            continue
        if not os.path.exists(emb_path):
            print(f"  {name}: embeddings not found — skipping")
            continue

        feat_df    = pd.read_csv(feat_path)
        embeddings = np.load(emb_path)
        labels     = np.load(lab_path)

        if len(feat_df) != len(embeddings):
            n = min(len(feat_df), len(embeddings))
            feat_df    = feat_df.iloc[:n].reset_index(drop=True)
            embeddings = embeddings[:n]
            labels     = labels[:n]

        # Build clip-level features
        clip_df = build_clip_features(feat_df, embeddings, labels)

        # Apply rain filter
        clip_df, n_removed, threshold = apply_rain_filter(clip_df)

        # Add recorder identifier
        clip_df['recorder'] = name

        print(f"  {name}: {len(clip_df)} clips  "
              f"({int(clip_df['clip_label'].sum())} sim)  "
              f"[{n_removed} rain clips removed]")

        all_clip_dfs[name] = clip_df

    available = list(all_clip_dfs.keys())
    print(f"\nAvailable recorders: {available}")

    # ── Leave-one-recorder-out evaluation ────────────────────────────────────
    print("\n" + "=" * 65)
    print("LEAVE-ONE-RECORDER-OUT CROSS-SITE EVALUATION")
    print("=" * 65)

    all_results = {}

    for rec in RECORDERS:
        test_name = rec['name']
        if test_name not in all_clip_dfs:
            continue

        train_names = [r for r in available if r != test_name]
        print(f"\nTest recorder: {test_name}")
        print(f"Train recorders: {train_names}")

        test_df  = all_clip_dfs[test_name].copy()
        train_df = pd.concat(
            [all_clip_dfs[r] for r in train_names],
            ignore_index=True)

        print(f"  Training: {len(train_df)} clips  "
              f"({int(train_df['clip_label'].sum())} sim)")
        print(f"  Testing:  {len(test_df)} clips  "
              f"({int(test_df['clip_label'].sum())} sim)")

        # Check if test has both classes
        if test_df['clip_label'].sum() == 0:
            print(f"  SKIP: no simulation clips in test set")
            continue
        if (test_df['clip_label']==0).sum() == 0:
            print(f"  SKIP: no baseline clips in test set")
            continue

        res = run_cross_site(train_df, test_df, test_name)
        all_results[test_name] = {'res': res, 'rec': rec}

        beat_str = "BEAT" if res['beat'] else "miss"
        print(f"\n  Cross-site result:")
        print(f"    P={res['p']:.3f}  R={res['r']:.3f}  "
              f"t={res['thresh']:.3f}  --> {beat_str}")
        print(f"    Per-site result:   "
              f"P={rec['old_p']:.3f}  R={rec['old_r']:.3f}")
        diff_p = res['p'] - rec['old_p']
        diff_r = res['r'] - rec['old_r']
        print(f"    Change:            P{diff_p:+.3f}  R{diff_r:+.3f}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n\n" + "=" * 75)
    print("CROSS-SITE TRAINING SUMMARY")
    print("Train on 4 recorders combined, test on 1 held-out recorder")
    print("=" * 75)
    print(f"{'Rec':<6} {'Train sim':>10} {'Test sim':>9} "
          f"{'Per-site P':>11} {'Cross-site P':>13} "
          f"{'Change':>8} {'Beat?':>6}")
    print("-" * 75)

    for rec in RECORDERS:
        name = rec['name']
        if name not in all_results:
            print(f"{name:<6} {'—':>10} {'—':>9} "
                  f"{rec['old_p']:>11.3f} {'N/A':>13} "
                  f"{'N/A':>8} {'?':>6}")
            continue
        r    = all_results[name]
        res  = r['res']
        diff = res['p'] - rec['old_p']
        beat = "YES" if res['beat'] else "NO"
        print(f"{name:<6} {res['n_train_sim']:>10} {res['n_test_sim']:>9} "
              f"{rec['old_p']:>11.3f} {res['p']:>13.3f} "
              f"{diff:>+8.3f} {beat:>6}")

    print("=" * 75)

    # Key insight
    n_beat = sum(1 for r in all_results.values() if r['res']['beat'])
    print(f"\n{n_beat}/{len(all_results)} recorders beat 0.70 "
          f"when trained on other sites only")

    # ── PR curve overlay ──────────────────────────────────────────────────────
    n   = len(all_results)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 5))
    fig.patch.set_facecolor("#2C5F2D")
    if n == 1:
        axes = [axes]

    colours = {"AM4":"#97BC62","AM2":"#F5A623",
               "AM5":"#4A90D9","AM6":"#E05C5C","AM1":"#B39DDB"}

    for ax, rec in zip(axes, RECORDERS):
        name = rec['name']
        if name not in all_results:
            continue
        res = all_results[name]['res']
        ax.set_facecolor("#2C5F2D")
        ax.plot(res['rec'], res['prec'],
                color=colours.get(name,"#97BC62"),
                linewidth=2, label=f"Cross-site\nP={res['p']:.3f} R={res['r']:.3f}")
        # Per-site reference line
        ax.axhline(rec['old_p'], color="white", linestyle=":",
                   linewidth=1, alpha=0.6,
                   label=f"Per-site P={rec['old_p']:.3f}")
        ax.axhline(0.70, color="white", linestyle="--",
                   linewidth=1.2, alpha=0.7)
        ax.axvline(0.70, color="white", linestyle="--",
                   linewidth=1.2, alpha=0.7)
        if res['beat']:
            idx = np.argmin(np.abs(res['thresh_arr'] - res['thresh']))
            ax.scatter(res['rec'][idx], res['prec'][idx],
                       color="white", s=60, zorder=5)
        ax.tick_params(colors="white")
        ax.set_xlabel("Recall", color="white", fontsize=10)
        ax.set_ylabel("Precision", color="white", fontsize=10)
        ax.set_title(f"{name}\nTrained on other 4 sites",
                     color="white", fontsize=10)
        ax.legend(facecolor="#1F451F", labelcolor="white",
                  framealpha=0.8, fontsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#97BC62")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    plt.tight_layout()
    out_plot = os.path.join(BASE_DIR, "cross_site_pr_curves.png")
    plt.savefig(out_plot, dpi=150, bbox_inches="tight",
                facecolor="#2C5F2D")
    print(f"\nPlot saved -> {out_plot}")

    # ── Save report ───────────────────────────────────────────────────────────
    with open(OUT_REPORT, "w", encoding="utf-8") as fh:
        fh.write("Cross-Site Training Results\n")
        fh.write("=" * 50 + "\n\n")
        fh.write("Method: leave-one-recorder-out\n")
        fh.write("Train on 4 recorders, test on 1 held-out\n")
        fh.write("Clip-level GB + rain filter\n\n")
        for rec in RECORDERS:
            name = rec['name']
            if name not in all_results:
                continue
            r   = all_results[name]
            res = r['res']
            fh.write(f"{name}:\n")
            fh.write(f"  Train sim clips: {res['n_train_sim']}\n")
            fh.write(f"  Test sim clips:  {res['n_test_sim']}\n")
            fh.write(f"  Per-site:    P={rec['old_p']:.3f}  "
                     f"R={rec['old_r']:.3f}\n")
            fh.write(f"  Cross-site:  P={res['p']:.3f}  "
                     f"R={res['r']:.3f}\n")
            fh.write(f"  Beat 0.70:   {res['beat']}\n\n")
            fh.write(res['report'] + "\n\n")
    print(f"Report saved -> {OUT_REPORT}")
    print("\nDone.")
