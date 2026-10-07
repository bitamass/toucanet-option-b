"""
35_rain_filter.py
------------------
Detects and removes rain-dominated clips from training and testing.

WHY RAIN IS A PROBLEM:
  Rain creates broadband noise — high energy spread uniformly across
  all frequencies. This looks acoustically unusual compared to the
  hourly baseline (which is mostly bird calls and insects with energy
  concentrated in specific frequency bands).

  Result: rain clips get flagged as disturbances even when nothing
  is happening. This drives down precision, especially at AM6 which
  is most affected by weather.

HOW WE DETECT RAIN:
  Three complementary acoustic signatures of rain:

  1. HIGH BROADBAND ENERGY — RMS energy is elevated
     Rain raises the overall energy level of the recording.

  2. LOW SPECTRAL CONTRAST — energy spread evenly across frequencies
     Bird calls and insects have energy concentrated in bands.
     Rain spreads energy uniformly. We measure this as:
     spectral flatness = geometric mean / arithmetic mean of spectrum
     Values near 1.0 = flat (rain-like)
     Values near 0.0 = peaked (bird-call-like)

  3. HIGH ZERO-CROSSING RATE VARIANCE — rain creates irregular noise
     Rain creates many rapid fluctuations in amplitude, leading to
     high and variable zero-crossing rates.

  We combine all three into a rain score and flag clips above a
  threshold as rain-dominated.

WHAT IT DOES:
  1. Loads each recorder's feature matrix
  2. Computes rain scores from existing spectral features
  3. Shows the distribution and suggests a threshold
  4. Removes rain clips and reruns the clip-level GB classifier
  5. Compares results before and after filtering

Uses saved feature matrices — no re-embedding needed.
Should finish in about 10-15 minutes for all recorders.
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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (precision_recall_fscore_support,
                              precision_recall_curve)

BASE_DIR = r"C:\Users\BitaMassoudi\KasmirWorldProjects\toucanet-option-b"

RECORDERS = [
    {
        'name':    'AM4',
        'feat':    'am4_full_feature_matrix.csv',
        'emb':     'am4_full_emb.npy',
        'lab':     'am4_full_labels.npy',
        'old_p':   0.841, 'old_r': 0.813,
    },
    {
        'name':    'AM2',
        'feat':    'am2_feature_matrix.csv',
        'emb':     'am2_time_controlled_emb.npy',
        'lab':     'am2_time_controlled_labels.npy',
        'old_p':   0.772, 'old_r': 0.801,
    },
    {
        'name':    'AM5',
        'feat':    'am5_feature_matrix.csv',
        'emb':     'am5_time_controlled_emb.npy',
        'lab':     'am5_time_controlled_labels.npy',
        'old_p':   0.708, 'old_r': 0.870,
    },
    {
        'name':    'AM6',
        'feat':    'am6_feature_matrix.csv',
        'emb':     'am6_time_controlled_emb.npy',
        'lab':     'am6_time_controlled_labels.npy',
        'old_p':   0.705, 'old_r': 0.700,
    },
    {
        'name':    'AM1',
        'feat':    'am1_feature_matrix.csv',
        'emb':     'am1_full_emb.npy',
        'lab':     'am1_full_labels.npy',
        'old_p':   0.785, 'old_r': 0.864,
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


def compute_rain_score(feat_df):
    """
    Compute a rain likelihood score for each clip.
    Score is a weighted combination of three rain signatures.
    Returns a Series with one score per segment row,
    and a clip-level score Series.
    """
    # Get one row per clip (features are identical within clip)
    clip_df = feat_df.groupby('clip_name').first().reset_index()

    # Signature 1: High RMS energy (normalised to 0-1 within dataset)
    rms = clip_df['rms_mean'].values
    rms_norm = (rms - rms.min()) / (rms.max() - rms.min() + 1e-10)

    # Signature 2: High spectral entropy = flat spectrum = rain-like
    # spec_entropy_mean is already computed in our features
    ent = clip_df['spec_entropy_mean'].values
    ent_norm = (ent - ent.min()) / (ent.max() - ent.min() + 1e-10)

    # Signature 3: Low spectral rolloff std = uniform energy over time
    # (rain is steady, bird calls are variable)
    rolloff_std = clip_df['rolloff_std'].values
    # Low rolloff_std = more rain-like, so invert
    rolloff_std_norm = 1.0 - (
        (rolloff_std - rolloff_std.min()) /
        (rolloff_std.max() - rolloff_std.min() + 1e-10))

    # Signature 4: Low silence fraction
    # Rain clips have very little silence
    silence = clip_df['silence_fraction'].values
    # Low silence = more rain-like, so invert
    silence_norm = 1.0 - (
        (silence - silence.min()) /
        (silence.max() - silence.min() + 1e-10))

    # Combined score — weighted average
    rain_score = (0.35 * rms_norm +
                  0.35 * ent_norm +
                  0.20 * rolloff_std_norm +
                  0.10 * silence_norm)

    clip_df['rain_score'] = rain_score
    return clip_df[['clip_name', 'rain_score', 'clip_label']]


def build_clip_features(feat_df, embeddings, labels):
    """Aggregate embeddings to clip level."""
    clip_rows = []
    for clip_name in feat_df['clip_name'].unique():
        mask = feat_df['clip_name'].values == clip_name
        clip_label = feat_df['clip_label'].values[mask][0]
        clip_hour  = feat_df['clip_hour'].values[mask][0]
        clip_embs  = embeddings[mask]
        emb_mean   = clip_embs.mean(axis=0)
        emb_std    = clip_embs.std(axis=0)
        emb_max    = clip_embs.max(axis=0)
        spectral   = feat_df.loc[mask, SPECTRAL_KEYS].values[0]
        row = {'clip_name': clip_name, 'clip_label': clip_label,
               'clip_hour': clip_hour}
        for k, v in zip(SPECTRAL_KEYS, spectral):
            row[k] = v
        for i, v in enumerate(emb_mean):
            row[f'emb_mean_{i}'] = v
        for i, v in enumerate(emb_std):
            row[f'emb_std_{i}'] = v
        for i, v in enumerate(emb_max):
            row[f'emb_max_{i}'] = v
        clip_rows.append(row)
    return pd.DataFrame(clip_rows)


def run_gb_clip_cv(clip_df):
    """Run gradient boosting clip-level CV with corrected splits."""
    clip_names  = clip_df['clip_name'].values
    clip_labels = clip_df['clip_label'].values
    meta_cols   = {'clip_name', 'clip_label', 'clip_hour'}
    feat_cols   = [c for c in clip_df.columns if c not in meta_cols]

    if len(np.unique(clip_labels)) < 2:
        return None
    if clip_labels.sum() < N_FOLDS:
        return None

    cv    = StratifiedKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    probs = np.zeros(len(clip_df), dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(
            cv.split(clip_names, clip_labels)):

        clip_tr = clip_df.iloc[tr_idx].copy().reset_index(drop=True)
        clip_va = clip_df.iloc[va_idx].copy().reset_index(drop=True)
        y_tr    = clip_labels[tr_idx].astype(int)
        y_va    = clip_labels[va_idx].astype(int)

        # Z-scores from training baseline only
        z_tr_rows, z_va_rows = [], []
        for hour in clip_tr['clip_hour'].unique():
            base_mask = ((clip_tr['clip_hour'] == hour) &
                         (clip_tr['clip_label'] == 0))
            base_rows = clip_tr.loc[base_mask, SPECTRAL_KEYS]
            if len(base_rows) == 0:
                continue
            hour_mean = base_rows.mean()
            hour_std  = base_rows.std().replace(0, 1e-6)
            tr_h = clip_tr['clip_hour'] == hour
            if tr_h.sum() > 0:
                z = (clip_tr.loc[tr_h, SPECTRAL_KEYS] -
                     hour_mean) / hour_std
                z.columns = [f"z_{c}" for c in z.columns]
                z_tr_rows.append(z)
            va_h = clip_va['clip_hour'] == hour
            if va_h.sum() > 0:
                z = (clip_va.loc[va_h, SPECTRAL_KEYS] -
                     hour_mean) / hour_std
                z.columns = [f"z_{c}" for c in z.columns]
                z_va_rows.append(z)

        z_tr_df = (pd.concat(z_tr_rows).sort_index()
                   if z_tr_rows else
                   pd.DataFrame(0, index=clip_tr.index,
                                columns=[f"z_{c}"
                                         for c in SPECTRAL_KEYS]))
        z_va_df = (pd.concat(z_va_rows).sort_index()
                   if z_va_rows else
                   pd.DataFrame(0, index=clip_va.index,
                                columns=[f"z_{c}"
                                         for c in SPECTRAL_KEYS]))

        emb_cols  = [c for c in feat_cols if c.startswith('emb_')]
        pca       = PCA(n_components=min(50, len(emb_cols)//10),
                        random_state=RANDOM_SEED)
        emb_tr_pc = pca.fit_transform(clip_tr[emb_cols].values)
        emb_va_pc = pca.transform(clip_va[emb_cols].values)

        X_tr = np.hstack([clip_tr[SPECTRAL_KEYS].values,
                           z_tr_df.values,
                           emb_tr_pc]).astype(np.float32)
        X_va = np.hstack([clip_va[SPECTRAL_KEYS].values,
                           z_va_df.values,
                           emb_va_pc]).astype(np.float32)

        scaler   = StandardScaler()
        X_tr_s   = scaler.fit_transform(X_tr)
        X_va_s   = scaler.transform(X_va)

        n_pos = y_tr.sum()
        n_neg = (y_tr == 0).sum()
        w = np.where(y_tr == 1, n_neg / max(n_pos, 1), 1.0)

        gb = GradientBoostingClassifier(
            n_estimators=100, max_depth=3,
            learning_rate=0.1, random_state=RANDOM_SEED,
            subsample=0.8)
        gb.fit(X_tr_s, y_tr, sample_weight=w)
        probs[va_idx] = gb.predict_proba(X_va_s)[:, 1]

    y_all = clip_labels.astype(int)
    prec, rec, thresh = precision_recall_curve(y_all, probs)
    valid = np.where((prec[:-1] >= 0.70) & (rec[:-1] >= 0.70))[0]

    if len(valid) > 0:
        best_idx    = valid[np.argmax(prec[valid] + rec[valid])]
        best_thresh = float(thresh[best_idx])
        preds       = (probs >= best_thresh).astype(int)
        p, r, _, _  = precision_recall_fscore_support(
            y_all, preds, average="binary", zero_division=0)
        beat = True
    else:
        best_idx    = np.argmax(prec[:-1] + rec[:-1])
        best_thresh = float(thresh[best_idx])
        preds       = (probs >= best_thresh).astype(int)
        p, r, _, _  = precision_recall_fscore_support(
            y_all, preds, average="binary", zero_division=0)
        beat = False

    return {'p': p, 'r': r, 'beat': beat,
            'thresh': best_thresh, 'prec': prec, 'rec': rec,
            'thresh_arr': thresh}


if __name__ == '__main__':

    OUT_REPORT = os.path.join(BASE_DIR, "rain_filter_results.txt")
    all_results = {}

    for rec in RECORDERS:
        name = rec['name']
        print(f"\n{'='*60}")
        print(f"{name} — Rain Filtering")
        print(f"{'='*60}")

        feat_path = os.path.join(BASE_DIR, rec['feat'])
        emb_path  = os.path.join(BASE_DIR, rec['emb'])
        lab_path  = os.path.join(BASE_DIR, rec['lab'])

        if not os.path.exists(feat_path):
            print(f"  Skipping — feature matrix not found")
            continue

        feat_df    = pd.read_csv(feat_path)
        embeddings = np.load(emb_path)
        labels     = np.load(lab_path)

        if len(feat_df) != len(embeddings):
            n = min(len(feat_df), len(embeddings))
            feat_df    = feat_df.iloc[:n].reset_index(drop=True)
            embeddings = embeddings[:n]
            labels     = labels[:n]

        # ── Compute rain scores ───────────────────────────────────────────
        rain_df = compute_rain_score(feat_df)

        # Show distribution
        baseline_rain = rain_df[rain_df['clip_label']==0]['rain_score']
        sim_rain      = rain_df[rain_df['clip_label']==1]['rain_score']

        print(f"  Rain score distribution:")
        print(f"    Baseline — mean={baseline_rain.mean():.3f}  "
              f"p75={baseline_rain.quantile(0.75):.3f}  "
              f"p90={baseline_rain.quantile(0.90):.3f}  "
              f"max={baseline_rain.max():.3f}")
        print(f"    Simulation — mean={sim_rain.mean():.3f}  "
              f"p75={sim_rain.quantile(0.75):.3f}  "
              f"max={sim_rain.max():.3f}")

        # Use 90th percentile of BASELINE as threshold
        # Clips above this are unusually noisy even for baseline
        threshold = baseline_rain.quantile(0.90)
        print(f"  Rain threshold (baseline p90): {threshold:.3f}")

        # Count what gets removed
        rain_clips = rain_df[rain_df['rain_score'] > threshold]['clip_name']
        n_rain_base = ((rain_df['rain_score'] > threshold) &
                       (rain_df['clip_label'] == 0)).sum()
        n_rain_sim  = ((rain_df['rain_score'] > threshold) &
                       (rain_df['clip_label'] == 1)).sum()
        n_total_clips = len(rain_df)

        print(f"  Clips flagged as rain: {len(rain_clips)} "
              f"({len(rain_clips)/n_total_clips:.1%} of total)")
        print(f"    Baseline clips removed: {n_rain_base}")
        print(f"    Simulation clips removed: {n_rain_sim}")

        if n_rain_sim > 0:
            pct_sim_lost = n_rain_sim / len(sim_rain) * 100
            print(f"    WARNING: {pct_sim_lost:.1f}% of simulation "
                  f"clips flagged as rain — threshold may be too aggressive")

        # ── Build clip-level features — BEFORE filtering ──────────────────
        print(f"\n  Building clip-level features (before filtering)...")
        clip_df_full = build_clip_features(feat_df, embeddings, labels)

        # Add rain scores
        clip_df_full = clip_df_full.merge(
            rain_df[['clip_name', 'rain_score']], on='clip_name', how='left')

        # ── Filter rain clips ─────────────────────────────────────────────
        keep_mask    = clip_df_full['rain_score'] <= threshold
        clip_df_filt = clip_df_full[keep_mask].copy().reset_index(drop=True)
        clip_df_filt = clip_df_filt.drop(columns=['rain_score'])
        clip_df_full_run = clip_df_full.drop(
            columns=['rain_score']).copy()

        print(f"  Before filtering: {len(clip_df_full_run)} clips  "
              f"({int(clip_df_full_run['clip_label'].sum())} sim)")
        print(f"  After filtering:  {len(clip_df_filt)} clips  "
              f"({int(clip_df_filt['clip_label'].sum())} sim)")

        # ── Run GB CV before filtering ────────────────────────────────────
        print(f"\n  Running GB CV before rain filtering...")
        res_before = run_gb_clip_cv(clip_df_full_run)

        # ── Run GB CV after filtering ─────────────────────────────────────
        print(f"  Running GB CV after rain filtering...")
        res_after = run_gb_clip_cv(clip_df_filt)

        if res_before and res_after:
            print(f"\n  Results comparison:")
            print(f"    Before: P={res_before['p']:.3f}  "
                  f"R={res_before['r']:.3f}  "
                  f"{'BEAT' if res_before['beat'] else 'miss'}")
            print(f"    After:  P={res_after['p']:.3f}  "
                  f"R={res_after['r']:.3f}  "
                  f"{'BEAT' if res_after['beat'] else 'miss'}")
            diff_p = res_after['p'] - res_before['p']
            diff_r = res_after['r'] - res_before['r']
            print(f"    Change: P{diff_p:+.3f}  R{diff_r:+.3f}")
            print(f"    Previous clip-level GB: "
                  f"P={rec['old_p']:.3f}  R={rec['old_r']:.3f}")

        all_results[name] = {
            'before': res_before,
            'after':  res_after,
            'n_removed': len(rain_clips),
            'n_sim_removed': int(n_rain_sim),
            'threshold': threshold,
            'rec': rec,
        }

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("RAIN FILTERING SUMMARY")
    print("=" * 70)
    print(f"{'Rec':<6} {'Removed':>8} {'Before P':>9} {'Before R':>9} "
          f"{'After P':>8} {'After R':>8} {'Change P':>9} {'Beat?':>6}")
    print("-" * 70)

    for rec in RECORDERS:
        name = rec['name']
        if name not in all_results:
            continue
        r = all_results[name]
        if not r['before'] or not r['after']:
            continue
        diff_p = r['after']['p'] - r['before']['p']
        beat   = "YES" if r['after']['beat'] else "no"
        print(f"{name:<6} {r['n_removed']:>8} "
              f"{r['before']['p']:>9.3f} {r['before']['r']:>9.3f} "
              f"{r['after']['p']:>8.3f} {r['after']['r']:>8.3f} "
              f"{diff_p:>+9.3f} {beat:>6}")
    print("=" * 70)

    # ── PR curve comparison plots ─────────────────────────────────────────────
    n   = len([r for r in all_results.values()
               if r['before'] and r['after']])
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
    fig.patch.set_facecolor("#2C5F2D")
    if n == 1:
        axes = [axes]

    ax_idx = 0
    for rec in RECORDERS:
        name = rec['name']
        if name not in all_results:
            continue
        r = all_results[name]
        if not r['before'] or not r['after']:
            continue

        ax = axes[ax_idx]
        ax.set_facecolor("#2C5F2D")
        ax.plot(r['before']['rec'], r['before']['prec'],
                color="#F5A623", linewidth=1.8,
                label=f"Before  P={r['before']['p']:.3f} "
                      f"R={r['before']['r']:.3f}")
        ax.plot(r['after']['rec'], r['after']['prec'],
                color="#97BC62", linewidth=1.8,
                label=f"After   P={r['after']['p']:.3f} "
                      f"R={r['after']['r']:.3f}")
        ax.axhline(0.70, color="white", linestyle="--",
                   linewidth=1.2, alpha=0.7)
        ax.axvline(0.70, color="white", linestyle=":",
                   linewidth=1.2, alpha=0.7)
        ax.tick_params(colors="white")
        ax.set_xlabel("Recall", color="white", fontsize=10)
        ax.set_ylabel("Precision", color="white", fontsize=10)
        ax.set_title(f"{name} — Rain Filter Effect\n"
                     f"({r['n_removed']} clips removed)",
                     color="white", fontsize=10)
        ax.legend(facecolor="#1F451F", labelcolor="white",
                  framealpha=0.8, fontsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#97BC62")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax_idx += 1

    plt.tight_layout()
    out_plot = os.path.join(BASE_DIR, "rain_filter_pr_curves.png")
    plt.savefig(out_plot, dpi=150, bbox_inches="tight",
                facecolor="#2C5F2D")
    print(f"\nPlot saved -> {out_plot}")

    # ── Save report ───────────────────────────────────────────────────────────
    with open(OUT_REPORT, "w", encoding="utf-8") as fh:
        fh.write("Rain Filtering Results\n")
        fh.write("=" * 50 + "\n\n")
        fh.write("Method: flag clips in top 10% of baseline rain score\n")
        fh.write("Rain score = weighted combo of RMS energy, spectral\n")
        fh.write("entropy, rolloff stability, and silence fraction.\n\n")
        for rec in RECORDERS:
            name = rec['name']
            if name not in all_results:
                continue
            r = all_results[name]
            if not r['before'] or not r['after']:
                continue
            fh.write(f"{name}:\n")
            fh.write(f"  Clips removed: {r['n_removed']} "
                     f"(threshold={r['threshold']:.3f})\n")
            fh.write(f"  Sim clips removed: {r['n_sim_removed']}\n")
            fh.write(f"  Before: P={r['before']['p']:.3f}  "
                     f"R={r['before']['r']:.3f}\n")
            fh.write(f"  After:  P={r['after']['p']:.3f}  "
                     f"R={r['after']['r']:.3f}\n")
            diff_p = r['after']['p'] - r['before']['p']
            fh.write(f"  Change: P{diff_p:+.3f}\n\n")
    print(f"Report saved -> {OUT_REPORT}")
    print("\nDone.")