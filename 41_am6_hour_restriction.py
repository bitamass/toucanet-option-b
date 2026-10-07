"""
41_am6_hour_restriction.py
---------------------------
Tests hour restriction as a fix for AM6 false alarms.

FINDING FROM SCRIPT 40:
  Hours 15 and 16 (3-4 PM) have 30-33% false alarm rates.
  At those hours normal baseline clips acoustically resemble
  simulation clips — high rolloff, high centroid, high bandwidth.
  No threshold can fix this within current features.

THE FIX:
  Simply exclude hours 15 and 16 from both training and testing.
  The classifier only runs on hours where it is reliable.

THREE VARIANTS TESTED:
  A: Exclude hours 15 and 16 only
  B: Exclude hours 15, 16, and 8 (8 AM also has 21% alarm rate)
  C: Keep only the cleanest hours (9, 13, 17, 18)

EXPECTED OUTCOME:
  Fewer false alarms → higher precision → potentially beat 0.70
  Trade-off: fewer clips total → less training data
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (precision_recall_fscore_support,
                              precision_recall_curve)

BASE_DIR = r"C:\Users\BitaMassoudi\KasmirWorldProjects\toucanet-option-b"
SEED     = 42

SPECTRAL_KEYS = (
    [f"mfcc_mean_{i}" for i in range(13)] +
    [f"mfcc_std_{i}"  for i in range(13)] +
    ["centroid_mean","centroid_std","rolloff_mean","rolloff_std",
     "bandwidth_mean","bandwidth_std","zcr_mean","zcr_std",
     "rms_mean","rms_std","silence_fraction",
     "spec_entropy_mean","spec_entropy_std",
     "temporal_entropy","onset_count"]
)
Z_KEYS = [f"z_{k}" for k in SPECTRAL_KEYS]

VARIANTS = {
    "A_exclude_15_16":      {"exclude": [15, 16]},
    "B_exclude_8_15_16":    {"exclude": [8, 15, 16]},
    "C_best_hours_only":    {"keep":    [9, 13, 17, 18]},
    "D_no_restriction":     {"exclude": []},
}


def build_clip_df(feat_df, emb):
    rows = []
    for cn in feat_df['clip_name'].unique():
        mask  = feat_df['clip_name'].values == cn
        cl    = feat_df['clip_label'].values[mask][0]
        hour  = feat_df['clip_hour'].values[mask][0]
        ce    = emb[mask]
        row   = {'clip_name': cn, 'clip_label': cl, 'clip_hour': hour}
        for k in SPECTRAL_KEYS:
            if k in feat_df.columns:
                row[k] = feat_df.loc[mask, k].values[0]
        for i, v in enumerate(ce.mean(0)): row[f'em_{i}'] = v
        for i, v in enumerate(ce.std(0)):  row[f'es_{i}'] = v
        for i, v in enumerate(ce.max(0)):  row[f'ex_{i}'] = v
        rows.append(row)
    return pd.DataFrame(rows)


def run_cv(clip_df):
    """Corrected clip-level 5-fold CV with GB."""
    ec      = [c for c in clip_df.columns
               if c.startswith(('em_', 'es_', 'ex_'))]
    cn      = clip_df['clip_name'].values
    cl      = clip_df['clip_label'].values
    cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    probs   = np.zeros(len(clip_df), dtype=np.float32)

    for tr, va in cv.split(cn, cl):
        tdf = clip_df.iloc[tr].copy().reset_index(drop=True)
        vdf = clip_df.iloc[va].copy().reset_index(drop=True)
        ytr = cl[tr].astype(int)

        z_tr_rows, z_va_rows = [], []
        for hour in tdf['clip_hour'].unique():
            bm = (tdf['clip_hour'] == hour) & (tdf['clip_label'] == 0)
            br = tdf.loc[bm, SPECTRAL_KEYS]
            if len(br) == 0: continue
            hm = br.mean(); hs = br.std().replace(0, 1e-6)
            th = tdf['clip_hour'] == hour
            if th.sum() > 0:
                z = (tdf.loc[th, SPECTRAL_KEYS] - hm) / hs
                z.columns = Z_KEYS; z_tr_rows.append(z)
            vh = vdf['clip_hour'] == hour
            if vh.sum() > 0:
                z = (vdf.loc[vh, SPECTRAL_KEYS] - hm) / hs
                z.columns = Z_KEYS; z_va_rows.append(z)

        z_tr = (pd.concat(z_tr_rows).sort_index() if z_tr_rows else
                pd.DataFrame(0, index=tdf.index, columns=Z_KEYS))
        z_va = (pd.concat(z_va_rows).sort_index() if z_va_rows else
                pd.DataFrame(0, index=vdf.index, columns=Z_KEYS))

        pca   = PCA(n_components=50, random_state=SEED)
        etr   = pca.fit_transform(tdf[ec].values)
        eva   = pca.transform(vdf[ec].values)

        X_tr  = np.hstack([tdf[SPECTRAL_KEYS].values,
                            z_tr.values, etr]).astype(np.float32)
        X_va  = np.hstack([vdf[SPECTRAL_KEYS].values,
                            z_va.values, eva]).astype(np.float32)

        sc     = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_va_s = sc.transform(X_va)

        n_pos = ytr.sum(); n_neg = (ytr == 0).sum()
        w     = np.where(ytr == 1, n_neg / max(n_pos, 1), 1.0)
        gb    = GradientBoostingClassifier(
            n_estimators=100, max_depth=3,
            learning_rate=0.1, random_state=SEED, subsample=0.8)
        gb.fit(X_tr_s, ytr, sample_weight=w)
        probs[va] = gb.predict_proba(X_va_s)[:, 1]

    y     = cl.astype(int)
    pr, rc, th = precision_recall_curve(y, probs)
    valid  = np.where((pr[:-1] >= 0.70) & (rc[:-1] >= 0.70))[0]
    if len(valid) > 0:
        bi = valid[np.argmax(pr[valid] + rc[valid])]
        bt = float(th[bi])
        preds = (probs >= bt).astype(int)
        p, r, _, _ = precision_recall_fscore_support(
            y, preds, average="binary", zero_division=0)
        beat = True
    else:
        bi = np.argmax(pr[:-1] + rc[:-1])
        bt = float(th[bi])
        preds = (probs >= bt).astype(int)
        p, r, _, _ = precision_recall_fscore_support(
            y, preds, average="binary", zero_division=0)
        beat = False

    fa   = ((preds == 1) & (y == 0)).sum()
    miss = ((preds == 0) & (y == 1)).sum()
    return {'p': p, 'r': r, 'beat': beat,
            'thresh': bt, 'fa': int(fa), 'miss': int(miss),
            'n_sim': int(y.sum()), 'n_base': int((y == 0).sum())}


if __name__ == '__main__':

    print("AM6 Hour Restriction Experiment")
    print("=" * 60)

    feat_df = pd.read_csv(
        os.path.join(BASE_DIR, 'am6_feature_matrix.csv'))
    emb     = np.load(
        os.path.join(BASE_DIR, 'am6_time_controlled_emb.npy'))
    labels  = np.load(
        os.path.join(BASE_DIR, 'am6_time_controlled_labels.npy'))

    n = min(len(feat_df), len(emb))
    feat_df = feat_df.iloc[:n].reset_index(drop=True)
    emb = emb[:n]

    print("Building clip-level features...")
    clip_df_full = build_clip_df(feat_df, emb)

    print(f"  Full dataset: {len(clip_df_full)} clips  "
          f"({int(clip_df_full['clip_label'].sum())} sim)\n")

    print("Hour distribution in full dataset:")
    for h in sorted(clip_df_full['clip_hour'].unique()):
        h_df = clip_df_full[clip_df_full['clip_hour'] == h]
        n_sim  = int(h_df['clip_label'].sum())
        n_base = int((h_df['clip_label'] == 0).sum())
        print(f"  Hour {h:2d}: {n_sim:3d} sim, {n_base:4d} base")

    results = {}

    for vname, vconfig in VARIANTS.items():
        print(f"\n{'─'*55}")
        print(f"Variant {vname}")

        if 'exclude' in vconfig and vconfig['exclude']:
            mask = ~clip_df_full['clip_hour'].isin(vconfig['exclude'])
            label = f"Excluding hours {vconfig['exclude']}"
        elif 'keep' in vconfig:
            mask = clip_df_full['clip_hour'].isin(vconfig['keep'])
            label = f"Keeping hours {vconfig['keep']} only"
        else:
            mask = pd.Series([True] * len(clip_df_full))
            label = "No restriction (baseline)"

        clip_sub = clip_df_full[mask].copy().reset_index(drop=True)
        n_sim  = int(clip_sub['clip_label'].sum())
        n_base = int((clip_sub['clip_label'] == 0).sum())

        print(f"  {label}")
        print(f"  Clips remaining: {len(clip_sub)}  "
              f"({n_sim} sim, {n_base} base)")

        if n_sim < 10 or n_base < 10 or len(np.unique(
                clip_sub['clip_label'].values)) < 2:
            print(f"  Insufficient data — skipping")
            continue

        res = run_cv(clip_sub)
        results[vname] = res

        beat_str = "BEAT" if res['beat'] else "miss"
        print(f"  P={res['p']:.3f}  R={res['r']:.3f}  "
              f"t={res['thresh']:.3f}  {beat_str}")
        print(f"  False alarms: {res['fa']}  Missed: {res['miss']}")

    # Summary
    print(f"\n\n{'='*65}")
    print("HOUR RESTRICTION SUMMARY")
    print("=" * 65)
    print(f"{'Variant':<28} {'Clips':>6} {'Sim':>5} "
          f"{'P':>7} {'R':>7} {'FA':>5} {'Beat?':>6}")
    print("-" * 65)

    for vname, vconfig in VARIANTS.items():
        if vname not in results:
            continue
        res = results[vname]

        if 'exclude' in vconfig and vconfig['exclude']:
            mask = ~clip_df_full['clip_hour'].isin(vconfig['exclude'])
        elif 'keep' in vconfig:
            mask = clip_df_full['clip_hour'].isin(vconfig['keep'])
        else:
            mask = pd.Series([True] * len(clip_df_full))

        n_clips = mask.sum()
        beat    = "YES" if res['beat'] else "no"
        print(f"{vname:<28} {n_clips:>6} {res['n_sim']:>5} "
              f"{res['p']:>7.3f} {res['r']:>7.3f} "
              f"{res['fa']:>5} {beat:>6}")

    print("=" * 65)

    n_beat = sum(1 for r in results.values() if r['beat'])
    if n_beat > 0:
        best = max(
            ((v, r) for v, r in results.items() if r['beat']),
            key=lambda x: x[1]['p'] + x[1]['r'])
        print(f"\nBest variant: {best[0]}")
        print(f"  P={best[1]['p']:.3f}  R={best[1]['r']:.3f}  "
              f"FA={best[1]['fa']}  BEAT target")
        print(f"\nConclusion: Excluding hours "
              f"{VARIANTS[best[0]].get('exclude', 'N/A')} "
              f"fixes AM6 precision above 0.70")
    else:
        print("\nNo variant beats both targets.")
        best_p = max(results.items(), key=lambda x: x[1]['p'])
        best_r = max(results.items(), key=lambda x: x[1]['r'])
        print(f"Best precision: {best_p[0]} P={best_p[1]['p']:.3f}")
        print(f"Best recall:    {best_r[0]} R={best_r[1]['r']:.3f}")
        print("\nAM6 may need more simulation data at hours 15-16")
        print("or a fundamentally different approach.")

    print("\nDone.")