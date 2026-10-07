"""
42_sequence_voting.py
----------------------
Sequence voting — a sliding window post-processing layer
on top of the existing per-clip classifier.

THE IDEA:
  Right now each 3-second clip is classified independently.
  A real disturbance creates a PERSISTENT pattern across many
  consecutive clips — birds go quiet in sequence, vehicle noise
  builds gradually, chainsaw runs for minutes.
  A random false alarm is a single isolated unusual clip.

  By requiring K out of the last N clips to be flagged before
  raising an alert, we filter isolated false alarms while keeping
  genuine sustained disturbance events.

HOW IT WORKS:
  1. Run the existing clip-level GB classifier on ALL clips
     in chronological order — getting a probability score per clip
  2. For each clip, look at the last N clips (sliding window)
  3. Only raise an alert if at least K of those N clips
     scored above the threshold
  4. This is pure post-processing — no retraining needed

PARAMETERS TESTED:
  Window size N: 3, 5, 10, 15 clips
  Votes required K: majority (K > N/2) to strict (K > 0.7*N)

EXPECTED OUTCOME:
  False alarms collapse because random noise does not create
  sustained sequences of unusual clips.
  True detections survive because genuine disturbance is persistent.
  Precision should push above 0.80 at most sites.

IMPORTANT NOTE:
  This requires clips to be in CHRONOLOGICAL ORDER within each
  recorder. We sort by clip_name (which encodes timestamp) before
  applying the sliding window.

Uses saved feature matrices — no re-embedding needed.
Should finish in about 15-20 minutes.
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = r"C:\Users\BitaMassoudi\KasmirWorldProjects\toucanet-option-b"
SEED     = 42

RECORDERS = [
    ('AM4','am4_full_feature_matrix.csv',
     'am4_full_emb.npy','am4_full_labels.npy',
     0.856, 0.906),
    ('AM2','am2_feature_matrix.csv',
     'am2_time_controlled_emb.npy','am2_time_controlled_labels.npy',
     0.779, 0.781),
    ('AM5','am5_feature_matrix.csv',
     'am5_time_controlled_emb.npy','am5_time_controlled_labels.npy',
     0.707, 0.932),
    ('AM6','am6_feature_matrix.csv',
     'am6_time_controlled_emb.npy','am6_time_controlled_labels.npy',
     0.703, 0.744),
    ('AM1','am1_feature_matrix.csv',
     'am1_full_emb.npy','am1_full_labels.npy',
     0.779, 0.898),
]

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

# Sliding window configurations to test
WINDOWS = [
    {'N': 3,  'K': 2,  'label': 'N=3  K=2  (2/3)'},
    {'N': 5,  'K': 3,  'label': 'N=5  K=3  (3/5)'},
    {'N': 5,  'K': 4,  'label': 'N=5  K=4  (4/5)'},
    {'N': 10, 'K': 6,  'label': 'N=10 K=6  (6/10)'},
    {'N': 10, 'K': 7,  'label': 'N=10 K=7  (7/10)'},
    {'N': 15, 'K': 9,  'label': 'N=15 K=9  (9/15)'},
]


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


def get_clip_probs(clip_df):
    """
    Run clip-level GB CV to get probability scores per clip.
    Returns array of probs in same order as clip_df.
    """
    ec   = [c for c in clip_df.columns
            if c.startswith(('em_', 'es_', 'ex_'))]
    cn   = clip_df['clip_name'].values
    cl   = clip_df['clip_label'].values
    cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    probs = np.zeros(len(clip_df), dtype=np.float32)

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

    return probs


def apply_sequence_voting(clip_df, probs, base_thresh, N, K):
    """
    Apply sliding window sequence voting.

    Clips must be sorted chronologically.
    For each clip: count how many of the last N clips
    (including this one) scored above base_thresh.
    Flag as alert only if count >= K.

    Returns array of final predictions (0 or 1).
    """
    # Sort chronologically by clip_name (encodes timestamp)
    sorted_idx = clip_df['clip_name'].argsort().values
    sorted_probs = probs[sorted_idx]
    sorted_labels = clip_df['clip_label'].values[sorted_idx]

    # Binary flags at base threshold
    flags = (sorted_probs >= base_thresh).astype(int)

    # Sliding window vote
    final_preds = np.zeros(len(flags), dtype=int)
    for i in range(len(flags)):
        window_start = max(0, i - N + 1)
        window_votes = flags[window_start:i+1].sum()
        final_preds[i] = 1 if window_votes >= K else 0

    # Map back to original order
    result = np.zeros(len(clip_df), dtype=int)
    result[sorted_idx] = final_preds
    return result


def evaluate(y, preds):
    p, r, _, _ = precision_recall_fscore_support(
        y, preds, average="binary", zero_division=0)
    beat = (p >= 0.70) and (r >= 0.70)
    fa   = int(((preds == 1) & (y == 0)).sum())
    miss = int(((preds == 0) & (y == 1)).sum())
    return p, r, beat, fa, miss


if __name__ == '__main__':

    all_results = {}
    OUT_REPORT  = os.path.join(BASE_DIR, "sequence_voting_results.txt")

    for name, feat_f, emb_f, lab_f, best_p, best_r in RECORDERS:
        print(f"\n{'='*60}")
        print(f"{name} — Sequence Voting")
        print(f"{'='*60}")

        feat_path = os.path.join(BASE_DIR, feat_f)
        emb_path  = os.path.join(BASE_DIR, emb_f)
        lab_path  = os.path.join(BASE_DIR, lab_f)

        if not os.path.exists(feat_path) or \
           not os.path.exists(emb_path):
            print(f"  Files not found — skipping")
            continue

        feat_df = pd.read_csv(feat_path)
        emb     = np.load(emb_path)
        labels  = np.load(lab_path)
        n = min(len(feat_df), len(emb))
        feat_df = feat_df.iloc[:n].reset_index(drop=True)
        emb = emb[:n]

        clip_df = build_clip_df(feat_df, emb)
        y       = clip_df['clip_label'].values.astype(int)

        print(f"  {len(clip_df)} clips  ({y.sum()} sim)")
        print(f"  Getting clip-level probability scores...")
        probs = get_clip_probs(clip_df)

        # Find best single-clip threshold
        pr, rc, th = precision_recall_curve(y, probs)
        valid = np.where((pr[:-1] >= 0.70) & (rc[:-1] >= 0.70))[0]
        if len(valid) > 0:
            bi = valid[np.argmax(pr[valid] + rc[valid])]
            base_thresh = float(th[bi])
        else:
            bi = np.argmax(pr[:-1] + rc[:-1])
            base_thresh = float(th[bi])

        # Single-clip baseline
        preds_single = (probs >= base_thresh).astype(int)
        p0, r0, beat0, fa0, miss0 = evaluate(y, preds_single)
        print(f"\n  Single-clip (no sequence):  "
              f"P={p0:.3f}  R={r0:.3f}  FA={fa0}  "
              f"{'BEAT' if beat0 else 'miss'}")

        rec_results = {'single': (p0, r0, beat0, fa0, miss0)}

        # Sequence voting variants
        print(f"\n  Sequence voting results:")
        print(f"  {'Config':<22} {'P':>7} {'R':>7} "
              f"{'FA':>5} {'Miss':>6} {'Beat?':>6}")
        print(f"  {'-'*55}")

        for w in WINDOWS:
            N, K, label = w['N'], w['K'], w['label']
            preds_seq = apply_sequence_voting(
                clip_df, probs, base_thresh, N, K)
            p, r, beat, fa, miss = evaluate(y, preds_seq)
            beat_str = "BEAT" if beat else "miss"
            flag = " ★" if beat and p > 0.80 else ""
            print(f"  {label:<22} {p:>7.3f} {r:>7.3f} "
                  f"{fa:>5} {miss:>6} {beat_str:>6}{flag}")
            rec_results[label] = (p, r, beat, fa, miss)

        all_results[name] = {
            'rec': (best_p, best_r),
            'results': rec_results
        }

        # Best sequence result
        best_seq = max(
            ((k, v) for k, v in rec_results.items() if k != 'single'),
            key=lambda x: (x[1][2], x[1][0] + x[1][1]))
        bk, bv = best_seq
        print(f"\n  Best sequence config: {bk}")
        print(f"    P={bv[0]:.3f}  R={bv[1]:.3f}  "
              f"FA={bv[3]}  {'BEAT' if bv[2] else 'miss'}")
        print(f"    Single-clip baseline: P={p0:.3f}  R={r0:.3f}  FA={fa0}")
        diff_p = bv[0] - p0
        diff_r = bv[1] - r0
        print(f"    Change: P{diff_p:+.3f}  R{diff_r:+.3f}  "
              f"FA reduced by {fa0 - bv[3]}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 75)
    print("SEQUENCE VOTING SUMMARY — BEST CONFIG PER RECORDER")
    print("=" * 75)
    print(f"{'Rec':<6} {'Old P':>7} {'Single P':>9} "
          f"{'Seq P':>7} {'Seq R':>7} {'FA':>5} {'Beat?':>6} {'Config'}")
    print("-" * 75)

    for name, feat_f, emb_f, lab_f, best_p, best_r in RECORDERS:
        if name not in all_results:
            continue
        rd  = all_results[name]
        sp0 = rd['results']['single'][0]
        sr0 = rd['results']['single'][1]

        # Best beating sequence
        beating = {k: v for k, v in rd['results'].items()
                   if k != 'single' and v[2]}
        if beating:
            bk = max(beating, key=lambda k: beating[k][0]+beating[k][1])
            bv = beating[bk]
        else:
            bk = max((k for k in rd['results'] if k != 'single'),
                     key=lambda k: rd['results'][k][0])
            bv = rd['results'][bk]

        beat = "YES" if bv[2] else "no"
        above08 = " ★" if bv[0] >= 0.80 else ""
        print(f"{name:<6} {best_p:>7.3f} {sp0:>9.3f} "
              f"{bv[0]:>7.3f} {bv[1]:>7.3f} {bv[3]:>5} "
              f"{beat:>6} {bk}{above08}")

    print("=" * 75)

    # Count recorders above 0.80
    n_above_08 = 0
    for name, *_ in RECORDERS:
        if name not in all_results: continue
        rd = all_results[name]
        beating = {k: v for k, v in rd['results'].items()
                   if k != 'single' and v[2]}
        if beating:
            best_p_seq = max(beating[k][0] for k in beating)
            if best_p_seq >= 0.80:
                n_above_08 += 1
    print(f"\nRecorders above 0.80 precision with sequence voting: "
          f"{n_above_08}/{len(all_results)}")

    # ── Save report ───────────────────────────────────────────────────────────
    with open(OUT_REPORT, "w", encoding="utf-8") as fh:
        fh.write("Sequence Voting Results\n")
        fh.write("=" * 50 + "\n\n")
        for name, feat_f, emb_f, lab_f, bp, br in RECORDERS:
            if name not in all_results: continue
            rd = all_results[name]
            fh.write(f"{name}:\n")
            fh.write(f"  Single-clip: P={rd['results']['single'][0]:.3f}"
                     f"  R={rd['results']['single'][1]:.3f}\n")
            beating = {k: v for k, v in rd['results'].items()
                       if k != 'single' and v[2]}
            if beating:
                bk = max(beating, key=lambda k: beating[k][0]+beating[k][1])
                bv = beating[bk]
                fh.write(f"  Best seq ({bk}): "
                         f"P={bv[0]:.3f}  R={bv[1]:.3f}  "
                         f"FA={bv[3]}  BEAT\n\n")
            else:
                fh.write(f"  No sequence config beats both targets\n\n")
    print(f"\nReport saved -> {OUT_REPORT}")
    print("\nDone.")