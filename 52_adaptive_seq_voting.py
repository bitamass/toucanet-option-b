"""
52_adaptive_seq_voting.py
--------------------------
Combines adaptive z-scores (script 49) with sequence voting (script 42).

FINDINGS SO FAR:
  Script 49 adaptive z-scores:
    20-clip rolling window gives best per-site results in entire project:
    AM4=0.972  AM2=0.909  AM5=0.859  AM6=0.736  AM1=0.949
    But cross-site: no improvement (avg P=0.353)

  Script 42 sequence voting:
    Best per-site results previously:
    AM4=0.884  AM2=0.952  AM5=0.728  AM6=0.875  AM1=0.961
    Reduced false alarms dramatically.

THIS SCRIPT:
  Combines both:
  1. Use adaptive z-scores (rolling 20-clip window) for better
     per-clip probability scores
  2. Apply sequence voting on top of those scores for temporal filtering

  Hypothesis: adaptive z-scores make each individual clip score
  more accurate. Sequence voting then filters the few remaining
  isolated false alarms. Combined precision should exceed both
  approaches individually.

WINDOW COMBINATIONS TESTED:
  For z-score window: 20 clips (best from script 49)
  For sequence voting: N=3K2, N=5K3, N=5K4, N=10K6, N=10K7, N=15K9

EXPECTED OUTCOME:
  Per-site precision potentially above 0.90 at most sites.
  This would be the strongest result in the entire project.
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

RECORDERS = [
    ('AM4','am4_full_feature_matrix.csv',
     'am4_full_emb.npy','am4_full_labels.npy', 0.884, 0.835),
    ('AM2','am2_feature_matrix.csv',
     'am2_time_controlled_emb.npy','am2_time_controlled_labels.npy', 0.952, 0.876),
    ('AM5','am5_feature_matrix.csv',
     'am5_time_controlled_emb.npy','am5_time_controlled_labels.npy', 0.728, 0.902),
    ('AM6','am6_feature_matrix.csv',
     'am6_time_controlled_emb.npy','am6_time_controlled_labels.npy', 0.875, 0.794),
    ('AM1','am1_feature_matrix.csv',
     'am1_full_emb.npy','am1_full_labels.npy', 0.961, 0.831),
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

SEQ_WINDOWS = [
    {'N':3,  'K':2},
    {'N':5,  'K':3},
    {'N':5,  'K':4},
    {'N':10, 'K':6},
    {'N':10, 'K':7},
    {'N':15, 'K':9},
]

N_ROLL = 20  # rolling baseline window size (best from script 49)


def parse_timestamp(clip_name):
    parts = clip_name.split('_')
    try:
        date = parts[3]; time = parts[4].replace('.wav','')
        return pd.Timestamp(f"{date[:4]}-{date[4:6]}-{date[6:8]} "
                           f"{time[:2]}:{time[2:4]}:{time[4:6]}")
    except:
        return pd.Timestamp('2025-01-01')


def build_clip_df(feat_df, emb):
    rows = []
    for cn in feat_df['clip_name'].unique():
        mask = feat_df['clip_name'].values == cn
        cl   = feat_df['clip_label'].values[mask][0]
        hour = feat_df['clip_hour'].values[mask][0]
        ts   = parse_timestamp(cn)
        ce   = emb[mask]
        row  = {'clip_name':cn,'clip_label':cl,
                'clip_hour':hour,'timestamp':ts}
        for k in SPECTRAL_KEYS:
            if k in feat_df.columns:
                row[k] = feat_df.loc[mask,k].values[0]
        for i,v in enumerate(ce.mean(0)): row[f'em_{i}']=v
        for i,v in enumerate(ce.std(0)):  row[f'es_{i}']=v
        for i,v in enumerate(ce.max(0)):  row[f'ex_{i}']=v
        rows.append(row)
    df = pd.DataFrame(rows).sort_values('timestamp').reset_index(drop=True)
    return df


def compute_adaptive_zscores(clip_df, n_clips):
    """Rolling z-scores using last n_clips baseline clips."""
    spec_vals = clip_df[SPECTRAL_KEYS].values
    labels    = clip_df['clip_label'].values
    z_rows    = []
    for i in range(len(clip_df)):
        before_mask = (labels[:i] == 0)
        before_idx  = np.where(before_mask)[0]
        if len(before_idx) == 0:
            z = np.zeros(len(SPECTRAL_KEYS))
        else:
            window_idx   = before_idx[-n_clips:]
            window_specs = spec_vals[window_idx]
            roll_mean    = window_specs.mean(axis=0)
            roll_std     = window_specs.std(axis=0)
            roll_std[roll_std < 1e-6] = 1e-6
            z = (spec_vals[i] - roll_mean) / roll_std
        z_rows.append(dict(zip(Z_KEYS, z)))
    return pd.DataFrame(z_rows, index=clip_df.index)


def apply_sequence_voting(clip_df, probs, base_thresh, N, K):
    """Sliding window in chronological order."""
    # clips already sorted by timestamp
    flags      = (probs >= base_thresh).astype(int)
    final_preds = np.zeros(len(flags), dtype=int)
    for i in range(len(flags)):
        ws = max(0, i - N + 1)
        if flags[ws:i+1].sum() >= K:
            final_preds[i] = 1
    return final_preds


def evaluate(y, preds):
    p,r,_,_=precision_recall_fscore_support(
        y,preds,average='binary',zero_division=0)
    return p,r,(p>=0.70 and r>=0.70)


def run_cv_adaptive(clip_df, n_roll):
    """5-fold CV with adaptive z-scores, returns prob array."""
    ec   = [c for c in clip_df.columns if c.startswith(('em_','es_','ex_'))]
    cn   = clip_df['clip_name'].values
    cl   = clip_df['clip_label'].values
    if len(np.unique(cl))<2 or cl.sum()<5: return None

    # Pre-compute adaptive z-scores on full dataset in chronological order
    z_df = compute_adaptive_zscores(clip_df, n_roll)
    z_df.columns = Z_KEYS

    cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    probs = np.zeros(len(clip_df), dtype=np.float32)

    for tr, va in cv.split(cn, cl):
        tdf = clip_df.iloc[tr].copy().reset_index(drop=True)
        vdf = clip_df.iloc[va].copy().reset_index(drop=True)
        ytr = cl[tr].astype(int)
        z_tr = z_df.iloc[tr].copy().reset_index(drop=True)
        z_va = z_df.iloc[va].copy().reset_index(drop=True)

        pca  = PCA(n_components=50, random_state=SEED)
        etr  = pca.fit_transform(tdf[ec].values)
        eva  = pca.transform(vdf[ec].values)

        X_tr = np.hstack([tdf[SPECTRAL_KEYS].values,
                           z_tr.values, etr]).astype(np.float32)
        X_va = np.hstack([vdf[SPECTRAL_KEYS].values,
                           z_va.values, eva]).astype(np.float32)

        sc      = StandardScaler()
        X_tr_s  = sc.fit_transform(X_tr)
        X_va_s  = sc.transform(X_va)

        n_pos=ytr.sum(); n_neg=(ytr==0).sum()
        w=np.where(ytr==1, n_neg/max(n_pos,1), 1.0)
        gb=GradientBoostingClassifier(n_estimators=100,max_depth=3,
           learning_rate=0.1,random_state=SEED,subsample=0.8)
        gb.fit(X_tr_s, ytr, sample_weight=w)
        probs[va] = gb.predict_proba(X_va_s)[:,1]

    return probs


if __name__ == '__main__':

    print("Adaptive Z-Scores + Sequence Voting")
    print("="*65)
    print(f"Rolling window: {N_ROLL} baseline clips")
    print("="*65)

    all_results = {}

    for name,ff,ef,lf,prev_p,prev_r in RECORDERS:
        print(f"\n{'─'*55}")
        print(f"{name}  prev best: P={prev_p:.3f} R={prev_r:.3f}")

        feat_path = os.path.join(BASE_DIR,ff)
        emb_path  = os.path.join(BASE_DIR,ef)
        if not os.path.exists(feat_path) or not os.path.exists(emb_path):
            print(f"  Files not found"); continue

        feat_df = pd.read_csv(feat_path)
        emb     = np.load(emb_path)
        n = min(len(feat_df),len(emb))
        feat_df = feat_df.iloc[:n].reset_index(drop=True); emb=emb[:n]

        clip_df = build_clip_df(feat_df, emb)
        y       = clip_df['clip_label'].values.astype(int)
        print(f"  {len(clip_df)} clips ({y.sum()} sim) — sorted chronologically")

        print(f"  Running CV with adaptive z-scores (rolling {N_ROLL} clips)...")
        probs = run_cv_adaptive(clip_df, N_ROLL)
        if probs is None:
            print(f"  Insufficient data"); continue

        # Find base threshold
        pr,rc,th = precision_recall_curve(y, probs)
        valid = np.where((pr[:-1]>=0.70)&(rc[:-1]>=0.70))[0]
        if len(valid)>0:
            bi=valid[np.argmax(pr[valid]+rc[valid])]; bt=float(th[bi])
        else:
            bi=np.argmax(pr[:-1]+rc[:-1]); bt=float(th[bi])

        # Single clip with adaptive z
        preds_single = (probs>=bt).astype(int)
        p0,r0,b0 = evaluate(y, preds_single)
        fa0 = int(((preds_single==1)&(y==0)).sum())
        print(f"\n  Adaptive z only (no seq):  P={p0:.3f} R={r0:.3f} "
              f"FA={fa0} {'BEAT' if b0 else 'miss'}")

        # Sequence voting on top
        print(f"  + Sequence voting:")
        print(f"  {'Config':<14} {'P':>7} {'R':>7} {'FA':>5} {'Beat?':>6} {'vs prev':>8}")
        print(f"  {'-'*50}")

        best_p,best_r,best_beat,best_cfg = p0,r0,b0,"adaptive_only"
        rec_results = {'adaptive_only':(p0,r0,b0,fa0)}

        for w in SEQ_WINDOWS:
            N,K = w['N'],w['K']
            preds_seq = apply_sequence_voting(clip_df,probs,bt,N,K)
            p,r,beat  = evaluate(y,preds_seq)
            fa        = int(((preds_seq==1)&(y==0)).sum())
            diff_p    = p-prev_p
            flag      = " ★" if beat and p>0.90 else ""
            print(f"  N={N:2d} K={K:2d}:       "
                  f"{p:>7.3f} {r:>7.3f} {fa:>5} {'BEAT' if beat else 'miss':>6} "
                  f"{diff_p:>+8.3f}{flag}")
            rec_results[f"N={N}K={K}"] = (p,r,beat,fa)
            if beat and p+r > best_p+best_r:
                best_p,best_r,best_beat,best_cfg = p,r,beat,f"N={N}K={K}"

        all_results[name] = {
            'best_p':best_p,'best_r':best_r,'best_beat':best_beat,
            'best_cfg':best_cfg,'prev_p':prev_p,'prev_r':prev_r,
            'results':rec_results
        }

        print(f"\n  Best: P={best_p:.3f} R={best_r:.3f} [{best_cfg}]")
        print(f"  Change vs prev best: P{best_p-prev_p:+.3f} R{best_r-prev_r:+.3f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n\n" + "="*75)
    print("ADAPTIVE Z-SCORE + SEQUENCE VOTING — FINAL SUMMARY")
    print("="*75)
    print(f"{'Rec':<6} {'Prev P':>7} {'Prev R':>7} "
          f"{'New P':>7} {'New R':>7} {'Change P':>9} {'Config'}")
    print("-"*75)

    for name,*_,prev_p,prev_r in RECORDERS:
        if name not in all_results: continue
        r = all_results[name]
        dp = r['best_p']-prev_p; dr = r['best_r']-prev_r
        star = " ★" if r['best_p']>=0.90 else ""
        print(f"{name:<6} {prev_p:>7.3f} {prev_r:>7.3f} "
              f"{r['best_p']:>7.3f} {r['best_r']:>7.3f} "
              f"{dp:>+9.3f} {r['best_cfg']}{star}")

    print("="*75)

    n_above_90 = sum(1 for r in all_results.values() if r['best_p']>=0.90)
    n_beat     = sum(1 for r in all_results.values() if r['best_beat'])
    print(f"\nRecorders above 0.90 precision: {n_above_90}/{len(all_results)}")
    print(f"Recorders above 0.70 precision: {n_beat}/{len(all_results)}")

    # Save report
    out = os.path.join(BASE_DIR,"adaptive_seq_voting_results.txt")
    with open(out,"w",encoding="utf-8") as fh:
        fh.write("Adaptive Z-Score + Sequence Voting Results\n"+"="*50+"\n\n")
        fh.write(f"Rolling window: {N_ROLL} baseline clips\n\n")
        for name,*_,prev_p,prev_r in RECORDERS:
            if name not in all_results: continue
            r=all_results[name]
            fh.write(f"{name}: P={r['best_p']:.3f} R={r['best_r']:.3f} "
                     f"[{r['best_cfg']}] "
                     f"change P{r['best_p']-prev_p:+.3f}\n")
    print(f"\nReport saved -> {out}")
    print("\nDone.")
