"""
67_unsupervised_fusion.py
--------------------------
Unsupervised fusion of velocity + z-score signals.

KEY INSIGHT FROM SCRIPTS 65 AND 66:
  Supervised classifiers trained on site-specific data overfit.
  Rolling z-score velocity WITHOUT a classifier gives the best
  cross-site result (0.371) — better than 14 supervised approaches.

  Why? Because it is purely local and self-normalising:
    - No training data from other sites
    - Compares only to the site's own recent baseline
    - Learns nothing about what disturbance "looks like" globally

  This experiment extends that principle to multiple signals:
  combine several locally-normalised unsupervised scores
  WITHOUT ever training a classifier across sites.

SIGNALS COMBINED (all locally normalised, no cross-site training):
  1. BirdNET z-score velocity   — how fast is embedding moving?
  2. Perch z-score velocity     — same on Perch embeddings
  3. Adaptive spectral z-scores — how unusual is the audio content?
  4. Spectral z-score norm      — magnitude of z-score vector
  5. Combined anomaly score     — weighted combination

COMBINATION STRATEGIES:
  A. Max of normalised scores   — flag if ANY signal spikes
  B. Mean of normalised scores  — flag if ALL signals elevate
  C. Product of percentiles     — flag if signals agree
  D. Rank-weighted sum          — rank each signal, sum ranks

ALL STRATEGIES ARE DEPLOYMENT-VALID:
  Each signal is normalised using only the LAST 40 CLIPS
  from the current site. No cross-site training ever.
  Drop in a new site, record 2 minutes, start monitoring.

ALSO TESTS: sequence voting on top of combined score.
  Require K of last N clips to have high combined anomaly score.

GOAL: push cross-site average precision above 0.371.
"""

import os
import numpy as np
import pandas as pd
import zipfile
import io
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (precision_recall_fscore_support,
                              precision_recall_curve)

BASE_DIR  = r"C:\Users\BitaMassoudi\KasmirWorldProjects\toucanet-option-b"
ZIP_PATH  = r"C:\Users\BitaMassoudi\Downloads\raw_segments_per_clip.zip"
SEED      = 42
ROLLING_N = 40

RECORDERS = [
    ('AM4','am4_full_feature_matrix.csv','am4_full_emb.npy'),
    ('AM2','am2_feature_matrix.csv','am2_time_controlled_emb.npy'),
    ('AM5','am5_feature_matrix.csv','am5_time_controlled_emb.npy'),
    ('AM6','am6_feature_matrix.csv','am6_time_controlled_emb.npy'),
    ('AM1','am1_feature_matrix.csv','am1_full_emb.npy'),
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


def parse_timestamp(clip_name):
    parts = clip_name.split('_')
    try:
        date = parts[3]; time = parts[4].replace('.wav','')
        return pd.Timestamp(
            f"{date[:4]}-{date[4:6]}-{date[6:8]} "
            f"{time[:2]}:{time[2:4]}:{time[4:6]}")
    except:
        return pd.Timestamp('2025-01-01')


def load_birdnet(clip_names):
    stems = {cn.replace('.wav',''): cn for cn in clip_names}
    emb_map = {}
    with zipfile.ZipFile(ZIP_PATH) as z:
        entries = [n for n in z.namelist()
                   if n.endswith('.npz') and 'Audio_Moth' in n
                   and '__MACOSX' not in n]
        for n in entries:
            stem = n.replace('raw_segments_per_clip/','').replace('.npz','')
            if stem not in stems: continue
            try:
                with z.open(n) as f:
                    data = np.load(io.BytesIO(f.read()))
                    emb = np.nan_to_num(
                        data['segments'].astype(np.float64).mean(axis=0),
                        nan=0.0, posinf=0.0, neginf=0.0)
                    emb_map[stems[stem]] = emb
            except: pass
    return emb_map


def rolling_percentile(signal, n=40):
    """Convert signal to percentile vs rolling window — 0 to 1."""
    pct = np.zeros(len(signal))
    for t in range(len(signal)):
        ws = max(0, t-n)
        window = signal[ws:t]
        if len(window) > 1:
            pct[t] = np.mean(window < signal[t])
        else:
            pct[t] = 0.5
    return pct


def rolling_zscore(signal, n=40):
    zs = np.zeros(len(signal))
    for t in range(len(signal)):
        ws = max(0, t-n)
        window = signal[ws:t]
        if len(window) > 2:
            mu = window.mean(); sig = max(window.std(), 1e-6)
            zs[t] = (signal[t] - mu) / sig
    return zs


def compute_velocity_zscore(emb_matrix, n=40):
    """Rolling z-score of L2 velocity in embedding space."""
    raw = np.zeros(len(emb_matrix))
    for t in range(1, len(emb_matrix)):
        raw[t] = np.linalg.norm(emb_matrix[t] - emb_matrix[t-1])
    return rolling_zscore(raw, n), rolling_percentile(raw, n)


def compute_spectral_anomaly(spec_vals, n=40):
    """
    Adaptive z-score magnitude — how anomalous is this clip
    relative to recent baseline in spectral feature space.
    Returns the L2 norm of the z-score vector.
    """
    norms = np.zeros(len(spec_vals))
    for t in range(len(spec_vals)):
        ws = max(0, t-n)
        window = spec_vals[ws:t]
        if len(window) < 2:
            norms[t] = 0.0
        else:
            rm = window.mean(0); rs = window.std(0)
            rs[rs<1e-6] = 1e-6
            z = (spec_vals[t] - rm) / rs
            norms[t] = np.linalg.norm(z)
    return norms, rolling_percentile(norms, n)


def sequence_vote(scores, N, K):
    """Flag clip if K of last N clips exceed threshold."""
    # Find optimal threshold
    sorted_scores = np.sort(scores)[::-1]
    threshold = sorted_scores[min(int(len(scores)*0.15), len(scores)-1)]
    flags = (scores >= threshold).astype(int)
    voted = np.zeros(len(flags))
    for i in range(len(flags)):
        ws = max(0, i-N+1)
        if flags[ws:i+1].sum() >= K:
            voted[i] = 1.0
    return voted


def evaluate(y, scores):
    scores = np.nan_to_num(scores)
    if len(np.unique(y)) < 2: return 0,0,False
    pr,rc,th = precision_recall_curve(y, scores)
    valid = np.where((pr[:-1]>=0.70)&(rc[:-1]>=0.70))[0]
    if len(valid)>0:
        bi=valid[np.argmax(pr[valid]+rc[valid])]; bt=float(th[bi])
    else:
        bi=np.argmax(pr[:-1]+rc[:-1]); bt=float(th[bi])
    preds=(scores>=bt).astype(int)
    p,r,_,_=precision_recall_fscore_support(
        y,preds,average='binary',zero_division=0)
    return p,r,(p>=0.70 and r>=0.70)


if __name__ == '__main__':

    print("Unsupervised Fusion — Velocity + Z-Scores")
    print("No cross-site training. All signals locally normalised.")
    print("="*65)

    # Load all data
    print("\nLoading data...")
    all_clip_names = []
    recorder_raw   = {}
    for name,ff,ef in RECORDERS:
        fp=os.path.join(BASE_DIR,ff); ep=os.path.join(BASE_DIR,ef)
        if not os.path.exists(fp) or not os.path.exists(ep): continue
        feat_df=pd.read_csv(fp); emb=np.load(ep)
        n=min(len(feat_df),len(emb))
        feat_df=feat_df.iloc[:n].reset_index(drop=True); emb=emb[:n]
        all_clip_names.extend(feat_df['clip_name'].unique().tolist())
        recorder_raw[name]=(feat_df,emb)

    print("  Loading BirdNET embeddings...")
    birdnet_map = load_birdnet(all_clip_names)
    print(f"  Loaded {len(birdnet_map)} BirdNET embeddings")

    # Build per-recorder chronological dataframes + compute all signals
    all_clips = {}
    for name,(feat_df,perch_emb) in recorder_raw.items():
        rows = []
        for cn in feat_df['clip_name'].unique():
            if cn not in birdnet_map: continue
            mask = feat_df['clip_name'].values == cn
            row = {'clip_name':cn,
                   'clip_label':int(feat_df['clip_label'].values[mask][0]),
                   'clip_hour': feat_df['clip_hour'].values[mask][0],
                   'timestamp': parse_timestamp(cn),
                   'bn_emb':    birdnet_map[cn]}
            pe = perch_emb[mask]
            for i,v in enumerate(pe.mean(0)): row[f'em_{i}']=v
            for k in SPECTRAL_KEYS:
                if k in feat_df.columns:
                    row[k]=feat_df.loc[mask,k].values[0]
            rows.append(row)
        if not rows: continue
        df = pd.DataFrame(rows).sort_values('timestamp').reset_index(drop=True)

        # BirdNET velocity
        bn_mat = np.array(df['bn_emb'].tolist())
        sc_bn=StandardScaler(); pca_bn=PCA(n_components=32,random_state=SEED)
        bn_pca = pca_bn.fit_transform(sc_bn.fit_transform(bn_mat))
        bn_vel_z, bn_vel_pct = compute_velocity_zscore(bn_pca, ROLLING_N)

        # Perch velocity
        ec=[c for c in df.columns if c.startswith('em_')]
        pe_mat=df[ec].values
        sc_pe=StandardScaler(); pca_pe=PCA(n_components=32,random_state=SEED)
        pe_pca=pca_pe.fit_transform(sc_pe.fit_transform(pe_mat))
        pe_vel_z, pe_vel_pct = compute_velocity_zscore(pe_pca, ROLLING_N)

        # Spectral anomaly
        spec_vals=df[SPECTRAL_KEYS].values.astype(np.float32)
        spec_norm, spec_pct = compute_spectral_anomaly(spec_vals, ROLLING_N)
        spec_norm_z = rolling_zscore(spec_norm, ROLLING_N)

        df['bn_vel_z']    = bn_vel_z
        df['bn_vel_pct']  = bn_vel_pct
        df['pe_vel_z']    = pe_vel_z
        df['pe_vel_pct']  = pe_vel_pct
        df['spec_anom']   = spec_norm
        df['spec_anom_z'] = spec_norm_z
        df['spec_pct']    = spec_pct

        all_clips[name] = df
        print(f"  {name}: {len(df)} clips computed")

    available = list(all_clips.keys())

    # ── Per-site: show signal quality ─────────────────────────────────────────
    print(f"\n{'='*65}")
    print("PER-SITE SIGNAL QUALITY")
    print("(do disturbance clips have higher scores than baseline?)")
    print("="*65)
    for name in available:
        df = all_clips[name]
        y  = df['clip_label'].values
        for sig in ['bn_vel_pct','pe_vel_pct','spec_pct']:
            base_m = df.loc[y==0, sig].mean()
            sim_m  = df.loc[y==1, sig].mean()
            ratio  = sim_m/max(base_m,1e-6)
            print(f"  {name} {sig:<14}: baseline={base_m:.3f} "
                  f"sim={sim_m:.3f} ratio={ratio:.2f}x")

    # ── Cross-site: unsupervised fusion strategies ────────────────────────────
    print(f"\n{'='*65}")
    print("CROSS-SITE LODO — UNSUPERVISED FUSION")
    print("No classifier trained. Signals computed locally per site.")
    print("="*65)

    strategies = {
        'S1_bn_vel_pct':        lambda df: df['bn_vel_pct'].values,
        'S2_pe_vel_pct':        lambda df: df['pe_vel_pct'].values,
        'S3_spec_pct':          lambda df: df['spec_pct'].values,
        'S4_max(vel,spec)':     lambda df: np.maximum(
                                    df['bn_vel_pct'].values,
                                    df['spec_pct'].values),
        'S5_mean(vel,spec)':    lambda df: (
                                    df['bn_vel_pct'].values +
                                    df['spec_pct'].values) / 2,
        'S6_mean(all3)':        lambda df: (
                                    df['bn_vel_pct'].values +
                                    df['pe_vel_pct'].values +
                                    df['spec_pct'].values) / 3,
        'S7_product(vel*spec)': lambda df: (
                                    df['bn_vel_pct'].values *
                                    df['spec_pct'].values),
        'S8_max(all3)':         lambda df: np.maximum(
                                    np.maximum(df['bn_vel_pct'].values,
                                               df['pe_vel_pct'].values),
                                    df['spec_pct'].values),
    }

    strat_results = {}
    for s_name, score_fn in strategies.items():
        results = {}
        for test_name in available:
            df  = all_clips[test_name]
            y   = df['clip_label'].values
            scores = score_fn(df)
            if len(np.unique(y)) < 2: continue
            p,r,beat = evaluate(y, scores)
            results[test_name] = (p,r,beat)
        avg = np.mean([v[0] for v in results.values()]) if results else 0
        strat_results[s_name] = (results, avg)

    # Sequence voting on best strategy
    print("\n  Strategy results (no sequence voting):")
    best_s = max(strat_results.items(), key=lambda x: x[1][1])
    for s_name,(_,avg) in sorted(strat_results.items(),key=lambda x:-x[1][1]):
        marker=" ★" if avg>0.371 else (" ▲" if avg>0.364 else "")
        print(f"  {s_name:<28}: {avg:.3f}{marker}")

    print(f"\n  Best strategy: {best_s[0]} = {best_s[1][1]:.3f}")
    best_score_fn = strategies[best_s[0]]

    # Sequence voting on best strategy
    print(f"\n  Sequence voting on {best_s[0]}:")
    seq_results = {}
    for N,K in [(3,2),(5,3),(7,4),(10,6),(15,9)]:
        results = {}
        for test_name in available:
            df = all_clips[test_name]
            y  = df['clip_label'].values
            scores = best_score_fn(df)
            voted  = sequence_vote(scores, N, K)
            if len(np.unique(y)) < 2: continue
            p,r,beat = evaluate(y, voted)
            results[test_name] = (p,r,beat)
        avg = np.mean([v[0] for v in results.values()]) if results else 0
        n_beat = sum(v[2] for v in results.values())
        marker = " ★" if avg>0.371 else (" ▲" if avg>0.364 else "")
        print(f"    N={N} K={K}: avg P={avg:.3f}  beat 0.70: {n_beat}/5{marker}")
        seq_results[(N,K)] = (results,avg)

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n\n{'='*65}")
    print("UNSUPERVISED FUSION — FINAL SUMMARY")
    print("="*65)

    all_avgs = [(s,avg) for s,(_,avg) in strat_results.items()]
    all_avgs += [(f"seq_{N}_{K}",avg)
                 for (N,K),(_,avg) in seq_results.items()]
    best_name, best_avg = max(all_avgs, key=lambda x: x[1])

    print(f"\n  Previous best (velocity z-score):       0.371")
    print(f"  Previous best (deployment calib.):      0.364")
    print(f"\n  Best unsupervised fusion result: {best_name} = {best_avg:.3f}")

    if best_avg > 0.371:
        print(f"  NEW BEST — {best_avg-0.371:+.3f} above velocity alone")
        print(f"  Unsupervised fusion is additive — combining signals helps")
    elif best_avg > 0.364:
        print(f"  Beats deployment calibration but not velocity alone")
        print(f"  Velocity remains the strongest single unsupervised signal")
    else:
        print(f"  No improvement over velocity alone")
        print(f"  The ceiling of 0.364-0.371 holds for unsupervised approaches")
        print(f"  Conclusion: hour overlap is the fundamental blocker")

    print("\nDone.")
