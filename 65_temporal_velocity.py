"""
65_temporal_velocity.py
------------------------
Temporal Velocity in Embedding Space
Griffin's Spring 2026 key finding — simpler standalone implementation.

THE IDEA IN ONE SENTENCE:
  Measure how fast the acoustic scene is moving through embedding space.
  Baseline = slow, smooth, predictable movement.
  Disturbance = sudden jump to a new region of embedding space.

WHY THIS IS DIFFERENT FROM EVERYTHING TRIED:
  All 12 previous approaches asked: "does this clip sound like disturbance?"
  That question is site-specific — disturbance at AM1 sounds different from AM4.

  This approach asks: "is the soundscape moving unusually fast right now?"
  That question may be site-agnostic — sudden acoustic change is sudden
  acoustic change regardless of what the background sounds like.

VELOCITY DEFINITION:
  v(t) = ||embedding(t) - embedding(t-1)||₂

  Raw velocity is site-specific (fast-moving sites have higher baseline velocity).
  Rolling z-score velocity is site-agnostic:
    z_v(t) = (v(t) - mean(v[t-40:t])) / std(v[t-40:t])

  This is the embedding-space equivalent of your adaptive z-scores —
  same principle but applied to trajectory speed rather than feature values.

THREE VELOCITY SIGNALS TESTED:
  1. Raw velocity (L2 distance between consecutive BirdNET embeddings)
  2. Rolling z-score velocity (adaptive, like your z-scores)
  3. Acceleration (change in velocity — double derivative)

CROSS-SITE ADVANTAGE:
  A high rolling z-score velocity means "moving faster than usual for THIS site"
  not "moving fast in absolute terms". This normalises out site-specific
  baseline movement rates.

COMPARED TO:
  - All previous cross-site approaches (best: 0.364)
  - Adaptive z-scores per-site (best: P=1.000 at AM1)
"""

import os
import numpy as np
import pandas as pd
import zipfile
import io
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (precision_recall_fscore_support,
                              precision_recall_curve)

BASE_DIR = r"C:\Users\BitaMassoudi\KasmirWorldProjects\toucanet-option-b"
ZIP_PATH = r"C:\Users\BitaMassoudi\Downloads\raw_segments_per_clip.zip"
SEED     = 42
ROLLING_N = 40   # rolling window for velocity normalisation

RECORDERS = [
    ('AM4','am4_full_feature_matrix.csv'),
    ('AM2','am2_feature_matrix.csv'),
    ('AM5','am5_feature_matrix.csv'),
    ('AM6','am6_feature_matrix.csv'),
    ('AM1','am1_feature_matrix.csv'),
]


def parse_timestamp(clip_name):
    parts = clip_name.split('_')
    try:
        date = parts[3]; time = parts[4].replace('.wav','')
        return pd.Timestamp(
            f"{date[:4]}-{date[4:6]}-{date[6:8]} "
            f"{time[:2]}:{time[2:4]}:{time[4:6]}")
    except:
        return pd.Timestamp('2025-01-01')


def load_embeddings(clip_names):
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


def compute_velocity_features(emb_seq, rolling_n=40):
    """
    Compute velocity features for a sequence of embeddings.
    Returns array of shape (n, 4):
      col 0: raw velocity
      col 1: rolling z-score velocity (adaptive)
      col 2: acceleration (velocity change)
      col 3: rolling z-score acceleration
    """
    n = len(emb_seq)
    raw_vel = np.zeros(n)
    for t in range(1, n):
        raw_vel[t] = np.linalg.norm(emb_seq[t] - emb_seq[t-1])

    # Acceleration = change in velocity
    accel = np.zeros(n)
    for t in range(2, n):
        accel[t] = abs(raw_vel[t] - raw_vel[t-1])

    # Rolling z-score normalisation
    def rolling_zscore(signal, n=40):
        zs = np.zeros(len(signal))
        for t in range(len(signal)):
            ws = max(0, t-n)
            window = signal[ws:t]
            if len(window) > 2:
                mu = window.mean(); sig = max(window.std(), 1e-6)
                zs[t] = (signal[t] - mu) / sig
            else:
                zs[t] = 0.0
        return zs

    z_vel   = rolling_zscore(raw_vel, rolling_n)
    z_accel = rolling_zscore(accel, rolling_n)

    return np.column_stack([raw_vel, z_vel, accel, z_accel])


def evaluate(y, scores):
    if len(np.unique(y)) < 2: return 0, 0, False
    scores = np.nan_to_num(scores, nan=0.0)
    pr,rc,th = precision_recall_curve(y, scores)
    valid = np.where((pr[:-1]>=0.70) & (rc[:-1]>=0.70))[0]
    if len(valid) > 0:
        bi = valid[np.argmax(pr[valid]+rc[valid])]; bt = float(th[bi])
    else:
        bi = np.argmax(pr[:-1]+rc[:-1]); bt = float(th[bi])
    preds = (scores >= bt).astype(int)
    p,r,_,_ = precision_recall_fscore_support(
        y, preds, average='binary', zero_division=0)
    return p, r, (p>=0.70 and r>=0.70)


if __name__ == '__main__':

    print("Temporal Velocity in Embedding Space")
    print("Griffin's Spring 2026 — standalone implementation")
    print("="*65)

    # Load all data
    print("\nLoading data...")
    all_clip_names = []
    recorder_meta  = {}
    for name, ff in RECORDERS:
        fp = os.path.join(BASE_DIR, ff)
        if not os.path.exists(fp): continue
        df = pd.read_csv(fp)
        clips = df['clip_name'].unique().tolist()
        all_clip_names.extend(clips)
        recorder_meta[name] = df

    print("  Loading BirdNET embeddings...")
    emb_map = load_embeddings(all_clip_names)
    print(f"  Loaded {len(emb_map)} embeddings")

    # Build chronologically sorted per-recorder data
    recorder_data = {}
    for name, df in recorder_meta.items():
        rows = []
        for cn in df['clip_name'].unique():
            if cn not in emb_map: continue
            mask = df['clip_name'].values == cn
            rows.append({
                'clip_name': cn,
                'clip_label': int(df['clip_label'].values[mask][0]),
                'clip_hour': df['clip_hour'].values[mask][0],
                'timestamp': parse_timestamp(cn),
                'emb': emb_map[cn]
            })
        if not rows: continue
        rdf = pd.DataFrame(rows).sort_values('timestamp').reset_index(drop=True)

        # PCA before velocity (reduces noise in distance computation)
        emb_matrix = np.array([r['emb'] for _,r in rdf.iterrows()])
        sc = StandardScaler()
        pca = PCA(n_components=32, random_state=SEED)
        emb_pca = pca.fit_transform(sc.fit_transform(emb_matrix))

        # Compute velocity features
        vel_feats = compute_velocity_features(emb_pca, ROLLING_N)
        rdf['raw_vel']   = vel_feats[:,0]
        rdf['z_vel']     = vel_feats[:,1]
        rdf['accel']     = vel_feats[:,2]
        rdf['z_accel']   = vel_feats[:,3]
        rdf['emb_pca']   = list(emb_pca)

        recorder_data[name] = rdf
        y = rdf['clip_label'].values
        n_sim = int(y.sum())

        # Per-site velocity stats
        base_vel = rdf.loc[y==0, 'raw_vel'].values
        sim_vel  = rdf.loc[y==1, 'raw_vel'].values
        print(f"  {name}: {len(rdf)} clips ({n_sim} sim) | "
              f"baseline velocity={base_vel.mean():.3f} | "
              f"sim velocity={sim_vel.mean():.3f} | "
              f"ratio={sim_vel.mean()/max(base_vel.mean(),1e-6):.2f}x")

    available = list(recorder_data.keys())

    # ── PER-SITE: velocity as anomaly score ───────────────────────────────────
    print(f"\n{'='*65}")
    print("PER-SITE — Rolling Z-Score Velocity as Anomaly Score")
    print("No classifier — purely unsupervised")
    print("="*65)

    per_site_vel = {}
    for name in available:
        rdf = recorder_data[name]
        y   = rdf['clip_label'].values
        # Skip first ROLLING_N clips (no reliable z-score yet)
        valid = np.arange(len(rdf)) >= ROLLING_N
        p,r,beat = evaluate(y[valid], rdf['z_vel'].values[valid])
        per_site_vel[name] = (p,r,beat)
        prev = {'AM4':0.928,'AM2':0.998,'AM5':0.847,'AM6':0.943,'AM1':1.000}
        print(f"  {name}: P={p:.3f} R={r:.3f} {'BEAT' if beat else 'miss'}  "
              f"(prev best: {prev.get(name,0):.3f})")

    # ── CROSS-SITE LODO ───────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("CROSS-SITE LODO — Three velocity signals tested")
    print("="*65)

    # Test 1: Rolling z-score velocity alone (no classifier)
    print("\n  Signal 1: Rolling z-score velocity (no classifier)")
    lodo_vel = {}
    for test_name in available:
        rdf = recorder_data[test_name]
        y_te = rdf['clip_label'].values
        valid = np.arange(len(rdf)) >= ROLLING_N
        if len(np.unique(y_te[valid])) < 2: continue
        p,r,beat = evaluate(y_te[valid], rdf['z_vel'].values[valid])
        lodo_vel[test_name] = (p,r,beat)
        print(f"    {test_name}: P={p:.3f} R={r:.3f} {'BEAT' if beat else 'miss'}")
    avg_vel = np.mean([v[0] for v in lodo_vel.values()])
    print(f"    Avg: {avg_vel:.3f}")

    # Test 2: Velocity + acceleration features with GB classifier (LODO)
    print("\n  Signal 2: Velocity + acceleration features with GB (LODO)")
    lodo_gb = {}
    for test_name in available:
        train_names = [r for r in available if r != test_name]
        feat_cols = ['raw_vel','z_vel','accel','z_accel']

        X_tr = np.vstack([recorder_data[r][feat_cols].values
                          for r in train_names])
        y_tr = np.concatenate([recorder_data[r]['clip_label'].values
                               for r in train_names])
        X_te = recorder_data[test_name][feat_cols].values
        y_te = recorder_data[test_name]['clip_label'].values

        X_tr = np.nan_to_num(X_tr); X_te = np.nan_to_num(X_te)
        if len(np.unique(y_te)) < 2: continue

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)

        n_pos=y_tr.sum(); n_neg=(y_tr==0).sum()
        w=np.where(y_tr==1, n_neg/max(n_pos,1), 1.0)
        gb=GradientBoostingClassifier(n_estimators=100,max_depth=3,
           learning_rate=0.1,random_state=SEED,subsample=0.8)
        gb.fit(X_tr_s,y_tr,sample_weight=w)
        probs=gb.predict_proba(X_te_s)[:,1]
        p,r,beat=evaluate(y_te,probs)
        lodo_gb[test_name]=(p,r,beat)
        print(f"    {test_name}: P={p:.3f} R={r:.3f} {'BEAT' if beat else 'miss'}")
    avg_gb = np.mean([v[0] for v in lodo_gb.values()])
    print(f"    Avg: {avg_gb:.3f}")

    # Test 3: PCA embeddings + velocity features combined (LODO)
    print("\n  Signal 3: BirdNET PCA + velocity combined (LODO)")
    lodo_combo = {}
    for test_name in available:
        train_names = [r for r in available if r != test_name]
        feat_cols = ['raw_vel','z_vel','accel','z_accel']

        def get_combined(rdf):
            emb_mat = np.array(rdf['emb_pca'].tolist())
            vel_mat = rdf[feat_cols].values
            return np.hstack([emb_mat, vel_mat])

        X_tr = np.vstack([get_combined(recorder_data[r]) for r in train_names])
        y_tr = np.concatenate([recorder_data[r]['clip_label'].values for r in train_names])
        X_te = get_combined(recorder_data[test_name])
        y_te = recorder_data[test_name]['clip_label'].values

        X_tr = np.nan_to_num(X_tr); X_te = np.nan_to_num(X_te)
        if len(np.unique(y_te)) < 2: continue

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr); X_te_s = sc.transform(X_te)
        n_pos=y_tr.sum(); n_neg=(y_tr==0).sum()
        w=np.where(y_tr==1,n_neg/max(n_pos,1),1.0)
        gb=GradientBoostingClassifier(n_estimators=100,max_depth=3,
           learning_rate=0.1,random_state=SEED,subsample=0.8)
        gb.fit(X_tr_s,y_tr,sample_weight=w)
        probs=gb.predict_proba(X_te_s)[:,1]
        p,r,beat=evaluate(y_te,probs)
        lodo_combo[test_name]=(p,r,beat)
        print(f"    {test_name}: P={p:.3f} R={r:.3f} {'BEAT' if beat else 'miss'}")
    avg_combo = np.mean([v[0] for v in lodo_combo.values()])
    print(f"    Avg: {avg_combo:.3f}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n\n{'='*65}")
    print("TEMPORAL VELOCITY — FINAL SUMMARY")
    print("="*65)

    print(f"\nPer-site velocity anomaly score (vs adaptive z-score best):")
    prev_ps = {'AM4':0.928,'AM2':0.998,'AM5':0.847,'AM6':0.943,'AM1':1.000}
    for name in available:
        p,r,beat = per_site_vel.get(name,(0,0,False))
        prev = prev_ps.get(name,0)
        print(f"  {name}: P={p:.3f} R={r:.3f} {'BEAT' if beat else 'miss'}  "
              f"({p-prev:+.3f} vs adaptive z-score)")

    print(f"\nCross-site LODO:")
    print(f"  {'Method':<35} {'Avg P':>7} {'vs best':>8}")
    print(f"  {'-'*52}")
    for label, avg, res in [
        ("Z-score velocity (no classifier)", avg_vel, lodo_vel),
        ("Velocity+accel features + GB",     avg_gb,  lodo_gb),
        ("BirdNET PCA + velocity + GB",       avg_combo, lodo_combo),
        ("Previous best (deployment calib.)", 0.364,  {}),
    ]:
        n_beat = sum(v[2] for v in res.values()) if res else 0
        beat_str = f"{n_beat}/5" if res else "—"
        print(f"  {label:<35} {avg:>7.3f} {avg-0.364:>+8.3f}  [{beat_str}]")

    best_avg = max(avg_vel, avg_gb, avg_combo)
    if best_avg > 0.364:
        print(f"\n  IMPROVEMENT: velocity features beat previous best by "
              f"{best_avg-0.364:+.3f}")
        print(f"  Griffin's insight confirmed: temporal dynamics transfer better")
    else:
        print(f"\n  No cross-site improvement from velocity features")
        print(f"  Check per-site velocity ratios above —")
        print(f"  if sim velocity >> baseline velocity, signal exists but")
        print(f"  may need longer temporal context to separate cross-site")

    print("\nDone.")
