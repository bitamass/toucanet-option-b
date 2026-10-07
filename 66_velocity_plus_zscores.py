"""
66_velocity_plus_zscores.py
-----------------------------
Combine temporal velocity with adaptive z-scores.

MOTIVATION:
  Script 65: z-score velocity alone gives cross-site P=0.371 (new best)
  Script 52b: adaptive z-scores per-site give P=0.847-1.000

  These two signals measure different things:
    Z-scores:  "is this clip unusual vs recent background?"  (what)
    Velocity:  "is the scene changing unusually fast?"       (how fast)

  They are complementary. A disturbance event causes both:
    1. Unusual acoustic content (z-score spike)
    2. Sudden movement in embedding space (velocity spike)

  A false alarm (loud bird, wind gust) might cause:
    1. Z-score spike — yes, it is unusual
    2. Velocity spike — maybe, but then returns quickly

  A real disturbance (vehicle, human) causes:
    1. Z-score spike — sustained over multiple clips
    2. Velocity spike — sustained, directional movement

  Combining both signals should reduce false alarms and improve
  cross-site transfer because both features are locally normalised.

THREE EXPERIMENTS:
  A. Z-score velocity (from script 65) — baseline: 0.371 cross-site
  B. Adaptive z-scores only (per-site best) — baseline: 0.928-1.000
  C. Velocity + adaptive z-scores combined — the new experiment

DEPLOYMENT:
  Both signals use only unlabelled baseline clips for normalisation.
  No simulation data needed at new site.
  Consistent with "pretrained globally, calibrated locally" model.
"""

import os
import numpy as np
import pandas as pd
import zipfile
import io
from sklearn.ensemble import GradientBoostingClassifier
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
Z_KEYS = [f"z_{k}" for k in SPECTRAL_KEYS]


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


def rolling_zscore_signal(signal, n=40):
    zs = np.zeros(len(signal))
    for t in range(len(signal)):
        ws = max(0, t-n)
        window = signal[ws:t]
        if len(window) > 2:
            mu = window.mean(); sig = max(window.std(), 1e-6)
            zs[t] = (signal[t] - mu) / sig
    return zs


def compute_velocity(emb_pca, rolling_n=40):
    n = len(emb_pca)
    raw_vel = np.zeros(n)
    for t in range(1, n):
        raw_vel[t] = np.linalg.norm(emb_pca[t] - emb_pca[t-1])
    z_vel = rolling_zscore_signal(raw_vel, rolling_n)
    accel = np.zeros(n)
    for t in range(2, n):
        accel[t] = abs(raw_vel[t] - raw_vel[t-1])
    z_accel = rolling_zscore_signal(accel, rolling_n)
    return np.column_stack([raw_vel, z_vel, accel, z_accel])


def compute_adaptive_zscores(spec_vals, rolling_n=40):
    z_rows = []
    for i in range(len(spec_vals)):
        ws = max(0, i-rolling_n)
        window = spec_vals[ws:i]
        if len(window) < 2:
            z = np.zeros(spec_vals.shape[1])
        else:
            rm = window.mean(0); rs = window.std(0)
            rs[rs<1e-6] = 1e-6
            z = (spec_vals[i] - rm) / rs
        z_rows.append(z)
    return np.array(z_rows)


def build_clip_df(feat_df, perch_emb, birdnet_map):
    rows = []
    for cn in feat_df['clip_name'].unique():
        mask = feat_df['clip_name'].values == cn
        if cn not in birdnet_map: continue
        cl   = feat_df['clip_label'].values[mask][0]
        hour = feat_df['clip_hour'].values[mask][0]
        ts   = parse_timestamp(cn)
        pe   = perch_emb[mask]
        row  = {'clip_name':cn,'clip_label':cl,
                'clip_hour':hour,'timestamp':ts,
                'birdnet':birdnet_map[cn]}
        for k in SPECTRAL_KEYS:
            if k in feat_df.columns:
                row[k] = feat_df.loc[mask,k].values[0]
        for i,v in enumerate(pe.mean(0)): row[f'em_{i}']=v
        rows.append(row)
    return pd.DataFrame(rows).sort_values('timestamp').reset_index(drop=True)


def evaluate(y, scores):
    scores = np.nan_to_num(scores)
    if len(np.unique(y)) < 2: return 0,0,False
    pr,rc,th = precision_recall_curve(y,scores)
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

    print("Velocity + Adaptive Z-Scores Combined")
    print("="*65)

    # Load all data
    print("\nLoading data...")
    all_clip_names = []
    recorder_meta  = {}
    for name,ff,ef in RECORDERS:
        fp=os.path.join(BASE_DIR,ff); ep=os.path.join(BASE_DIR,ef)
        if not os.path.exists(fp) or not os.path.exists(ep): continue
        feat_df=pd.read_csv(fp); emb=np.load(ep)
        n=min(len(feat_df),len(emb))
        feat_df=feat_df.iloc[:n].reset_index(drop=True); emb=emb[:n]
        all_clip_names.extend(feat_df['clip_name'].unique().tolist())
        recorder_meta[name]=(feat_df,emb)

    print("  Loading BirdNET embeddings...")
    birdnet_map = load_birdnet(all_clip_names)
    print(f"  Loaded {len(birdnet_map)} BirdNET embeddings")

    # Build per-recorder dataframes with all features
    all_clips = {}
    for name,(feat_df,perch_emb) in recorder_meta.items():
        clip_df = build_clip_df(feat_df, perch_emb, birdnet_map)
        if len(clip_df) == 0: continue

        # Perch PCA for velocity
        ec = [c for c in clip_df.columns if c.startswith('em_')]
        sc_p = StandardScaler()
        pca_p = PCA(n_components=32, random_state=SEED)
        perch_pca = pca_p.fit_transform(sc_p.fit_transform(
            clip_df[ec].values))

        # BirdNET PCA for velocity
        bn_matrix = np.array(clip_df['birdnet'].tolist())
        sc_b = StandardScaler()
        pca_b = PCA(n_components=32, random_state=SEED)
        bn_pca = pca_b.fit_transform(sc_b.fit_transform(bn_matrix))

        # Velocity on Perch
        vel_perch = compute_velocity(perch_pca, ROLLING_N)
        # Velocity on BirdNET
        vel_birdnet = compute_velocity(bn_pca, ROLLING_N)

        # Adaptive z-scores on spectral features
        spec_vals = clip_df[SPECTRAL_KEYS].values.astype(np.float32)
        z_scores = compute_adaptive_zscores(spec_vals, ROLLING_N)

        # Store all feature sets
        clip_df['vel_perch_raw']    = vel_perch[:,0]
        clip_df['vel_perch_z']      = vel_perch[:,1]
        clip_df['vel_bn_raw']       = vel_birdnet[:,0]
        clip_df['vel_bn_z']         = vel_birdnet[:,1]
        clip_df['vel_bn_accel_z']   = vel_birdnet[:,3]
        for i,k in enumerate(Z_KEYS):
            clip_df[k] = z_scores[:,i]

        all_clips[name] = clip_df
        n_sim = int((clip_df['clip_label']==1).sum())
        print(f"  {name}: {len(clip_df)} clips ({n_sim} sim)")

    available = list(all_clips.keys())

    # ── LODO with four combinations ───────────────────────────────────────────
    experiments = {
        'A_vel_bn_z_only':    ['vel_bn_z'],
        'B_zscores_only':     Z_KEYS,
        'C_vel_bn_z+zscores': ['vel_bn_z','vel_bn_accel_z'] + Z_KEYS,
        'D_vel_perch_z+zscores': ['vel_perch_z'] + Z_KEYS,
        'E_all_vel+zscores':  ['vel_perch_z','vel_perch_raw',
                                'vel_bn_z','vel_bn_raw','vel_bn_accel_z'] + Z_KEYS,
    }

    all_results = {}
    for exp_name, feat_cols in experiments.items():
        results = {}
        for test_name in available:
            train_names = [r for r in available if r!=test_name]
            X_tr = np.vstack([all_clips[r][feat_cols].values
                              for r in train_names])
            y_tr = np.concatenate([all_clips[r]['clip_label'].values
                                   for r in train_names])
            X_te = all_clips[test_name][feat_cols].values
            y_te = all_clips[test_name]['clip_label'].values
            X_tr=np.nan_to_num(X_tr); X_te=np.nan_to_num(X_te)
            if len(np.unique(y_te))<2: continue
            sc=StandardScaler()
            X_tr_s=sc.fit_transform(X_tr); X_te_s=sc.transform(X_te)
            n_pos=y_tr.sum(); n_neg=(y_tr==0).sum()
            w=np.where(y_tr==1,n_neg/max(n_pos,1),1.0)
            gb=GradientBoostingClassifier(n_estimators=100,max_depth=3,
               learning_rate=0.1,random_state=SEED,subsample=0.8)
            gb.fit(X_tr_s,y_tr,sample_weight=w)
            probs=gb.predict_proba(X_te_s)[:,1]
            p,r,beat=evaluate(y_te,probs)
            results[test_name]=(p,r,beat)
        avg=np.mean([v[0] for v in results.values()]) if results else 0
        all_results[exp_name]=(results,avg)

    # ── Per-site: vel_bn_z + adaptive z-scores ────────────────────────────────
    print(f"\n{'='*65}")
    print("PER-SITE: Velocity + Adaptive Z-Scores (within-site CV)")
    print("="*65)
    from sklearn.model_selection import StratifiedKFold
    feat_cols_combined = ['vel_bn_z','vel_bn_accel_z'] + Z_KEYS
    prev_ps = {'AM4':0.928,'AM2':0.998,'AM5':0.847,'AM6':0.943,'AM1':1.000}

    for name in available:
        clip_df = all_clips[name].copy().reset_index(drop=True)
        y = clip_df['clip_label'].values.astype(int)
        X = np.nan_to_num(clip_df[feat_cols_combined].values)
        cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
        probs_all = np.zeros(len(y))
        for tr,va in cv.split(X,y):
            sc=StandardScaler()
            Xtr_s=sc.fit_transform(X[tr]); Xva_s=sc.transform(X[va])
            n_pos=y[tr].sum(); n_neg=(y[tr]==0).sum()
            w=np.where(y[tr]==1,n_neg/max(n_pos,1),1.0)
            gb=GradientBoostingClassifier(n_estimators=100,max_depth=3,
               learning_rate=0.1,random_state=SEED,subsample=0.8)
            gb.fit(Xtr_s,y[tr],sample_weight=w)
            probs_all[va]=gb.predict_proba(Xva_s)[:,1]
        p,r,beat=evaluate(y,probs_all)
        prev=prev_ps.get(name,0)
        print(f"  {name}: P={p:.3f} R={r:.3f} {'BEAT' if beat else 'miss'}  "
              f"(prev: {prev:.3f}  change: {p-prev:+.3f})")

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n\n{'='*65}")
    print("VELOCITY + Z-SCORES — CROSS-SITE SUMMARY")
    print("="*65)
    print(f"\n{'Experiment':<30} {'Avg P':>7} {'vs 0.364':>9} {'Beat 0.70':>10}")
    print("-"*60)
    for exp_name,(res,avg) in sorted(all_results.items(),key=lambda x:-x[1][1]):
        n_beat=sum(v[2] for v in res.values())
        marker=" ★" if avg>0.364 else ""
        print(f"{exp_name:<30} {avg:>7.3f} {avg-0.364:>+9.3f} "
              f"{n_beat}/5{marker}")
    print(f"\n  Previous best (velocity z-score, no classifier): 0.371")
    print(f"  Previous best (deployment calibration):          0.364")

    best_exp = max(all_results.items(), key=lambda x: x[1][1])
    best_name, (best_res, best_avg) = best_exp
    print(f"\n  Best this experiment: {best_name} = {best_avg:.3f}")
    if best_avg > 0.371:
        print(f"  NEW BEST — beats velocity-only by {best_avg-0.371:+.3f}")
        print(f"  Combining velocity with z-scores is additive")
    elif best_avg > 0.364:
        print(f"  Beats deployment calibration but not velocity-only")
        print(f"  Velocity signal is best used without a classifier")
    else:
        print(f"  Adding z-scores does not improve over velocity alone")
        print(f"  Velocity signal is site-specific when combined with classifier")

    print("\nDone.")
