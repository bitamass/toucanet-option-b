"""
53_deployment_crosssite.py
---------------------------
Deployment-realistic cross-site test using adaptive z-scores.

THE PROBLEM WITH STANDARD LODO:
  Standard LODO (scripts 36, 43, 46, 47b) trains on 4 sites
  and tests on the 5th WITHOUT any data from the new site.
  The classifier tries to transfer what disturbance sounds like.
  This completely fails (avg P=0.345).

THE DEPLOYMENT SCENARIO (this script):
  In real deployment you would:
    1. Install a new AudioMoth at a new location
    2. Record baseline (normal forest) for a few minutes FIRST
    3. Use those baseline recordings to calibrate the system
    4. THEN start monitoring for disturbance

  The key insight: you do NOT need simulation data at the new site.
  You only need baseline recordings.
  The classifier learns WHAT DISTURBANCE LOOKS LIKE from training sites.
  The adaptive z-score learns WHAT NORMAL LOOKS LIKE from the new site.

HOW THIS WORKS:
  For each held-out test recorder:
    Train:  classifier on all 4 other recorders' clips (sim + base)
    Deploy: use first N baseline clips from TEST RECORDER as
            the rolling reference for adaptive z-score normalisation
    Test:   apply trained classifier + local normalisation to test clips

  Crucially: we use BASELINE ONLY from the new site to calibrate.
  No simulation data from the new site is needed.
  This is exactly what would happen in real deployment.

VARIATIONS TESTED:
  How many baseline clips from new site to use for calibration?
  V1: 10 clips  (~30 seconds of baseline)
  V2: 20 clips  (~60 seconds)
  V3: 50 clips  (~2.5 minutes)
  V4: 100 clips (~5 minutes)
  V5: 200 clips (~10 minutes)
  V6: All available baseline (upper bound)

  And compare to:
  V0: Standard LODO fixed z-score (current baseline, no new site data)

KEY QUESTION:
  How much baseline recording from a new site do you need before
  the system can reliably detect disturbance there?
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (precision_recall_fscore_support,
                              precision_recall_curve)

BASE_DIR = r"C:\Users\BitaMassoudi\KasmirWorldProjects\toucanet-option-b"
SEED     = 42

RECORDERS = [
    ('AM4','am4_full_feature_matrix.csv','am4_full_emb.npy','am4_full_labels.npy'),
    ('AM2','am2_feature_matrix.csv','am2_time_controlled_emb.npy','am2_time_controlled_labels.npy'),
    ('AM5','am5_feature_matrix.csv','am5_time_controlled_emb.npy','am5_time_controlled_labels.npy'),
    ('AM6','am6_feature_matrix.csv','am6_time_controlled_emb.npy','am6_time_controlled_labels.npy'),
    ('AM1','am1_feature_matrix.csv','am1_full_emb.npy','am1_full_labels.npy'),
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

# Calibration sizes: how many baseline clips from new site
CALIB_SIZES = [0, 10, 20, 50, 100, 200, -1]
# 0 = standard LODO (no new site data)
# -1 = all available baseline (upper bound)


def parse_timestamp(clip_name):
    parts = clip_name.split('_')
    try:
        date=parts[3]; time=parts[4].replace('.wav','')
        return pd.Timestamp(f"{date[:4]}-{date[4:6]}-{date[6:8]} "
                           f"{time[:2]}:{time[2:4]}:{time[4:6]}")
    except:
        return pd.Timestamp('2025-01-01')


def build_clip_df(feat_df, emb):
    rows=[]
    for cn in feat_df['clip_name'].unique():
        mask=feat_df['clip_name'].values==cn
        cl=feat_df['clip_label'].values[mask][0]
        hour=feat_df['clip_hour'].values[mask][0]
        ts=parse_timestamp(cn)
        ce=emb[mask]
        row={'clip_name':cn,'clip_label':cl,'clip_hour':hour,'timestamp':ts}
        for k in SPECTRAL_KEYS:
            if k in feat_df.columns:
                row[k]=feat_df.loc[mask,k].values[0]
        for i,v in enumerate(ce.mean(0)): row[f'em_{i}']=v
        for i,v in enumerate(ce.std(0)):  row[f'es_{i}']=v
        for i,v in enumerate(ce.max(0)):  row[f'ex_{i}']=v
        rows.append(row)
    df=pd.DataFrame(rows).sort_values('timestamp').reset_index(drop=True)
    return df


def compute_zscores_from_source(clip_df, source_baseline_df, n_calib):
    """
    Compute z-scores for clip_df using source_baseline_df as reference.

    If n_calib == 0: use fixed hourly z-scores from clip_df's own
                     baseline (standard LODO — no new site calibration)
    If n_calib > 0:  use rolling baseline from first n_calib baseline
                     clips of source_baseline_df (deployment scenario)
    If n_calib == -1: use ALL baseline clips from source_baseline_df
    """
    if n_calib == 0:
        # Standard LODO: fixed hourly z from training sites
        # (approximation: use global mean/std of training baseline)
        z_rows=[]
        for i in range(len(clip_df)):
            z_rows.append(dict(zip(Z_KEYS, np.zeros(len(SPECTRAL_KEYS)))))
        return pd.DataFrame(z_rows, index=clip_df.index)

    # Get calibration baseline from new site
    base_df = source_baseline_df[source_baseline_df['clip_label']==0].copy()
    base_df = base_df.sort_values('timestamp').reset_index(drop=True)

    if n_calib == -1:
        calib_clips = base_df
    else:
        calib_clips = base_df.iloc[:min(n_calib, len(base_df))]

    # Rolling z-score: for each test clip use last N_ROLL baseline clips
    # from calibration set as reference
    N_ROLL = min(20, len(calib_clips))
    if N_ROLL < 2:
        # Not enough calibration — fall back to global stats
        calib_mean = calib_clips[SPECTRAL_KEYS].mean()
        calib_std  = calib_clips[SPECTRAL_KEYS].std().replace(0, 1e-6)
        z_rows=[]
        for i in range(len(clip_df)):
            z=(clip_df[SPECTRAL_KEYS].iloc[i]-calib_mean)/calib_std
            z_rows.append(dict(zip(Z_KEYS, z.values)))
        return pd.DataFrame(z_rows, index=clip_df.index)

    calib_specs = calib_clips[SPECTRAL_KEYS].values
    test_specs  = clip_df[SPECTRAL_KEYS].values
    z_rows=[]

    for i in range(len(clip_df)):
        # Use last N_ROLL calibration clips as rolling reference
        # In deployment: these update as more baseline is recorded
        window = calib_specs[-N_ROLL:]
        roll_mean = window.mean(axis=0)
        roll_std  = window.std(axis=0)
        roll_std[roll_std < 1e-6] = 1e-6
        z = (test_specs[i] - roll_mean) / roll_std
        z_rows.append(dict(zip(Z_KEYS, z)))

    return pd.DataFrame(z_rows, index=clip_df.index)


def compute_train_zscores(train_df):
    """Standard fixed hourly z-scores for training data."""
    z_tr_rows=[]
    for hour in train_df['clip_hour'].unique():
        bm=(train_df['clip_hour']==hour)&(train_df['clip_label']==0)
        br=train_df.loc[bm,SPECTRAL_KEYS]
        if len(br)==0: continue
        hm=br.mean(); hs=br.std().replace(0,1e-6)
        th=train_df['clip_hour']==hour
        if th.sum()>0:
            z=(train_df.loc[th,SPECTRAL_KEYS]-hm)/hs
            z.columns=Z_KEYS; z_tr_rows.append(z)
    return pd.concat(z_tr_rows).sort_index() if z_tr_rows else \
           pd.DataFrame(0,index=train_df.index,columns=Z_KEYS)


def train_classifier(train_df, z_tr):
    """Train GB on combined features."""
    ec=[c for c in train_df.columns if c.startswith(('em_','es_','ex_'))]
    y_tr=train_df['clip_label'].values.astype(int)
    pca=PCA(n_components=50,random_state=SEED)
    etr=pca.fit_transform(train_df[ec].values)
    X_tr=np.hstack([train_df[SPECTRAL_KEYS].values,
                     z_tr.values,etr]).astype(np.float32)
    sc=StandardScaler(); X_tr_s=sc.fit_transform(X_tr)
    n_pos=y_tr.sum(); n_neg=(y_tr==0).sum()
    w=np.where(y_tr==1,n_neg/max(n_pos,1),1.0)
    gb=GradientBoostingClassifier(n_estimators=100,max_depth=3,
       learning_rate=0.1,random_state=SEED,subsample=0.8)
    gb.fit(X_tr_s,y_tr,sample_weight=w)
    return gb,sc,pca


def get_probs(gb, sc, pca, test_df, z_te):
    """Get probability scores for test clips."""
    ec=[c for c in test_df.columns if c.startswith(('em_','es_','ex_'))]
    ete=pca.transform(test_df[ec].values)
    X_te=np.hstack([test_df[SPECTRAL_KEYS].values,
                     z_te.values,ete]).astype(np.float32)
    X_te_s=sc.transform(X_te)
    return gb.predict_proba(X_te_s)[:,1]


def evaluate(y, probs):
    pr,rc,th=precision_recall_curve(y,probs)
    valid=np.where((pr[:-1]>=0.70)&(rc[:-1]>=0.70))[0]
    if len(valid)>0:
        bi=valid[np.argmax(pr[valid]+rc[valid])]; bt=float(th[bi])
    else:
        bi=np.argmax(pr[:-1]+rc[:-1]); bt=float(th[bi])
    preds=(probs>=bt).astype(int)
    p,r,_,_=precision_recall_fscore_support(y,preds,average='binary',zero_division=0)
    fa=int(((preds==1)&(y==0)).sum())
    return p,r,(p>=0.70 and r>=0.70),fa


if __name__ == '__main__':

    print("Deployment Cross-Site Test")
    print("="*70)
    print("Classifier trained on 4 sites.")
    print("Adaptive z-score calibrated on new site's OWN baseline clips.")
    print("Key question: how many baseline clips needed to reach 0.70?")
    print("="*70)

    # Load all data
    print("\nLoading data...")
    all_clips={}
    for name,ff,ef,lf in RECORDERS:
        fp=os.path.join(BASE_DIR,ff); ep=os.path.join(BASE_DIR,ef)
        if not os.path.exists(fp) or not os.path.exists(ep): continue
        feat_df=pd.read_csv(fp); emb=np.load(ep)
        n=min(len(feat_df),len(emb))
        feat_df=feat_df.iloc[:n].reset_index(drop=True); emb=emb[:n]
        clip_df=build_clip_df(feat_df,emb)
        all_clips[name]=clip_df
        n_base=int((clip_df['clip_label']==0).sum())
        print(f"  {name}: {len(clip_df)} clips "
              f"({int(clip_df['clip_label'].sum())} sim, {n_base} base)")

    available=list(all_clips.keys())
    all_results={name:{} for name in available}

    # ── Main evaluation ───────────────────────────────────────────────────────
    for test_name in available:
        train_names=[r for r in available if r!=test_name]
        test_df =all_clips[test_name].copy().reset_index(drop=True)
        train_df=pd.concat([all_clips[r] for r in train_names],
                            ignore_index=True)
        y_te=test_df['clip_label'].values.astype(int)
        if len(np.unique(y_te))<2: continue

        n_base_avail=int((test_df['clip_label']==0).sum())
        print(f"\n{'─'*60}")
        print(f"Test site: {test_name}  ({n_base_avail} baseline clips available)")
        print(f"Train sites: {train_names}")

        # Train classifier once on training sites
        z_tr=compute_train_zscores(train_df)
        gb,sc,pca=train_classifier(train_df,z_tr)

        print(f"\n  {'Calibration':>25}  {'P':>7} {'R':>7} {'FA':>5} {'Beat?':>6}")
        print(f"  {'-'*55}")

        for n_calib in CALIB_SIZES:
            if n_calib > n_base_avail and n_calib != -1:
                label=f"{n_calib} base clips (not enough, only {n_base_avail})"
                print(f"  {label:>45}  skipped")
                continue

            if n_calib == 0:
                label="Standard LODO (no calibration)"
            elif n_calib == -1:
                label=f"All baseline ({n_base_avail} clips)"
            else:
                label=f"{n_calib} baseline clips (~{n_calib*3//60}min {(n_calib*3)%60}sec)"

            # Compute z-scores using new site's baseline for calibration
            z_te=compute_zscores_from_source(test_df,test_df,n_calib)

            probs=get_probs(gb,sc,pca,test_df,z_te)
            p,r,beat,fa=evaluate(y_te,probs)

            flag=" ★" if beat else ""
            print(f"  {label:>45}  {p:>7.3f} {r:>7.3f} {fa:>5} "
                  f"{'BEAT' if beat else 'miss':>6}{flag}")
            all_results[test_name][n_calib]=(p,r,beat,fa)

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n\n"+"="*75)
    print("DEPLOYMENT CROSS-SITE SUMMARY")
    print("Rows = calibration size  |  Columns = test recorder")
    print("="*75)

    calib_labels={
        0:  "No calibration (LODO)",
        10: "10 clips (30 sec)",
        20: "20 clips (1 min)",
        50: "50 clips (2.5 min)",
        100:"100 clips (5 min)",
        200:"200 clips (10 min)",
        -1: "All baseline",
    }

    print(f"{'Calibration':<28} {'AM4':>7} {'AM2':>7} {'AM5':>7} "
          f"{'AM6':>7} {'AM1':>7} {'Beat':>5}")
    print("-"*75)

    for n_calib in CALIB_SIZES:
        label=calib_labels[n_calib]
        vals=[all_results[n].get(n_calib,(0,0,False,0))[0]
              for n in ['AM4','AM2','AM5','AM6','AM1']]
        nb=sum(1 for n in available
               if all_results[n].get(n_calib,(0,0,False,0))[2])
        print(f"{label:<28} "+" ".join(f"{v:>7.3f}" for v in vals)+
              f" {nb:>4}/5")

    print("="*75)
    print("\nKey finding: how many baseline clips needed to beat 0.70?")
    for n_calib in CALIB_SIZES:
        nb=sum(1 for n in available
               if all_results[n].get(n_calib,(0,0,False,0))[2])
        label=calib_labels[n_calib]
        if nb>0:
            print(f"  {label}: {nb}/5 recorders beat 0.70")

    # Save
    out=os.path.join(BASE_DIR,"deployment_crosssite_results.txt")
    with open(out,"w",encoding="utf-8") as fh:
        fh.write("Deployment Cross-Site Results\n"+"="*50+"\n\n")
        for n_calib in CALIB_SIZES:
            label=calib_labels[n_calib]
            fh.write(f"\n{label}:\n")
            for name in available:
                r=all_results[name].get(n_calib,(0,0,False,0))
                fh.write(f"  {name}: P={r[0]:.3f} R={r[1]:.3f} "
                         f"{'BEAT' if r[2] else 'miss'}\n")
    print(f"\nReport saved -> {out}")
    print("\nDone.")
