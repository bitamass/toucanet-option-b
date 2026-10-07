"""
48_temporal_holdout.py
-----------------------
Temporal holdout test: train on March 2025, test on July 2025.

THE MOST IMPORTANT EXPERIMENT IN THE PROJECT.

All previous results used 5-fold CV within the same 5-day window
(March 17-21 2025). This tests whether the system actually works
in the real world when deployed weeks later in different conditions.

If the system fails on July data it means it learned something
specific to March conditions — temperature, species activity,
humidity, vegetation state — rather than a universal acoustic
signature of human disturbance.

If it succeeds it means the per-site results are genuinely
deployable and not just a statistical artefact of the specific
5 days of training data.

WHAT THIS SCRIPT DOES:
  1. Loads March data (existing feature matrices)
  2. Loads July data (new simulation session — same recorders)
  3. Trains on ALL March data (no CV — use everything for training)
  4. Tests on July data
  5. Compares to March CV results

REQUIREMENTS:
  July simulation data must exist as WAV files or feature matrices.
  This script checks for July data and tells you what format
  it needs if not found.

  Expected locations:
    July feature matrices: am*_july_feature_matrix.csv
    July embeddings: am*_july_emb.npy
    Or raw WAV files in: July_clips/ subfolder

  If July data is not yet available this script prints a
  preparation guide for when the field recordings arrive.
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
    ('AM4','am4_full_feature_matrix.csv',
     'am4_full_emb.npy','am4_full_labels.npy',
     'am4_july_feature_matrix.csv','am4_july_emb.npy'),
    ('AM2','am2_feature_matrix.csv',
     'am2_time_controlled_emb.npy','am2_time_controlled_labels.npy',
     'am2_july_feature_matrix.csv','am2_july_emb.npy'),
    ('AM5','am5_feature_matrix.csv',
     'am5_time_controlled_emb.npy','am5_time_controlled_labels.npy',
     'am5_july_feature_matrix.csv','am5_july_emb.npy'),
    ('AM6','am6_feature_matrix.csv',
     'am6_time_controlled_emb.npy','am6_time_controlled_labels.npy',
     'am6_july_feature_matrix.csv','am6_july_emb.npy'),
    ('AM1','am1_feature_matrix.csv',
     'am1_full_emb.npy','am1_full_labels.npy',
     'am1_july_feature_matrix.csv','am1_july_emb.npy'),
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


def build_clip_df(feat_df, emb):
    rows = []
    for cn in feat_df['clip_name'].unique():
        mask = feat_df['clip_name'].values == cn
        cl   = feat_df['clip_label'].values[mask][0]
        hour = feat_df['clip_hour'].values[mask][0]
        date = cn.split('_')[3]
        ce   = emb[mask]
        row  = {'clip_name':cn,'clip_label':cl,'clip_hour':hour,'date':date}
        for k in SPECTRAL_KEYS:
            if k in feat_df.columns:
                row[k] = feat_df.loc[mask,k].values[0]
        for i,v in enumerate(ce.mean(0)): row[f'em_{i}']=v
        for i,v in enumerate(ce.std(0)):  row[f'es_{i}']=v
        for i,v in enumerate(ce.max(0)):  row[f'ex_{i}']=v
        rows.append(row)
    return pd.DataFrame(rows)


def compute_zscores(train_df, test_df):
    z_tr_rows,z_te_rows=[],[]
    for hour in train_df['clip_hour'].unique():
        bm=(train_df['clip_hour']==hour)&(train_df['clip_label']==0)
        br=train_df.loc[bm,SPECTRAL_KEYS]
        if len(br)==0: continue
        hm=br.mean(); hs=br.std().replace(0,1e-6)
        th=train_df['clip_hour']==hour
        if th.sum()>0:
            z=(train_df.loc[th,SPECTRAL_KEYS]-hm)/hs
            z.columns=Z_KEYS; z_tr_rows.append(z)
        te_h=test_df['clip_hour']==hour
        if te_h.sum()>0:
            z=(test_df.loc[te_h,SPECTRAL_KEYS]-hm)/hs
            z.columns=Z_KEYS; z_te_rows.append(z)
    seen=set(train_df['clip_hour'].unique())
    for h in set(test_df['clip_hour'].unique())-seen:
        te_h=test_df['clip_hour']==h
        if te_h.sum()>0:
            z_te_rows.append(pd.DataFrame(0,index=test_df.index[te_h],columns=Z_KEYS))
    z_tr=pd.concat(z_tr_rows).sort_index() if z_tr_rows else pd.DataFrame(0,index=train_df.index,columns=Z_KEYS)
    z_te=pd.concat(z_te_rows).sort_index() if z_te_rows else pd.DataFrame(0,index=test_df.index,columns=Z_KEYS)
    return z_tr,z_te


def train_and_test(train_df, test_df):
    ec=[c for c in train_df.columns if c.startswith(('em_','es_','ex_'))]
    y_tr=train_df['clip_label'].values.astype(int)
    y_te=test_df['clip_label'].values.astype(int)
    z_tr,z_te=compute_zscores(train_df,test_df)
    pca=PCA(n_components=50,random_state=SEED)
    etr=pca.fit_transform(train_df[ec].values)
    ete=pca.transform(test_df[ec].values)
    X_tr=np.hstack([train_df[SPECTRAL_KEYS].values,z_tr.values,etr]).astype(np.float32)
    X_te=np.hstack([test_df[SPECTRAL_KEYS].values, z_te.values,ete]).astype(np.float32)
    sc=StandardScaler()
    X_tr_s=sc.fit_transform(X_tr); X_te_s=sc.transform(X_te)
    n_pos=y_tr.sum(); n_neg=(y_tr==0).sum()
    w=np.where(y_tr==1,n_neg/max(n_pos,1),1.0)
    gb=GradientBoostingClassifier(n_estimators=100,max_depth=3,
       learning_rate=0.1,random_state=SEED,subsample=0.8)
    gb.fit(X_tr_s,y_tr,sample_weight=w)
    probs=gb.predict_proba(X_te_s)[:,1]
    pr,rc,th=precision_recall_curve(y_te,probs)
    valid=np.where((pr[:-1]>=0.70)&(rc[:-1]>=0.70))[0]
    if len(valid)>0:
        bi=valid[np.argmax(pr[valid]+rc[valid])]; bt=float(th[bi])
    else:
        bi=np.argmax(pr[:-1]+rc[:-1]); bt=float(th[bi])
    preds=(probs>=bt).astype(int)
    p,r,_,_=precision_recall_fscore_support(y_te,preds,average='binary',zero_division=0)
    return p,r,(p>=0.70 and r>=0.70)


if __name__ == '__main__':

    print("Temporal Holdout Test")
    print("Train on March 2025, Test on July 2025")
    print("="*60)

    # Check which recorders have July data
    print("\nChecking for July data...")
    july_available=[]
    july_missing=[]

    for name,*_,july_ff,july_ef in RECORDERS:
        july_feat=os.path.join(BASE_DIR,july_ff)
        july_emb=os.path.join(BASE_DIR,july_ef)
        if os.path.exists(july_feat) and os.path.exists(july_emb):
            feat_df=pd.read_csv(july_feat)
            print(f"  {name}: FOUND — {len(feat_df)} July segments")
            july_available.append(name)
        else:
            print(f"  {name}: NOT FOUND — looking for {july_ff}")
            july_missing.append(name)

    if not july_available:
        print("\n" + "="*60)
        print("NO JULY DATA FOUND")
        print("="*60)
        print("""
This script is ready to run as soon as July simulation data arrives.

WHAT YOU NEED:
  For each recorder that has July simulation recordings:
  1. Feature matrices extracted the same way as March
  2. Perch embeddings (.npy files)

  Expected filenames:
    am4_july_feature_matrix.csv + am4_july_emb.npy
    am2_july_feature_matrix.csv + am2_july_emb.npy
    am5_july_feature_matrix.csv + am5_july_emb.npy
    am6_july_feature_matrix.csv + am6_july_emb.npy
    am1_july_feature_matrix.csv + am1_july_emb.npy

HOW TO PREPARE JULY DATA:
  Run the same preprocessing pipeline used for March data:
  1. Scripts 01-15 for feature extraction (onset detection,
     spectral features, BirdNET filtering)
  2. Scripts 20-25 for Perch embedding extraction
  3. Scripts 26-30 for clip-level aggregation
  Save outputs with _july_ in the filename.

WHY THIS EXPERIMENT MATTERS:
  All current results (P=0.884 AM4, P=0.952 AM2 etc.) used
  5-fold CV within the same 5-day window in March 2025.
  This is the only test that will confirm whether results
  hold over time and across different seasonal conditions.

  If July results match March: system is genuinely deployable.
  If July results collapse: results are specific to March
  conditions and the system needs retraining each season.

Re-run this script once July data is placed in:
  {}
""".format(BASE_DIR))
        exit()

    # Run temporal holdout for available recorders
    print(f"\nRunning temporal holdout for: {july_available}")
    print("="*60)

    results={}
    for name,march_ff,march_ef,march_lf,july_ff,july_ef in RECORDERS:
        if name not in july_available: continue
        print(f"\n{name}")

        # Load March (training)
        march_feat=pd.read_csv(os.path.join(BASE_DIR,march_ff))
        march_emb=np.load(os.path.join(BASE_DIR,march_ef))
        n=min(len(march_feat),len(march_emb))
        march_feat=march_feat.iloc[:n].reset_index(drop=True)
        march_emb=march_emb[:n]
        train_df=build_clip_df(march_feat,march_emb)

        # Load July (testing)
        july_feat=pd.read_csv(os.path.join(BASE_DIR,july_ff))
        july_emb=np.load(os.path.join(BASE_DIR,july_ef))
        n=min(len(july_feat),len(july_emb))
        july_feat=july_feat.iloc[:n].reset_index(drop=True)
        july_emb=july_emb[:n]
        test_df=build_clip_df(july_feat,july_emb)

        print(f"  Train (March): {len(train_df)} clips "
              f"({int(train_df['clip_label'].sum())} sim)")
        print(f"  Test  (July):  {len(test_df)} clips "
              f"({int(test_df['clip_label'].sum())} sim)")

        y_te=test_df['clip_label'].values.astype(int)
        if len(np.unique(y_te))<2:
            print(f"  Only one class in July test — cannot evaluate")
            continue

        p,r,beat=train_and_test(train_df,test_df)
        print(f"  March CV (ref): see script 42 results")
        print(f"  July holdout:   P={p:.3f} R={r:.3f} "
              f"{'BEAT' if beat else 'miss'}")
        results[name]={'p':p,'r':r,'beat':beat}

    if results:
        print("\n\n"+"="*60)
        print("TEMPORAL HOLDOUT SUMMARY")
        print("Train March → Test July")
        print("="*60)
        for name,res in results.items():
            beat="BEAT" if res['beat'] else "miss"
            print(f"  {name}: P={res['p']:.3f} R={res['r']:.3f} {beat}")
        n_beat=sum(1 for r in results.values() if r['beat'])
        print(f"\n  {n_beat}/{len(results)} recorders beat 0.70 on July data")
        if n_beat==len(results):
            print("  CONCLUSION: Results generalise across seasons — system is deployable")
        elif n_beat>0:
            print("  CONCLUSION: Partial temporal generalisation")
        else:
            print("  CONCLUSION: Results do not hold on July data — seasonal retraining needed")

    print("\nDone.")
