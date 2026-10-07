"""
46_generalisation.py
---------------------
Experiments to improve cross-site generalisation.

BACKGROUND:
  Script 36: cross-site training completely failed (0/5 beat 0.70)
  Script 37: all feature sets fail equally cross-site
  Root causes identified:
    1. Z-scores are site-specific by design
    2. Perch embeddings capture site-specific soundscapes
    3. Not enough diverse simulation data

THIS SCRIPT TESTS THREE GENERALISATION STRATEGIES:

Strategy 1 — Remove z-scores cross-site
  Z-scores measure deviation from site-specific hourly baseline.
  They are the MOST site-specific features we have.
  When training cross-site, remove z-scores entirely.
  Use only raw spectral + Perch embeddings.
  Hypothesis: raw features might transfer better than normalised ones.

Strategy 2 — Delta features only cross-site
  Script 44 showed delta features (changes between segments)
  improved AM4 precision by +0.028.
  Delta features capture HOW sound changes within a clip —
  a vehicle approaching creates increasing low-frequency energy
  regardless of which site it is at.
  Hypothesis: change patterns generalise better than absolute values.

Strategy 3 — Site-adversarial feature selection
  First: train a classifier to predict WHICH RECORDER a clip is from
  using only acoustic features.
  Then: identify which features are most predictive of recorder identity.
  Remove those features — they are carrying site-specific information.
  Retrain disturbance classifier without site-identifying features.
  Hypothesis: removing site fingerprint features improves generalisation.

All strategies use leave-one-recorder-out evaluation.
Compared against script 36 baseline (all features, cross-site fails).
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (precision_recall_fscore_support,
                              precision_recall_curve, accuracy_score)

BASE_DIR = r"C:\Users\BitaMassoudi\KasmirWorldProjects\toucanet-option-b"
SEED     = 42
K_SEGS   = 4

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


def build_clip_df(feat_df, emb):
    """Build clip-level features with all approaches."""
    rows = []
    for cn in feat_df['clip_name'].unique():
        mask = feat_df['clip_name'].values == cn
        cl   = feat_df['clip_label'].values[mask][0]
        hour = feat_df['clip_hour'].values[mask][0]
        ce   = emb[mask]
        n_seg= len(ce)
        row  = {'clip_name':cn,'clip_label':cl,'clip_hour':hour}
        for k in SPECTRAL_KEYS:
            if k in feat_df.columns:
                row[k] = feat_df.loc[mask,k].values[0]
        # Aggregated embeddings
        for i,v in enumerate(ce.mean(0)): row[f'agg_mean_{i}']=v
        for i,v in enumerate(ce.std(0)):  row[f'agg_std_{i}']=v
        for i,v in enumerate(ce.max(0)):  row[f'agg_max_{i}']=v
        # Delta features
        if n_seg >= 2:
            deltas=np.diff(ce,axis=0)
            dm=deltas.mean(0); ds=deltas.std(0)
            fd=deltas[0]; ld=deltas[-1]
        else:
            dm=ds=fd=ld=np.zeros(ce.shape[1])
        for i,v in enumerate(dm): row[f'dlt_mean_{i}']=v
        for i,v in enumerate(ds): row[f'dlt_std_{i}']=v
        for i,v in enumerate(fd): row[f'dlt_first_{i}']=v
        for i,v in enumerate(ld): row[f'dlt_last_{i}']=v
        rows.append(row)
    return pd.DataFrame(rows)


def compute_zscores_from_train(train_df, test_df):
    """Z-scores computed from training data only."""
    z_tr_rows,z_te_rows=[],[]
    Z_KEYS=[f"z_{k}" for k in SPECTRAL_KEYS]
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
    for hour in set(test_df['clip_hour'].unique())-seen:
        te_h=test_df['clip_hour']==hour
        if te_h.sum()>0:
            z=pd.DataFrame(0,index=test_df.index[te_h],columns=Z_KEYS)
            z_te_rows.append(z)
    z_tr=pd.concat(z_tr_rows).sort_index() if z_tr_rows else pd.DataFrame(0,index=train_df.index,columns=[f"z_{k}" for k in SPECTRAL_KEYS])
    z_te=pd.concat(z_te_rows).sort_index() if z_te_rows else pd.DataFrame(0,index=test_df.index,columns=[f"z_{k}" for k in SPECTRAL_KEYS])
    return z_tr,z_te


def train_eval(X_tr,y_tr,X_te,y_te):
    """Train GB and find best threshold."""
    sc=StandardScaler(); X_tr_s=sc.fit_transform(X_tr); X_te_s=sc.transform(X_te)
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
    return p,r,(p>=0.70 and r>=0.70),gb


if __name__ == '__main__':

    print("Generalisation Experiments")
    print("="*65)

    # Load all data
    print("Loading data...")
    all_clips={}
    for name,ff,ef,lf in RECORDERS:
        fp=os.path.join(BASE_DIR,ff); ep=os.path.join(BASE_DIR,ef)
        if not os.path.exists(fp) or not os.path.exists(ep): continue
        feat_df=pd.read_csv(fp); emb=np.load(ep)
        n=min(len(feat_df),len(emb))
        feat_df=feat_df.iloc[:n].reset_index(drop=True); emb=emb[:n]
        clip_df=build_clip_df(feat_df,emb)
        clip_df['recorder']=name
        all_clips[name]=clip_df
        print(f"  {name}: {len(clip_df)} clips ({int(clip_df['clip_label'].sum())} sim)")

    available=list(all_clips.keys())
    agg_cols=[c for c in all_clips[available[0]].columns if c.startswith('agg_')]
    dlt_cols=[c for c in all_clips[available[0]].columns if c.startswith('dlt_')]

    # ── Strategy 3 first: identify site-specific features ──────────────────
    print("\n" + "="*55)
    print("Strategy 3 — Site adversarial feature selection")
    print("Finding which features identify recorder site...")

    # Pool all clips and train recorder-ID classifier
    all_df = pd.concat(all_clips.values(),ignore_index=True)
    y_site = all_df['recorder'].map(
        {n:i for i,n in enumerate(available)}).values
    X_spec = all_df[SPECTRAL_KEYS].values.astype(np.float32)
    sc_site=StandardScaler(); X_spec_s=sc_site.fit_transform(X_spec)

    rf=RandomForestClassifier(n_estimators=100,random_state=SEED,n_jobs=-1)
    rf.fit(X_spec_s,y_site)
    site_acc=accuracy_score(y_site,rf.predict(X_spec_s))
    print(f"  Recorder ID accuracy from spectral features: {site_acc:.3f}")
    print(f"  (1.0 = features perfectly identify site, 0.2 = random)")

    # Get feature importances for site prediction
    importances=pd.Series(rf.feature_importances_,index=SPECTRAL_KEYS)
    importances=importances.sort_values(ascending=False)
    print(f"\n  Top 10 most site-specific spectral features:")
    for feat,imp in importances.head(10).items():
        print(f"    {feat:<30} importance={imp:.4f}")

    # Remove top N most site-specific features
    site_features_to_remove=importances.head(10).index.tolist()
    generalised_spec=[k for k in SPECTRAL_KEYS if k not in site_features_to_remove]
    print(f"\n  Keeping {len(generalised_spec)}/{len(SPECTRAL_KEYS)} spectral features")
    print(f"  Removed: {site_features_to_remove}")

    # ── LODO evaluation for each strategy ──────────────────────────────────
    strategies = {
        'S0_baseline':      {'use_spec':True, 'use_z':True,  'use_agg':True,  'use_dlt':False,'use_gen_spec':False},
        'S1_no_zscores':    {'use_spec':True, 'use_z':False, 'use_agg':True,  'use_dlt':False,'use_gen_spec':False},
        'S2_delta_only':    {'use_spec':False,'use_z':False, 'use_agg':False, 'use_dlt':True, 'use_gen_spec':False},
        'S3_adversarial':   {'use_spec':False,'use_z':False, 'use_agg':True,  'use_dlt':False,'use_gen_spec':True},
        'S4_no_z_plus_dlt': {'use_spec':True, 'use_z':False, 'use_agg':True,  'use_dlt':True, 'use_gen_spec':False},
    }

    all_results={s:{} for s in strategies}

    for test_name in available:
        train_names=[r for r in available if r!=test_name]
        test_df =all_clips[test_name].copy().reset_index(drop=True)
        train_df=pd.concat([all_clips[r] for r in train_names],ignore_index=True)
        y_tr=train_df['clip_label'].values.astype(int)
        y_te=test_df['clip_label'].values.astype(int)
        if len(np.unique(y_te))<2: continue

        print(f"\n{'─'*55}")
        print(f"Test: {test_name}  Train: {train_names}")
        print(f"  {'Strategy':<22} {'P':>7} {'R':>7} {'Beat?':>6}")
        print(f"  {'-'*45}")

        # Z-scores from training data
        z_tr,z_te=compute_zscores_from_train(train_df,test_df)

        # PCA on agg embeddings
        pca_agg=PCA(n_components=50,random_state=SEED)
        etr_agg=pca_agg.fit_transform(train_df[agg_cols].values)
        ete_agg=pca_agg.transform(test_df[agg_cols].values)

        # PCA on delta features
        pca_dlt=PCA(n_components=40,random_state=SEED)
        etr_dlt=pca_dlt.fit_transform(train_df[dlt_cols].values)
        ete_dlt=pca_dlt.transform(test_df[dlt_cols].values)

        for sname,sconf in strategies.items():
            parts_tr,parts_te=[],[]
            if sconf['use_spec']:
                parts_tr.append(train_df[SPECTRAL_KEYS].values)
                parts_te.append(test_df[SPECTRAL_KEYS].values)
            if sconf['use_gen_spec']:
                parts_tr.append(train_df[generalised_spec].values)
                parts_te.append(test_df[generalised_spec].values)
            if sconf['use_z']:
                parts_tr.append(z_tr.values)
                parts_te.append(z_te.values)
            if sconf['use_agg']:
                parts_tr.append(etr_agg)
                parts_te.append(ete_agg)
            if sconf['use_dlt']:
                parts_tr.append(etr_dlt)
                parts_te.append(ete_dlt)

            if not parts_tr:
                print(f"  {sname:<22} no features"); continue

            X_tr=np.hstack(parts_tr).astype(np.float32)
            X_te=np.hstack(parts_te).astype(np.float32)
            p,r,beat,_=train_eval(X_tr,y_tr,X_te,y_te)
            flag=" ★" if beat else ""
            print(f"  {sname:<22} {p:>7.3f} {r:>7.3f} {'BEAT' if beat else 'miss':>6}{flag}")
            all_results[sname][test_name]=(p,r,beat)

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n\n"+"="*70)
    print("GENERALISATION SUMMARY — CROSS-SITE PRECISION")
    print("="*70)
    print(f"{'Strategy':<22} {'AM4':>6} {'AM2':>6} {'AM5':>6} {'AM6':>6} {'AM1':>6} {'Beat':>5}")
    print("-"*70)
    for sname in strategies:
        res=all_results[sname]
        vals=[res.get(n,(0,0,False))[0] for n in ['AM4','AM2','AM5','AM6','AM1']]
        n_beat=sum(1 for n in available if all_results[sname].get(n,(0,0,False))[2])
        print(f"{sname:<22} "+" ".join(f"{v:>6.3f}" for v in vals)+f" {n_beat:>4}/5")
    print("="*70)

    # Key finding
    print("\nKey question: which strategy generalises best cross-site?")
    best_s=max(strategies,key=lambda s:sum(
        all_results[s].get(n,(0,0,False))[0] for n in available))
    print(f"Best strategy: {best_s}")
    print(f"Average cross-site P: {np.mean([all_results[best_s].get(n,(0,0,False))[0] for n in available]):.3f}")

    print("\nDone.")
