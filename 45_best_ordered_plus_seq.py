"""
45_best_ordered_plus_seq.py
----------------------------
Combines best ordered segment approach per recorder
with sequence voting.

FINDINGS FROM SCRIPT 44:
  AM4: delta features (B) beat baseline by +0.028
  AM5: ordered concat (A) beat baseline by +0.036
  AM2, AM6, AM1: baseline (mean/std/max) still best

THIS SCRIPT:
  Uses the best approach per recorder then applies
  sequence voting on top — testing whether the combination
  pushes above current best results (script 42 + seq voting).

  AM4: delta features + seq voting  (was P=0.884 seq only)
  AM5: ordered concat + seq voting  (was P=0.728 seq only)
  AM2: baseline + seq voting        (reference)
  AM6: baseline + seq voting        (reference)
  AM1: baseline + seq voting        (reference)

BEST APPROACH MAP:
  AM4 → Approach B (delta features)
  AM5 → Approach A (ordered concat, K=4 segments)
  AM2 → Approach BASE (mean/std/max)
  AM6 → Approach BASE (mean/std/max)
  AM1 → Approach BASE (mean/std/max)
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
K_SEGS   = 4

RECORDERS = [
    ('AM4','am4_full_feature_matrix.csv',
     'am4_full_emb.npy','am4_full_labels.npy',
     'B', 0.884, 0.835),
    ('AM2','am2_feature_matrix.csv',
     'am2_time_controlled_emb.npy','am2_time_controlled_labels.npy',
     'BASE', 0.952, 0.876),
    ('AM5','am5_feature_matrix.csv',
     'am5_time_controlled_emb.npy','am5_time_controlled_labels.npy',
     'A', 0.728, 0.902),
    ('AM6','am6_feature_matrix.csv',
     'am6_time_controlled_emb.npy','am6_time_controlled_labels.npy',
     'BASE', 0.875, 0.794),
    ('AM1','am1_feature_matrix.csv',
     'am1_full_emb.npy','am1_full_labels.npy',
     'BASE', 0.961, 0.831),
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

WINDOWS = [
    {'N':3,  'K':2},{'N':5,  'K':3},{'N':5,  'K':4},
    {'N':10, 'K':6},{'N':10, 'K':7},{'N':15, 'K':9},
]


def build_clip_df(feat_df, emb, approach):
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
        # Always include aggregated
        for i,v in enumerate(ce.mean(0)): row[f'agg_mean_{i}']=v
        for i,v in enumerate(ce.std(0)):  row[f'agg_std_{i}']=v
        for i,v in enumerate(ce.max(0)):  row[f'agg_max_{i}']=v
        if approach == 'A':
            for k in range(K_SEGS):
                seg = ce[k] if k < n_seg else np.zeros(ce.shape[1])
                for i,v in enumerate(seg): row[f'ord_s{k}_{i}']=v
        if approach == 'B':
            if n_seg >= 2:
                deltas     = np.diff(ce, axis=0)
                delta_mean = deltas.mean(0)
                delta_std  = deltas.std(0)
                first_d    = deltas[0]
                last_d     = deltas[-1]
            else:
                delta_mean=delta_std=first_d=last_d=np.zeros(ce.shape[1])
            for i,v in enumerate(delta_mean):  row[f'dlt_mean_{i}']=v
            for i,v in enumerate(delta_std):   row[f'dlt_std_{i}']=v
            for i,v in enumerate(first_d):     row[f'dlt_first_{i}']=v
            for i,v in enumerate(last_d):      row[f'dlt_last_{i}']=v
        rows.append(row)
    return pd.DataFrame(rows)


def run_cv_get_probs(clip_df):
    meta = {'clip_name','clip_label','clip_hour'}
    emb_cols = [c for c in clip_df.columns
                if c not in meta and c not in SPECTRAL_KEYS]
    cn = clip_df['clip_name'].values
    cl = clip_df['clip_label'].values
    cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
    probs = np.zeros(len(clip_df),dtype=np.float32)
    for tr,va in cv.split(cn,cl):
        tdf=clip_df.iloc[tr].copy().reset_index(drop=True)
        vdf=clip_df.iloc[va].copy().reset_index(drop=True)
        ytr=cl[tr].astype(int)
        z_tr_rows,z_va_rows=[],[]
        for hour in tdf['clip_hour'].unique():
            bm=(tdf['clip_hour']==hour)&(tdf['clip_label']==0)
            br=tdf.loc[bm,SPECTRAL_KEYS]
            if len(br)==0: continue
            hm=br.mean(); hs=br.std().replace(0,1e-6)
            th=tdf['clip_hour']==hour
            if th.sum()>0:
                z=(tdf.loc[th,SPECTRAL_KEYS]-hm)/hs
                z.columns=Z_KEYS; z_tr_rows.append(z)
            vh=vdf['clip_hour']==hour
            if vh.sum()>0:
                z=(vdf.loc[vh,SPECTRAL_KEYS]-hm)/hs
                z.columns=Z_KEYS; z_va_rows.append(z)
        z_tr=pd.concat(z_tr_rows).sort_index() if z_tr_rows else pd.DataFrame(0,index=tdf.index,columns=Z_KEYS)
        z_va=pd.concat(z_va_rows).sort_index() if z_va_rows else pd.DataFrame(0,index=vdf.index,columns=Z_KEYS)
        n_comp=min(80,len(emb_cols)//10,len(tdf)-1)
        pca=PCA(n_components=n_comp,random_state=SEED)
        etr=pca.fit_transform(tdf[emb_cols].values)
        eva=pca.transform(vdf[emb_cols].values)
        X_tr=np.hstack([tdf[SPECTRAL_KEYS].values,z_tr.values,etr]).astype(np.float32)
        X_va=np.hstack([vdf[SPECTRAL_KEYS].values,z_va.values,eva]).astype(np.float32)
        sc=StandardScaler()
        X_tr_s=sc.fit_transform(X_tr); X_va_s=sc.transform(X_va)
        n_pos=ytr.sum(); n_neg=(ytr==0).sum()
        w=np.where(ytr==1,n_neg/max(n_pos,1),1.0)
        gb=GradientBoostingClassifier(n_estimators=100,max_depth=3,
           learning_rate=0.1,random_state=SEED,subsample=0.8)
        gb.fit(X_tr_s,ytr,sample_weight=w)
        probs[va]=gb.predict_proba(X_va_s)[:,1]
    return probs, cl.astype(int)


def apply_seq_voting(clip_df, probs, bt, N, K):
    idx   = clip_df['clip_name'].argsort().values
    sp    = probs[idx]
    flags = (sp>=bt).astype(int)
    fp    = np.zeros(len(flags),dtype=int)
    for i in range(len(flags)):
        ws=max(0,i-N+1)
        if flags[ws:i+1].sum()>=K: fp[i]=1
    result=np.zeros(len(clip_df),dtype=int); result[idx]=fp
    return result


def evaluate(y, preds):
    p,r,_,_=precision_recall_fscore_support(y,preds,average='binary',zero_division=0)
    return p,r,(p>=0.70 and r>=0.70)


if __name__ == '__main__':

    print("Best Ordered Approach + Sequence Voting")
    print("="*65)

    all_results = {}

    for name,ff,ef,lf,approach,prev_p,prev_r in RECORDERS:
        print(f"\n{'─'*55}")
        print(f"{name}  approach={approach}  prev best: P={prev_p:.3f} R={prev_r:.3f}")

        feat_df=pd.read_csv(os.path.join(BASE_DIR,ff))
        emb=np.load(os.path.join(BASE_DIR,ef))
        n=min(len(feat_df),len(emb))
        feat_df=feat_df.iloc[:n].reset_index(drop=True); emb=emb[:n]

        clip_df=build_clip_df(feat_df,emb,approach)
        print(f"  {len(clip_df)} clips  features: {len([c for c in clip_df.columns if c not in {'clip_name','clip_label','clip_hour'}])} cols")

        probs,y=run_cv_get_probs(clip_df)

        # Find base threshold
        pr,rc,th=precision_recall_curve(y,probs)
        valid=np.where((pr[:-1]>=0.70)&(rc[:-1]>=0.70))[0]
        if len(valid)>0:
            bi=valid[np.argmax(pr[valid]+rc[valid])]; bt=float(th[bi])
        else:
            bi=np.argmax(pr[:-1]+rc[:-1]); bt=float(th[bi])

        preds_single=(probs>=bt).astype(int)
        ps,rs,bs=evaluate(y,preds_single)
        print(f"  Single-clip: P={ps:.3f} R={rs:.3f} {'BEAT' if bs else 'miss'}")

        best_p,best_r,best_beat,best_cfg=ps,rs,bs,"single"
        print(f"  Sequence voting:")
        for w in WINDOWS:
            preds_seq=apply_seq_voting(clip_df,probs,bt,w['N'],w['K'])
            p,r,beat=evaluate(y,preds_seq)
            flag=" ★" if beat and p>best_p else ""
            print(f"    N={w['N']:2d} K={w['K']:2d}: P={p:.3f} R={r:.3f} {'BEAT' if beat else 'miss'}{flag}")
            if beat and p+r > best_p+best_r:
                best_p,best_r,best_beat=p,r,beat
                best_cfg=f"N={w['N']} K={w['K']}"

        diff_p=best_p-prev_p
        diff_r=best_r-prev_r
        print(f"\n  Best: P={best_p:.3f} R={best_r:.3f} [{best_cfg}]")
        print(f"  vs prev best: P{diff_p:+.3f} R{diff_r:+.3f}")
        all_results[name]={'p':best_p,'r':best_r,'cfg':best_cfg,
                           'prev_p':prev_p,'prev_r':prev_r,'approach':approach}

    print("\n\n"+"="*70)
    print("FINAL SUMMARY — BEST ORDERED + SEQ VOTING vs PREVIOUS BEST")
    print("="*70)
    print(f"{'Rec':<6} {'Approach':>8} {'Prev P':>8} {'Prev R':>8} "
          f"{'New P':>7} {'New R':>7} {'Change P':>9} {'Config'}")
    print("-"*70)
    for name,*_,approach,prev_p,prev_r in RECORDERS:
        if name not in all_results: continue
        r=all_results[name]
        dp=r['p']-prev_p; dr=r['r']-prev_r
        print(f"{name:<6} {approach:>8} {prev_p:>8.3f} {prev_r:>8.3f} "
              f"{r['p']:>7.3f} {r['r']:>7.3f} {dp:>+9.3f} {r['cfg']}")
    print("="*70)
    print("Done.")
