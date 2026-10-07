"""
44_ordered_segments.py
-----------------------
NLP-style ordered segment modeling.

MOTIVATION (Sammy's suggestion):
  Current clip-level approach aggregates Perch embeddings with
  mean/std/max — which loses temporal order within the clip.

  A 3-second clip with a vehicle approaching might have segments
  that go: [quiet] [quiet] [rumble starts] [louder] [loud] [loud]
  Averaging those gives a medium-loudness embedding.
  But the PATTERN of change — quiet then building — is diagnostic.

  NLP models treat sentences as ordered sequences of word embeddings.
  We treat clips as ordered sequences of segment embeddings.
  The ORDER of segments might carry signal that aggregation destroys.

THREE APPROACHES TESTED:

  Approach A — Concatenate ordered segments
    Take the first K segments in chronological order.
    Concatenate their Perch embeddings: K × 1536 features.
    Gives classifier direct access to the temporal sequence.
    K=4 tested (most clips have ≥4 segments).

  Approach B — Delta features (differences between segments)
    Compute embedding[i+1] - embedding[i] for each consecutive pair.
    Captures HOW the sound changes within the clip.
    A vehicle approaching creates a directional change pattern.
    A random noise creates random changes.

  Approach C — Combined: aggregated + ordered + deltas
    Mean/std/max (script 34 approach) PLUS first 4 segments
    in order PLUS delta features.
    Most information available to the classifier.

All approaches use corrected clip-level CV with no leakage.
Compared against script 34 baseline (mean/std/max only).

Runtime: about 30-40 minutes for all approaches on all recorders.
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
K_SEGS   = 4   # number of ordered segments to use

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


def build_clip_df(feat_df, emb, approach):
    """
    Build clip-level features using different segment ordering approaches.

    approach A: concat first K segments in order
    approach B: delta features (differences between consecutive segments)
    approach C: agg (mean/std/max) + ordered + deltas combined
    approach BASE: mean/std/max only (baseline comparison)
    """
    rows = []
    for cn in feat_df['clip_name'].unique():
        mask  = feat_df['clip_name'].values == cn
        cl    = feat_df['clip_label'].values[mask][0]
        hour  = feat_df['clip_hour'].values[mask][0]
        # Segments in original order (onset order = temporal order)
        ce    = emb[mask]   # shape: (n_segs, 1536)
        n_seg = len(ce)

        row = {'clip_name':cn,'clip_label':cl,'clip_hour':hour,
               'n_segments':n_seg}

        # Spectral features (always included)
        for k in SPECTRAL_KEYS:
            if k in feat_df.columns:
                row[k] = feat_df.loc[mask,k].values[0]

        if approach == 'BASE' or approach == 'C':
            # Aggregated embeddings (mean, std, max)
            for i,v in enumerate(ce.mean(0)): row[f'agg_mean_{i}']=v
            for i,v in enumerate(ce.std(0)):  row[f'agg_std_{i}']=v
            for i,v in enumerate(ce.max(0)):  row[f'agg_max_{i}']=v

        if approach == 'A' or approach == 'C':
            # First K segments in order — pad with zeros if fewer
            for k in range(K_SEGS):
                seg = ce[k] if k < n_seg else np.zeros(ce.shape[1])
                for i,v in enumerate(seg): row[f'ord_s{k}_{i}']=v

        if approach == 'B' or approach == 'C':
            # Delta features: difference between consecutive segments
            if n_seg >= 2:
                deltas = np.diff(ce, axis=0)  # shape: (n_seg-1, 1536)
                # Mean and std of all deltas
                delta_mean = deltas.mean(0)
                delta_std  = deltas.std(0)
                # First delta (segment 0 → segment 1)
                first_delta = deltas[0]
                # Last delta (second-to-last → last segment)
                last_delta  = deltas[-1]
            else:
                delta_mean  = np.zeros(ce.shape[1])
                delta_std   = np.zeros(ce.shape[1])
                first_delta = np.zeros(ce.shape[1])
                last_delta  = np.zeros(ce.shape[1])

            for i,v in enumerate(delta_mean):  row[f'dlt_mean_{i}']=v
            for i,v in enumerate(delta_std):   row[f'dlt_std_{i}']=v
            for i,v in enumerate(first_delta): row[f'dlt_first_{i}']=v
            for i,v in enumerate(last_delta):  row[f'dlt_last_{i}']=v

        rows.append(row)
    return pd.DataFrame(rows)


def run_cv(clip_df, approach_name):
    """Corrected clip-level 5-fold CV."""
    meta = {'clip_name','clip_label','clip_hour','n_segments'}
    ec   = [c for c in clip_df.columns if c not in meta and
            not c in SPECTRAL_KEYS]
    cn   = clip_df['clip_name'].values
    cl   = clip_df['clip_label'].values

    if len(np.unique(cl)) < 2 or cl.sum() < 5:
        return None

    cv    = StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
    probs = np.zeros(len(clip_df),dtype=np.float32)

    for fold,(tr,va) in enumerate(cv.split(cn,cl)):
        tdf=clip_df.iloc[tr].copy().reset_index(drop=True)
        vdf=clip_df.iloc[va].copy().reset_index(drop=True)
        ytr=cl[tr].astype(int)

        # Z-scores from training baseline only
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

        # PCA on embedding columns only (high dimensional)
        emb_cols=[c for c in ec if any(c.startswith(p) for p in
                  ['agg_','ord_','dlt_'])]
        spec_cols=[c for c in ec if c not in emb_cols]

        if emb_cols:
            n_comp=min(80,len(emb_cols)//10,len(tdf)-1)
            pca=PCA(n_components=n_comp,random_state=SEED)
            etr=pca.fit_transform(tdf[emb_cols].values)
            eva=pca.transform(vdf[emb_cols].values)
        else:
            etr=np.zeros((len(tdf),1)); eva=np.zeros((len(vdf),1))

        spec_tr=tdf[spec_cols].values if spec_cols else np.zeros((len(tdf),1))
        spec_va=vdf[spec_cols].values if spec_cols else np.zeros((len(vdf),1))

        X_tr=np.hstack([tdf[SPECTRAL_KEYS].values,z_tr.values,
                         spec_tr,etr]).astype(np.float32)
        X_va=np.hstack([vdf[SPECTRAL_KEYS].values,z_va.values,
                         spec_va,eva]).astype(np.float32)

        sc=StandardScaler()
        X_tr_s=sc.fit_transform(X_tr); X_va_s=sc.transform(X_va)

        n_pos=ytr.sum(); n_neg=(ytr==0).sum()
        w=np.where(ytr==1,n_neg/max(n_pos,1),1.0)
        gb=GradientBoostingClassifier(n_estimators=100,max_depth=3,
           learning_rate=0.1,random_state=SEED,subsample=0.8)
        gb.fit(X_tr_s,ytr,sample_weight=w)
        probs[va]=gb.predict_proba(X_va_s)[:,1]

        p_f,r_f,_,_=precision_recall_fscore_support(
            cl[va].astype(int),(probs[va]>=0.5).astype(int),
            average='binary',zero_division=0)
        print(f"    Fold {fold+1}: P={p_f:.3f} R={r_f:.3f}")

    y=cl.astype(int)
    pr,rc,th=precision_recall_curve(y,probs)
    valid=np.where((pr[:-1]>=0.70)&(rc[:-1]>=0.70))[0]
    if len(valid)>0:
        bi=valid[np.argmax(pr[valid]+rc[valid])]; bt=float(th[bi])
        preds=(probs>=bt).astype(int)
        p,r,_,_=precision_recall_fscore_support(y,preds,average='binary',zero_division=0)
        beat=True
    else:
        bi=np.argmax(pr[:-1]+rc[:-1]); bt=float(th[bi])
        preds=(probs>=bt).astype(int)
        p,r,_,_=precision_recall_fscore_support(y,preds,average='binary',zero_division=0)
        beat=False

    n_feats=X_tr.shape[1]
    return {'p':p,'r':r,'beat':beat,'n_feats':n_feats}


if __name__ == '__main__':

    approaches = ['BASE','A','B','C']
    approach_labels = {
        'BASE': 'Aggregated (mean/std/max) — baseline',
        'A':    'Ordered concat (first 4 segs in order)',
        'B':    'Delta features (changes between segs)',
        'C':    'Combined (agg + ordered + deltas)',
    }

    OUT = os.path.join(BASE_DIR,"ordered_segments_results.txt")
    all_results = {}

    for name,ff,ef,lf,best_p,best_r in RECORDERS:
        print(f"\n{'='*60}")
        print(f"{name} — Ordered Segment Modeling")
        print(f"{'='*60}")
        print(f"  Baseline (seq voting): P={best_p:.3f}  R={best_r:.3f}")

        feat_path=os.path.join(BASE_DIR,ff)
        emb_path=os.path.join(BASE_DIR,ef)
        if not os.path.exists(feat_path) or not os.path.exists(emb_path):
            print(f"  Files not found — skipping"); continue

        feat_df=pd.read_csv(feat_path)
        emb=np.load(emb_path)
        labels=np.load(os.path.join(BASE_DIR,lf))
        n=min(len(feat_df),len(emb))
        feat_df=feat_df.iloc[:n].reset_index(drop=True); emb=emb[:n]

        # Check min segments
        seg_counts=feat_df.groupby('clip_name').size()
        print(f"  Segments per clip: min={seg_counts.min()} "
              f"mean={seg_counts.mean():.1f} max={seg_counts.max()}")

        rec_results = {}
        for approach in approaches:
            print(f"\n  Approach {approach}: {approach_labels[approach]}")
            clip_df=build_clip_df(feat_df,emb,approach)
            res=run_cv(clip_df,approach)
            if res:
                beat_str="BEAT" if res['beat'] else "miss"
                print(f"  → P={res['p']:.3f}  R={res['r']:.3f}  "
                      f"{beat_str}  ({res['n_feats']} features after PCA)")
                rec_results[approach]=res
            else:
                print(f"  → Insufficient data")

        all_results[name]=rec_results

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n\n"+"="*75)
    print("ORDERED SEGMENT MODELING SUMMARY")
    print("Does segment order improve over mean/std/max aggregation?")
    print("="*75)
    print(f"{'Rec':<6} {'BASE P':>8} {'BASE R':>8} "
          f"{'A (order) P':>12} {'B (delta) P':>12} {'C (combo) P':>12}")
    print("-"*75)

    for name,*_,best_p,best_r in RECORDERS:
        if name not in all_results: continue
        res=all_results[name]
        def gp(k): return res[k]['p'] if k in res else 0
        def gr(k): return res[k]['r'] if k in res else 0
        print(f"{name:<6} {gp('BASE'):>8.3f} {gr('BASE'):>8.3f} "
              f"{gp('A'):>12.3f} {gp('B'):>12.3f} {gp('C'):>12.3f}")

    print("="*75)
    print("\nKey question: does any ordered approach beat BASE?")
    for name,*_ in RECORDERS:
        if name not in all_results: continue
        res=all_results[name]
        if 'BASE' not in res: continue
        base_p=res['BASE']['p']
        for approach in ['A','B','C']:
            if approach in res and res[approach]['p']>base_p+0.02:
                print(f"  {name} Approach {approach}: "
                      f"P={res[approach]['p']:.3f} > BASE P={base_p:.3f} "
                      f"(+{res[approach]['p']-base_p:.3f})")

    with open(OUT,"w",encoding="utf-8") as fh:
        fh.write("Ordered Segment Modeling Results\n"+"="*50+"\n\n")
        fh.write(f"K_SEGS={K_SEGS} (segments used for ordering)\n\n")
        for name,*_,best_p,best_r in RECORDERS:
            if name not in all_results: continue
            fh.write(f"{name}:\n")
            for ap,res in all_results[name].items():
                fh.write(f"  {ap}: P={res['p']:.3f} R={res['r']:.3f} "
                         f"{'BEAT' if res['beat'] else 'miss'}\n")
            fh.write("\n")
    print(f"\nReport saved -> {OUT}")
    print("\nDone.")
