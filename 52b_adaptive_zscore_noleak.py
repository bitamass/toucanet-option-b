"""
52b_adaptive_zscore_noleak.py
------------------------------
Corrected adaptive z-scores — no label leakage.

THE PROBLEM WITH SCRIPT 52:
  Script 52 computed rolling z-scores using "last 20 BASELINE clips"
  meaning clips where clip_label == 0.
  In deployment you do NOT have the label before running the classifier.
  Using true labels to select baseline clips is label-assisted
  preprocessing — a form of data leakage.
  The extraordinary results (AM4=0.972, AM2=0.965) may be inflated.

THE FIX:
  Use the last 20 UNLABELLED clips regardless of their true label.
  This is what deployment actually looks like — the system sees clips
  in chronological order and normalises each one against the last 20
  clips it has already processed, without knowing which were baseline.

  Three variants tested:
  V1: Last 20 clips (unlabelled, deployment-valid)
  V2: Last 40 clips (unlabelled, deployment-valid)
  V3: Initial calibration period (first 60 clips assumed baseline-only)
      This simulates "record baseline before monitoring starts"
  V4: Self-updating — reject clips above threshold from baseline model
      (approximates a slowly-updating robust baseline)

COMPARED TO:
  - Script 52 result (label-assisted, upper bound)
  - Script 42 result (fixed hourly z-scores + seq voting, current best honest)

The difference between V1 and the script 52 result reveals
how much the label leakage inflated the numbers.
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
     'am4_full_emb.npy','am4_full_labels.npy', 0.973, 0.884),
    ('AM2','am2_feature_matrix.csv',
     'am2_time_controlled_emb.npy','am2_time_controlled_labels.npy', 0.965, 0.952),
    ('AM5','am5_feature_matrix.csv',
     'am5_time_controlled_emb.npy','am5_time_controlled_labels.npy', 0.884, 0.728),
    ('AM6','am6_feature_matrix.csv',
     'am6_time_controlled_emb.npy','am6_time_controlled_labels.npy', 0.886, 0.875),
    ('AM1','am1_feature_matrix.csv',
     'am1_full_emb.npy','am1_full_labels.npy', 0.951, 0.961),
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
    {'N':3, 'K':2}, {'N':5, 'K':3}, {'N':5, 'K':4},
    {'N':10,'K':6}, {'N':10,'K':7}, {'N':15,'K':9},
]


def parse_timestamp(clip_name):
    parts = clip_name.split('_')
    try:
        date=parts[3]; time=parts[4].replace('.wav','')
        return pd.Timestamp(f"{date[:4]}-{date[4:6]}-{date[6:8]} "
                           f"{time[:2]}:{time[2:4]}:{time[4:6]}")
    except: return pd.Timestamp('2025-01-01')


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


def compute_zscores_unlabelled(clip_df, n_clips):
    """
    DEPLOYMENT-VALID rolling z-scores.
    Uses last n_clips UNLABELLED clips as reference.
    No knowledge of which clips are baseline or simulation.
    """
    spec_vals = clip_df[SPECTRAL_KEYS].values
    z_rows = []
    for i in range(len(clip_df)):
        # Use last n_clips clips regardless of label
        window_start = max(0, i - n_clips)
        window = spec_vals[window_start:i]
        if len(window) < 2:
            z = np.zeros(len(SPECTRAL_KEYS))
        else:
            roll_mean = window.mean(axis=0)
            roll_std  = window.std(axis=0)
            roll_std[roll_std < 1e-6] = 1e-6
            z = (spec_vals[i] - roll_mean) / roll_std
        z_rows.append(dict(zip(Z_KEYS, z)))
    return pd.DataFrame(z_rows, index=clip_df.index)


def compute_zscores_initial_calib(clip_df, n_calib=60):
    """
    V3: Use first n_calib clips as calibration period.
    Assumes monitoring starts with a known-clean baseline period.
    After calibration, use fixed mean/std from those first clips.
    """
    spec_vals = clip_df[SPECTRAL_KEYS].values
    calib_mean = spec_vals[:n_calib].mean(axis=0)
    calib_std  = spec_vals[:n_calib].std(axis=0)
    calib_std[calib_std < 1e-6] = 1e-6
    z_all = (spec_vals - calib_mean) / calib_std
    z_rows = [dict(zip(Z_KEYS, z_all[i])) for i in range(len(clip_df))]
    return pd.DataFrame(z_rows, index=clip_df.index)


def compute_zscores_self_update(clip_df, n_clips=20, threshold=2.5):
    """
    V4: Self-updating baseline.
    Only add clip to rolling baseline if its current z-score
    is below threshold (low anomaly score).
    Approximates a slowly-updating robust baseline that rejects
    suspected disturbances automatically.
    """
    spec_vals = clip_df[SPECTRAL_KEYS].values
    baseline_buffer = []
    z_rows = []
    for i in range(len(clip_df)):
        if len(baseline_buffer) < 2:
            z = np.zeros(len(SPECTRAL_KEYS))
            baseline_buffer.append(spec_vals[i])
        else:
            window = np.array(baseline_buffer[-n_clips:])
            roll_mean = window.mean(axis=0)
            roll_std  = window.std(axis=0)
            roll_std[roll_std < 1e-6] = 1e-6
            z = (spec_vals[i] - roll_mean) / roll_std
            # Only update baseline if clip looks normal
            if np.abs(z).mean() < threshold:
                baseline_buffer.append(spec_vals[i])
        z_rows.append(dict(zip(Z_KEYS, z)))
    return pd.DataFrame(z_rows, index=clip_df.index)


def compute_zscores_labelled(clip_df, n_clips):
    """
    LEAKY version (script 52 approach) — for comparison only.
    Uses last n_clips BASELINE clips (requires true labels).
    """
    spec_vals = clip_df[SPECTRAL_KEYS].values
    labels    = clip_df['clip_label'].values
    z_rows    = []
    for i in range(len(clip_df)):
        before_base = np.where(labels[:i] == 0)[0]
        if len(before_base) < 2:
            z = np.zeros(len(SPECTRAL_KEYS))
        else:
            window_idx = before_base[-n_clips:]
            window = spec_vals[window_idx]
            roll_mean = window.mean(axis=0)
            roll_std  = window.std(axis=0)
            roll_std[roll_std < 1e-6] = 1e-6
            z = (spec_vals[i] - roll_mean) / roll_std
        z_rows.append(dict(zip(Z_KEYS, z)))
    return pd.DataFrame(z_rows, index=clip_df.index)


def run_cv(clip_df, z_df):
    """5-fold CV with pre-computed z-scores."""
    ec  = [c for c in clip_df.columns if c.startswith(('em_','es_','ex_'))]
    cn  = clip_df['clip_name'].values
    cl  = clip_df['clip_label'].values
    if len(np.unique(cl))<2 or cl.sum()<5: return None
    cv    = StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
    probs = np.zeros(len(clip_df),dtype=np.float32)
    for tr,va in cv.split(cn,cl):
        tdf=clip_df.iloc[tr].copy().reset_index(drop=True)
        vdf=clip_df.iloc[va].copy().reset_index(drop=True)
        ytr=cl[tr].astype(int)
        z_tr=z_df.iloc[tr].copy().reset_index(drop=True)
        z_va=z_df.iloc[va].copy().reset_index(drop=True)
        pca=PCA(n_components=50,random_state=SEED)
        etr=pca.fit_transform(tdf[ec].values)
        eva=pca.transform(vdf[ec].values)
        X_tr=np.hstack([tdf[SPECTRAL_KEYS].values,
                         z_tr[Z_KEYS].values,etr]).astype(np.float32)
        X_va=np.hstack([vdf[SPECTRAL_KEYS].values,
                         z_va[Z_KEYS].values,eva]).astype(np.float32)
        sc=StandardScaler()
        X_tr_s=sc.fit_transform(X_tr); X_va_s=sc.transform(X_va)
        n_pos=ytr.sum(); n_neg=(ytr==0).sum()
        w=np.where(ytr==1,n_neg/max(n_pos,1),1.0)
        gb=GradientBoostingClassifier(n_estimators=100,max_depth=3,
           learning_rate=0.1,random_state=SEED,subsample=0.8)
        gb.fit(X_tr_s,ytr,sample_weight=w)
        probs[va]=gb.predict_proba(X_va_s)[:,1]
    return probs, cl.astype(int)


def best_seq_voting(clip_df, probs, y):
    """Find best threshold then apply all sequence voting configs."""
    pr,rc,th=precision_recall_curve(y,probs)
    valid=np.where((pr[:-1]>=0.70)&(rc[:-1]>=0.70))[0]
    if len(valid)>0:
        bi=valid[np.argmax(pr[valid]+rc[valid])]; bt=float(th[bi])
    else:
        bi=np.argmax(pr[:-1]+rc[:-1]); bt=float(th[bi])
    # Single clip
    preds=(probs>=bt).astype(int)
    p,r,_,_=precision_recall_fscore_support(y,preds,average='binary',zero_division=0)
    results={'single':(p,r,(p>=0.70 and r>=0.70))}
    # Sequence voting
    for w in SEQ_WINDOWS:
        flags=(probs>=bt).astype(int)
        fp=np.zeros(len(flags),dtype=int)
        for i in range(len(flags)):
            ws=max(0,i-w['N']+1)
            if flags[ws:i+1].sum()>=w['K']: fp[i]=1
        p,r,_,_=precision_recall_fscore_support(y,fp,average='binary',zero_division=0)
        key=f"N={w['N']}K={w['K']}"
        results[key]=(p,r,(p>=0.70 and r>=0.70))
    # Best beating result
    beating={k:v for k,v in results.items() if v[2]}
    if beating:
        bk=max(beating,key=lambda k:beating[k][0]+beating[k][1])
        return results[bk][0],results[bk][1],True,bk
    bk=max(results,key=lambda k:results[k][0])
    return results[bk][0],results[bk][1],False,bk


if __name__ == '__main__':

    VARIANTS = {
        'V1_unlabelled_20':   ('Unlabelled rolling 20 clips',    'deployment-valid'),
        'V2_unlabelled_40':   ('Unlabelled rolling 40 clips',    'deployment-valid'),
        'V3_initial_calib':   ('Initial 60-clip calibration',    'deployment-valid'),
        'V4_self_update':     ('Self-updating baseline',          'deployment-valid'),
        'V_leaky_20':         ('Label-assisted 20 clips [LEAKY]','NOT deployment-valid'),
    }

    print("Adaptive Z-Score — Label Leakage Check")
    print("="*65)
    print("Comparing deployment-valid vs label-assisted versions")
    print("="*65)

    all_results={v:{} for v in VARIANTS}

    for name,ff,ef,lf,prev_leaky,prev_fixed in RECORDERS:
        print(f"\n{'─'*55}")
        print(f"{name}  prev(leaky)={prev_leaky:.3f}  prev(fixed seq)={prev_fixed:.3f}")

        feat_df=pd.read_csv(os.path.join(BASE_DIR,ff))
        emb=np.load(os.path.join(BASE_DIR,ef))
        n=min(len(feat_df),len(emb))
        feat_df=feat_df.iloc[:n].reset_index(drop=True); emb=emb[:n]
        clip_df=build_clip_df(feat_df,emb)

        print(f"  {len(clip_df)} clips chronologically ordered")

        # Compute z-scores for each variant
        zdfs = {
            'V1_unlabelled_20': compute_zscores_unlabelled(clip_df, 20),
            'V2_unlabelled_40': compute_zscores_unlabelled(clip_df, 40),
            'V3_initial_calib': compute_zscores_initial_calib(clip_df, 60),
            'V4_self_update':   compute_zscores_self_update(clip_df, 20, 2.5),
            'V_leaky_20':       compute_zscores_labelled(clip_df, 20),
        }

        print(f"  {'Variant':<35} {'P':>7} {'R':>7} {'Beat?':>6} {'Config'}")
        print(f"  {'-'*60}")

        for vname,(vlabel,validity) in VARIANTS.items():
            result=run_cv(clip_df, zdfs[vname])
            if result is None:
                print(f"  {vlabel:<35} insufficient data")
                continue
            probs,y=result
            p,r,beat,cfg=best_seq_voting(clip_df,probs,y)
            all_results[vname][name]=(p,r,beat)
            leaky_flag=" [LEAKY]" if "leaky" in vname.lower() else ""
            print(f"  {vlabel:<35} {p:>7.3f} {r:>7.3f} "
                  f"{'BEAT' if beat else 'miss':>6} {cfg}{leaky_flag}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n\n"+"="*75)
    print("LABEL LEAKAGE CHECK SUMMARY")
    print("="*75)
    print(f"{'Variant':<35} {'AM4':>7} {'AM2':>7} {'AM5':>7} "
          f"{'AM6':>7} {'AM1':>7} {'Beat':>5} {'Valid?'}")
    print("-"*75)

    for vname,(vlabel,validity) in VARIANTS.items():
        res=all_results[vname]
        vals=[res.get(n,(0,0,False))[0] for n in
              ['AM4','AM2','AM5','AM6','AM1']]
        nb=sum(1 for n in ['AM4','AM2','AM5','AM6','AM1']
               if res.get(n,(0,0,False))[2])
        valid_flag="YES" if "deployment-valid" in validity else "NO (leaky)"
        print(f"{vlabel:<35} "+" ".join(f"{v:>7.3f}" for v in vals)+
              f" {nb:>4}/5  {valid_flag}")

    print("="*75)
    print("\nKey comparison:")
    leaky_avg=np.mean([all_results['V_leaky_20'].get(n,(0,0,False))[0]
                       for n in ['AM4','AM2','AM5','AM6','AM1']])
    v1_avg=np.mean([all_results['V1_unlabelled_20'].get(n,(0,0,False))[0]
                    for n in ['AM4','AM2','AM5','AM6','AM1']])
    print(f"  Label-assisted (leaky):    avg P={leaky_avg:.3f}")
    print(f"  Unlabelled (deployment):   avg P={v1_avg:.3f}")
    print(f"  Inflation from leakage:    {leaky_avg-v1_avg:+.3f}")
    print("\nDone.")
