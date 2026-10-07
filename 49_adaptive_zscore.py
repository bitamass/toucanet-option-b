"""
49_adaptive_zscore.py
----------------------
Adaptive z-scores using rolling baseline instead of fixed hourly averages.

THE PROBLEM WITH CURRENT Z-SCORES:
  Current approach computes z-scores by subtracting the mean of ALL
  baseline clips at the same hour across all 5 days of recording.
  This is site-specific but not time-adaptive — it treats 8 PM on
  March 17 the same as 8 PM on March 21 even if conditions differ.

THE IDEA (Sammy's background subtraction):
  Instead of:
    z = (clip - mean_of_all_baseline_clips_at_this_hour)
  Use:
    z = (clip - mean_of_baseline_clips_in_last_30_minutes)

  This is like video background subtraction — the background model
  updates continuously as conditions change. A vehicle at 3 PM
  sounds unusual relative to the LAST 30 MINUTES of 3 PM recordings
  at this specific site, not relative to the average 3 PM across
  all five recording days.

WHY THIS MIGHT GENERALISE BETTER CROSS-SITE:
  The rolling baseline captures what is normal RIGHT NOW at this site.
  Cross-site, you could use the new site's own baseline recordings
  to build the rolling baseline — no simulation data needed.
  This means you never need to transfer what "disturbance sounds like"
  — you only need to transfer "this sounds different from recent normal."

THREE WINDOW SIZES TESTED:
  W1: 30 minutes  (10 clips × 3 seconds = 30 sec... use clip count)
  W2: 60 minutes
  W3: 120 minutes

  In clip terms at 3-second clips:
  W1: last 20 baseline clips
  W2: last 40 baseline clips
  W3: last 80 baseline clips

EVALUATION:
  Per-site 5-fold CV for each window size.
  LODO cross-site for each window size.
  Compare to fixed z-score baseline (script 34).
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

WINDOWS = [
    {'name':'W_20clips',  'n_clips':20,  'label':'20 baseline clips (~1 min)'},
    {'name':'W_40clips',  'n_clips':40,  'label':'40 baseline clips (~2 min)'},
    {'name':'W_80clips',  'n_clips':80,  'label':'80 baseline clips (~4 min)'},
    {'name':'W_fixed',    'n_clips':None,'label':'Fixed hourly (current approach)'},
]


def parse_timestamp(clip_name):
    """Extract datetime from clip name for chronological ordering."""
    parts = clip_name.split('_')
    try:
        date = parts[3]  # YYYYMMDD
        time = parts[4].replace('.wav','')  # HHMMSS
        return pd.Timestamp(f"{date[:4]}-{date[4:6]}-{date[6:8]} "
                           f"{time[:2]}:{time[2:4]}:{time[4:6]}")
    except:
        return pd.Timestamp('2025-01-01')


def build_clip_df(feat_df, emb):
    """Build clip-level dataframe with timestamps for ordering."""
    rows = []
    for cn in feat_df['clip_name'].unique():
        mask = feat_df['clip_name'].values == cn
        cl   = feat_df['clip_label'].values[mask][0]
        hour = feat_df['clip_hour'].values[mask][0]
        ts   = parse_timestamp(cn)
        ce   = emb[mask]
        row  = {'clip_name':cn,'clip_label':cl,'clip_hour':hour,'timestamp':ts}
        for k in SPECTRAL_KEYS:
            if k in feat_df.columns:
                row[k] = feat_df.loc[mask,k].values[0]
        for i,v in enumerate(ce.mean(0)): row[f'em_{i}']=v
        for i,v in enumerate(ce.std(0)):  row[f'es_{i}']=v
        for i,v in enumerate(ce.max(0)):  row[f'ex_{i}']=v
        rows.append(row)
    df = pd.DataFrame(rows)
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def compute_adaptive_zscores(clip_df, n_clips):
    """
    Compute rolling z-scores for each clip using the last n_clips
    baseline clips as the reference distribution.

    For each clip at position i:
      - Find the last n_clips baseline clips before position i
      - Compute mean and std of those clips' spectral features
      - z = (current_clip - rolling_mean) / rolling_std

    If fewer than 5 baseline clips available in window, fall back
    to all available baseline clips before this point.
    """
    Z_KEYS = [f"z_{k}" for k in SPECTRAL_KEYS]
    z_rows = []

    spec_vals = clip_df[SPECTRAL_KEYS].values
    labels    = clip_df['clip_label'].values

    for i in range(len(clip_df)):
        # Find baseline clips before this position
        before_mask = (labels[:i] == 0)
        before_idx  = np.where(before_mask)[0]

        if len(before_idx) == 0:
            # No baseline clips yet — use zeros
            z = np.zeros(len(SPECTRAL_KEYS))
        else:
            # Use last n_clips baseline clips
            window_idx = before_idx[-n_clips:]
            window_specs = spec_vals[window_idx]
            roll_mean = window_specs.mean(axis=0)
            roll_std  = window_specs.std(axis=0)
            roll_std[roll_std < 1e-6] = 1e-6
            z = (spec_vals[i] - roll_mean) / roll_std

        z_rows.append(dict(zip(Z_KEYS, z)))

    return pd.DataFrame(z_rows, index=clip_df.index)


def compute_fixed_zscores(train_df, test_df=None):
    """Current approach — hourly z-scores from training baseline."""
    Z_KEYS = [f"z_{k}" for k in SPECTRAL_KEYS]
    z_tr_rows, z_te_rows = [], []

    for hour in train_df['clip_hour'].unique():
        bm = (train_df['clip_hour']==hour)&(train_df['clip_label']==0)
        br = train_df.loc[bm, SPECTRAL_KEYS]
        if len(br) == 0: continue
        hm = br.mean(); hs = br.std().replace(0, 1e-6)
        th = train_df['clip_hour'] == hour
        if th.sum() > 0:
            z = (train_df.loc[th,SPECTRAL_KEYS]-hm)/hs
            z.columns=Z_KEYS; z_tr_rows.append(z)
        if test_df is not None:
            te_h = test_df['clip_hour'] == hour
            if te_h.sum() > 0:
                z = (test_df.loc[te_h,SPECTRAL_KEYS]-hm)/hs
                z.columns=Z_KEYS; z_te_rows.append(z)

    z_tr = pd.concat(z_tr_rows).sort_index() if z_tr_rows else \
           pd.DataFrame(0,index=train_df.index,columns=Z_KEYS)

    if test_df is not None:
        seen = set(train_df['clip_hour'].unique())
        for h in set(test_df['clip_hour'].unique())-seen:
            te_h = test_df['clip_hour']==h
            if te_h.sum()>0:
                z_te_rows.append(pd.DataFrame(0,index=test_df.index[te_h],columns=Z_KEYS))
        z_te = pd.concat(z_te_rows).sort_index() if z_te_rows else \
               pd.DataFrame(0,index=test_df.index,columns=Z_KEYS)
        return z_tr, z_te
    return z_tr


def run_cv(clip_df, z_df, emb_cols):
    """5-fold clip-level CV."""
    Z_KEYS = [f"z_{k}" for k in SPECTRAL_KEYS]
    cn = clip_df['clip_name'].values
    cl = clip_df['clip_label'].values
    if len(np.unique(cl)) < 2 or cl.sum() < 5: return None

    cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    probs = np.zeros(len(clip_df), dtype=np.float32)

    for tr, va in cv.split(cn, cl):
        tdf = clip_df.iloc[tr].copy().reset_index(drop=True)
        vdf = clip_df.iloc[va].copy().reset_index(drop=True)
        ytr = cl[tr].astype(int)

        # Use pre-computed adaptive z-scores (sliced to fold)
        z_tr = z_df.iloc[tr].copy().reset_index(drop=True)
        z_va = z_df.iloc[va].copy().reset_index(drop=True)

        pca  = PCA(n_components=50, random_state=SEED)
        etr  = pca.fit_transform(tdf[emb_cols].values)
        eva  = pca.transform(vdf[emb_cols].values)

        X_tr = np.hstack([tdf[SPECTRAL_KEYS].values,
                           z_tr[Z_KEYS].values, etr]).astype(np.float32)
        X_va = np.hstack([vdf[SPECTRAL_KEYS].values,
                           z_va[Z_KEYS].values, eva]).astype(np.float32)

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr); X_va_s = sc.transform(X_va)

        n_pos=ytr.sum(); n_neg=(ytr==0).sum()
        w=np.where(ytr==1,n_neg/max(n_pos,1),1.0)
        gb=GradientBoostingClassifier(n_estimators=100,max_depth=3,
           learning_rate=0.1,random_state=SEED,subsample=0.8)
        gb.fit(X_tr_s,ytr,sample_weight=w)
        probs[va]=gb.predict_proba(X_va_s)[:,1]

    y  = cl.astype(int)
    pr,rc,th = precision_recall_curve(y,probs)
    valid = np.where((pr[:-1]>=0.70)&(rc[:-1]>=0.70))[0]
    if len(valid)>0:
        bi=valid[np.argmax(pr[valid]+rc[valid])]; bt=float(th[bi])
    else:
        bi=np.argmax(pr[:-1]+rc[:-1]); bt=float(th[bi])
    preds=(probs>=bt).astype(int)
    p,r,_,_=precision_recall_fscore_support(y,preds,average='binary',zero_division=0)
    return p,r,(p>=0.70 and r>=0.70)


def run_lodo(all_clips, test_name):
    """Cross-site LODO for each window configuration."""
    Z_KEYS = [f"z_{k}" for k in SPECTRAL_KEYS]
    train_names=[r for r in all_clips if r!=test_name]
    test_df =all_clips[test_name]['df'].copy().reset_index(drop=True)
    train_df=pd.concat([all_clips[r]['df'] for r in train_names],ignore_index=True)
    y_tr=train_df['clip_label'].values.astype(int)
    y_te=test_df['clip_label'].values.astype(int)
    if len(np.unique(y_te))<2: return {}

    emb_cols=[c for c in train_df.columns if c.startswith(('em_','es_','ex_'))]
    results={}

    for w in WINDOWS:
        wname=w['name']; n_clips=w['n_clips']

        if n_clips is None:
            z_tr,z_te=compute_fixed_zscores(train_df,test_df)
        else:
            # For LODO: compute rolling z using training sites' baseline only
            # Apply to test using training distribution
            # (Approximate: use fixed z from training since we don't have
            # test site baseline in training set)
            z_tr=compute_adaptive_zscores(train_df,n_clips)
            z_te=compute_adaptive_zscores(test_df,n_clips)

        pca=PCA(n_components=50,random_state=SEED)
        etr=pca.fit_transform(train_df[emb_cols].values)
        ete=pca.transform(test_df[emb_cols].values)

        X_tr=np.hstack([train_df[SPECTRAL_KEYS].values,
                         z_tr[Z_KEYS].values,etr]).astype(np.float32)
        X_te=np.hstack([test_df[SPECTRAL_KEYS].values,
                         z_te[Z_KEYS].values,ete]).astype(np.float32)

        sc=StandardScaler()
        X_tr_s=sc.fit_transform(X_tr); X_te_s=sc.transform(X_te)

        n_pos=y_tr.sum(); n_neg=(y_tr==0).sum()
        w_=np.where(y_tr==1,n_neg/max(n_pos,1),1.0)
        gb=GradientBoostingClassifier(n_estimators=100,max_depth=3,
           learning_rate=0.1,random_state=SEED,subsample=0.8)
        gb.fit(X_tr_s,y_tr,sample_weight=w_)
        probs=gb.predict_proba(X_te_s)[:,1]

        pr,rc,th=precision_recall_curve(y_te,probs)
        valid=np.where((pr[:-1]>=0.70)&(rc[:-1]>=0.70))[0]
        if len(valid)>0:
            bi=valid[np.argmax(pr[valid]+rc[valid])]; bt=float(th[bi])
        else:
            bi=np.argmax(pr[:-1]+rc[:-1]); bt=float(th[bi])
        preds=(probs>=bt).astype(int)
        p,r,_,_=precision_recall_fscore_support(y_te,preds,average='binary',zero_division=0)
        results[wname]=(p,r,(p>=0.70 and r>=0.70))

    return results


if __name__ == '__main__':

    Z_KEYS = [f"z_{k}" for k in SPECTRAL_KEYS]

    print("Adaptive Z-Score Experiment")
    print("="*60)
    print("Rolling baseline vs fixed hourly baseline")
    print("="*60)

    # Load all data
    all_clips={}
    for name,ff,ef,lf in RECORDERS:
        fp=os.path.join(BASE_DIR,ff); ep=os.path.join(BASE_DIR,ef)
        if not os.path.exists(fp) or not os.path.exists(ep): continue
        feat_df=pd.read_csv(fp); emb=np.load(ep)
        n=min(len(feat_df),len(emb))
        feat_df=feat_df.iloc[:n].reset_index(drop=True); emb=emb[:n]
        clip_df=build_clip_df(feat_df,emb)
        clip_df['recorder']=name
        all_clips[name]={'df':clip_df}
        print(f"  {name}: {len(clip_df)} clips ({int(clip_df['clip_label'].sum())} sim)")

    available=list(all_clips.keys())
    emb_cols=[c for c in all_clips[available[0]]['df'].columns
              if c.startswith(('em_','es_','ex_'))]

    # ── Per-site CV ───────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("PER-SITE CV — adaptive vs fixed z-scores")
    print("="*65)

    per_site={w['name']:{} for w in WINDOWS}

    for name in available:
        clip_df=all_clips[name]['df'].copy().reset_index(drop=True)
        print(f"\n{name}:")
        for w in WINDOWS:
            wname=w['name']; n_clips=w['n_clips']
            if n_clips is None:
                z_df=compute_fixed_zscores(clip_df)
                z_df=z_df.reset_index(drop=True)
                z_df.columns=Z_KEYS
            else:
                z_df=compute_adaptive_zscores(clip_df,n_clips)
                z_df=z_df.reset_index(drop=True)
                z_df.columns=Z_KEYS
            res=run_cv(clip_df,z_df,emb_cols)
            if res:
                p,r,beat=res
                per_site[wname][name]=(p,r,beat)
                print(f"  {w['label']:<35} P={p:.3f} R={r:.3f} {'BEAT' if beat else 'miss'}")

    # ── LODO cross-site ───────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("LODO CROSS-SITE — adaptive vs fixed z-scores")
    print("="*65)

    lodo={w['name']:{} for w in WINDOWS}
    for test_name in available:
        print(f"\nTest: {test_name}")
        results=run_lodo(all_clips,test_name)
        for wname,(p,r,beat) in results.items():
            lodo[wname][test_name]=(p,r,beat)
            label=next(w['label'] for w in WINDOWS if w['name']==wname)
            print(f"  {label:<35} P={p:.3f} R={r:.3f} {'BEAT' if beat else 'miss'}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n\n"+"="*70)
    print("SUMMARY — PER-SITE PRECISION")
    print("="*70)
    print(f"{'Window':<35} {'AM4':>6} {'AM2':>6} {'AM5':>6} {'AM6':>6} {'AM1':>6} {'Beat':>5}")
    print("-"*70)
    for w in WINDOWS:
        wn=w['name']; res=per_site[wn]
        vals=[res.get(n,(0,0,False))[0] for n in ['AM4','AM2','AM5','AM6','AM1']]
        nb=sum(1 for n in available if res.get(n,(0,0,False))[2])
        print(f"{w['label']:<35} "+" ".join(f"{v:>6.3f}" for v in vals)+f" {nb:>4}/5")

    print("\n"+"="*70)
    print("SUMMARY — LODO CROSS-SITE PRECISION")
    print("="*70)
    print(f"{'Window':<35} {'AM4':>6} {'AM2':>6} {'AM5':>6} {'AM6':>6} {'AM1':>6} {'Beat':>5}")
    print("-"*70)
    for w in WINDOWS:
        wn=w['name']; res=lodo[wn]
        vals=[res.get(n,(0,0,False))[0] for n in ['AM4','AM2','AM5','AM6','AM1']]
        nb=sum(1 for n in available if res.get(n,(0,0,False))[2])
        print(f"{w['label']:<35} "+" ".join(f"{v:>6.3f}" for v in vals)+f" {nb:>4}/5")

    print("\nKey question: does adaptive z-score improve cross-site precision?")
    fixed_avg=np.mean([lodo['W_fixed'].get(n,(0,0,False))[0] for n in available])
    best_adaptive=max('W_20clips','W_40clips','W_80clips',
                      key=lambda wn:np.mean([lodo[wn].get(n,(0,0,False))[0] for n in available]))
    best_avg=np.mean([lodo[best_adaptive].get(n,(0,0,False))[0] for n in available])
    print(f"  Fixed z-score avg cross-site P:    {fixed_avg:.3f}")
    print(f"  Best adaptive avg cross-site P:    {best_avg:.3f} ({best_adaptive})")
    print(f"  Improvement: {best_avg-fixed_avg:+.3f}")
    print("\nDone.")
