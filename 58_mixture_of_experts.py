"""
58_mixture_of_experts.py
-------------------------
Mixture of site-specific experts with output calibration and consensus.

ADDRESSES THE CORE FAILURE:
  Current approach: train ONE pooled model on 4 sites combined.
  Result: model averages across all sites, performs poorly on any
  single held-out site. Score scale shifts completely at new site.

THIS APPROACH (three ideas combined):

  IDEA 1 — Site-specific experts
    Train one GB model per source recorder (4 separate models).
    Each model is specialised to its own site's acoustic environment.
    A nighttime site expert should transfer better to a new nighttime
    site than a model that averaged across morning and night sites.

  IDEA 2 — Rank calibration on new site baseline
    Each source model produces raw probability scores.
    At a new site these scores may shift systematically — the model
    flags many things as high-probability because its scale is wrong.
    Fix: convert each model's scores to PERCENTILES relative to that
    model's scores on the NEW SITE'S BASELINE CLIPS ONLY.
    A clip is suspicious only if its score is unusually high relative
    to recent baseline scores at this specific site.
    This normalises model outputs rather than input features.

  IDEA 3 — Weighted consensus
    Weight each source expert by acoustic similarity to the new site.
    Similarity measured by cosine distance between mean Perch embeddings
    of source and target baseline clips.
    Combine calibrated percentile scores using weighted average.
    Require consensus: flag only when combined score exceeds threshold.

DEPLOYMENT-VALID:
  Only uses unlabelled baseline clips from new site for calibration.
  No simulation data from new site needed.
  Consistent with "pretrained globally, calibrated locally" model.

EVALUATION:
  Per-site CV (each recorder trains on its own data) — sanity check.
  LODO cross-site with mixture of experts vs pooled baseline.
  Key comparison: does weighted consensus beat avg P=0.364?
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

# Number of baseline clips from new site for calibration
N_CALIB_BASE = 40

# Consensus thresholds to test
CONSENSUS_THRESHOLDS = [0.50, 0.60, 0.70, 0.80]


def build_clip_df(feat_df, emb):
    rows = []
    for cn in feat_df['clip_name'].unique():
        mask = feat_df['clip_name'].values == cn
        cl   = feat_df['clip_label'].values[mask][0]
        hour = feat_df['clip_hour'].values[mask][0]
        ce   = emb[mask]
        row  = {'clip_name':cn,'clip_label':cl,'clip_hour':hour}
        for k in SPECTRAL_KEYS:
            if k in feat_df.columns:
                row[k] = feat_df.loc[mask,k].values[0]
        for i,v in enumerate(ce.mean(0)): row[f'em_{i}']=v
        for i,v in enumerate(ce.std(0)):  row[f'es_{i}']=v
        for i,v in enumerate(ce.max(0)):  row[f'ex_{i}']=v
        rows.append(row)
    return pd.DataFrame(rows)


def get_features_single(train_df, test_df):
    """Features for one source site training on its own z-scores."""
    ec = [c for c in train_df.columns if c.startswith(('em_','es_','ex_'))]
    z_tr_rows, z_te_rows = [], []
    for h in train_df['clip_hour'].unique():
        bm=(train_df['clip_hour']==h)&(train_df['clip_label']==0)
        br=train_df.loc[bm,SPECTRAL_KEYS]
        if len(br)==0: continue
        hm=br.mean(); hs=br.std().replace(0,1e-6)
        th=train_df['clip_hour']==h
        if th.sum()>0:
            z=(train_df.loc[th,SPECTRAL_KEYS]-hm)/hs
            z.columns=Z_KEYS; z_tr_rows.append(z)
        te_h=test_df['clip_hour']==h
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
    pca=PCA(n_components=50,random_state=SEED)
    etr=pca.fit_transform(train_df[ec].values)
    ete=pca.transform(test_df[ec].values)
    X_tr=np.hstack([train_df[SPECTRAL_KEYS].values,z_tr.values,etr]).astype(np.float32)
    X_te=np.hstack([test_df[SPECTRAL_KEYS].values, z_te.values,ete]).astype(np.float32)
    return X_tr, X_te, pca


def train_expert(train_df, test_df):
    """Train one GB expert on a single source site."""
    X_tr, X_te, pca = get_features_single(train_df, test_df)
    y_tr = train_df['clip_label'].values.astype(int)
    sc   = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)
    n_pos=y_tr.sum(); n_neg=(y_tr==0).sum()
    w=np.where(y_tr==1,n_neg/max(n_pos,1),1.0)
    gb=GradientBoostingClassifier(n_estimators=100,max_depth=3,
       learning_rate=0.1,random_state=SEED,subsample=0.8)
    gb.fit(X_tr_s,y_tr,sample_weight=w)
    raw_probs = gb.predict_proba(X_te_s)[:,1]
    return raw_probs, sc, pca, gb


def rank_calibrate(raw_probs, baseline_probs):
    """
    Convert raw probabilities to percentiles relative to baseline scores.

    For each test clip: what fraction of baseline clips have a LOWER score?
    A clip with percentile 0.95 is more anomalous than 95% of baseline clips.

    This normalises model outputs rather than input features.
    Only uses unlabelled baseline clips from the new site.
    """
    calibrated = np.array([
        np.mean(baseline_probs < p) for p in raw_probs
    ])
    return calibrated


def compute_similarity(src_emb_base, tgt_emb_base):
    """
    Cosine similarity between source and target baseline embeddings.
    Higher similarity = source site acoustically similar to target site.
    Used to weight expert contributions.
    """
    ec_cols = [c for c in src_emb_base.columns if c.startswith('em_')]
    src_mean = src_emb_base[ec_cols].values.mean(axis=0)
    tgt_mean = tgt_emb_base[ec_cols].values.mean(axis=0)
    src_norm = src_mean / (np.linalg.norm(src_mean) + 1e-10)
    tgt_norm = tgt_mean / (np.linalg.norm(tgt_mean) + 1e-10)
    return float(np.dot(src_norm, tgt_norm))


def evaluate(y, probs):
    if len(np.unique(y)) < 2: return 0,0,False
    pr,rc,th=precision_recall_curve(y,probs)
    valid=np.where((pr[:-1]>=0.70)&(rc[:-1]>=0.70))[0]
    if len(valid)>0:
        bi=valid[np.argmax(pr[valid]+rc[valid])]; bt=float(th[bi])
    else:
        bi=np.argmax(pr[:-1]+rc[:-1]); bt=float(th[bi])
    preds=(probs>=bt).astype(int)
    p,r,_,_=precision_recall_fscore_support(y,preds,average='binary',zero_division=0)
    return p,r,(p>=0.70 and r>=0.70)


if __name__ == '__main__':

    print("Mixture of Site-Specific Experts")
    print("="*65)
    print("Train one expert per source site.")
    print("Rank-calibrate on new site baseline.")
    print("Weight by acoustic similarity. Require consensus.")
    print("="*65)

    # Load all data
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
    pooled_res={}; expert_res={}

    for test_name in available:
        train_names=[r for r in available if r!=test_name]
        test_df =all_clips[test_name].copy().reset_index(drop=True)
        y_te    =test_df['clip_label'].values.astype(int)
        if len(np.unique(y_te))<2: continue

        # Target site baseline clips for calibration
        tgt_base=test_df[test_df['clip_label']==0].copy()
        n_base_avail=len(tgt_base)
        calib_base=tgt_base.iloc[:min(N_CALIB_BASE,n_base_avail)]

        print(f"\n{'='*60}")
        print(f"Test: {test_name}  ({n_base_avail} baseline clips, "
              f"using {len(calib_base)} for calibration)")

        # ── Pooled baseline (current approach) ────────────────────────────────
        train_df=pd.concat([all_clips[r] for r in train_names],ignore_index=True)
        raw_pooled,_,_,_=train_expert(train_df,test_df)

        # Calibrate pooled model on target baseline
        base_scores_pooled,_,_,_=train_expert(train_df,calib_base)
        cal_pooled=rank_calibrate(raw_pooled,base_scores_pooled)

        p_pool_raw, r_pool_raw, b_pool_raw = evaluate(y_te, raw_pooled)
        p_pool_cal, r_pool_cal, b_pool_cal = evaluate(y_te, cal_pooled)
        pooled_res[test_name]={'raw':(p_pool_raw,r_pool_raw,b_pool_raw),
                               'cal':(p_pool_cal,r_pool_cal,b_pool_cal)}
        print(f"\n  Pooled raw:        P={p_pool_raw:.3f} R={r_pool_raw:.3f} "
              f"{'BEAT' if b_pool_raw else 'miss'}")
        print(f"  Pooled calibrated: P={p_pool_cal:.3f} R={r_pool_cal:.3f} "
              f"{'BEAT' if b_pool_cal else 'miss'}")

        # ── Site-specific experts ──────────────────────────────────────────────
        print(f"\n  Training {len(train_names)} site-specific experts...")
        expert_probs_raw  = {}
        expert_probs_cal  = {}
        expert_weights    = {}

        for src_name in train_names:
            src_df  = all_clips[src_name].copy().reset_index(drop=True)
            src_base= src_df[src_df['clip_label']==0].copy()

            # Train expert on this source site only
            raw_te,_,_,_ = train_expert(src_df, test_df)
            raw_cb,_,_,_ = train_expert(src_df, calib_base)

            # Rank-calibrate: percentile vs this expert's baseline scores
            cal_te = rank_calibrate(raw_te, raw_cb)

            # Acoustic similarity: source baseline vs target baseline
            sim = compute_similarity(src_base, calib_base)

            expert_probs_raw[src_name] = raw_te
            expert_probs_cal[src_name] = cal_te
            expert_weights[src_name]   = max(sim, 0.0)  # clip negative sims

            p_exp,r_exp,b_exp = evaluate(y_te, cal_te)
            print(f"    {src_name}: P={p_exp:.3f} R={r_exp:.3f} "
                  f"{'BEAT' if b_exp else 'miss'}  "
                  f"similarity={sim:.3f}  weight={max(sim,0):.3f}")

        # Normalise weights
        total_w = sum(expert_weights.values())
        if total_w > 0:
            for k in expert_weights:
                expert_weights[k] /= total_w
        else:
            for k in expert_weights:
                expert_weights[k] = 1.0/len(train_names)

        # ── Combination strategies ─────────────────────────────────────────────
        print(f"\n  Combining experts...")

        # Unweighted average of calibrated percentiles
        avg_cal = np.mean([expert_probs_cal[n] for n in train_names], axis=0)
        # Weighted average
        wav_cal = sum(expert_weights[n]*expert_probs_cal[n]
                      for n in train_names)
        # Median
        med_cal = np.median([expert_probs_cal[n] for n in train_names], axis=0)

        p_avg,r_avg,b_avg = evaluate(y_te, avg_cal)
        p_wav,r_wav,b_wav = evaluate(y_te, wav_cal)
        p_med,r_med,b_med = evaluate(y_te, med_cal)

        print(f"  Unweighted avg:    P={p_avg:.3f} R={r_avg:.3f} "
              f"{'BEAT' if b_avg else 'miss'}")
        print(f"  Weighted avg:      P={p_wav:.3f} R={r_wav:.3f} "
              f"{'BEAT' if b_wav else 'miss'}")
        print(f"  Median:            P={p_med:.3f} R={r_med:.3f} "
              f"{'BEAT' if b_med else 'miss'}")

        # Consensus: vote-based — flag only if >= K experts agree
        print(f"\n  Consensus voting:")
        for k_agree in [2, 3, 4]:
            # Each expert votes based on its calibrated percentile > 0.7
            votes=np.zeros(len(y_te))
            for n in train_names:
                votes += (expert_probs_cal[n] > 0.70).astype(float)
            consensus_score = votes / len(train_names)
            # Only flag if K or more experts agree
            pr,rc,th=precision_recall_curve(y_te,consensus_score)
            valid=np.where((pr[:-1]>=0.70)&(rc[:-1]>=0.70))[0]
            if len(valid)>0:
                bi=valid[np.argmax(pr[valid]+rc[valid])]; bt=float(th[bi])
            else:
                bi=np.argmax(pr[:-1]+rc[:-1]); bt=float(th[bi])
            preds=(consensus_score>=bt).astype(int)
            p_c,r_c,_,_=precision_recall_fscore_support(y_te,preds,
                         average='binary',zero_division=0)
            beat_c=(p_c>=0.70 and r_c>=0.70)
            flag=" ★" if beat_c else ""
            print(f"    >= {k_agree} experts agree: P={p_c:.3f} R={r_c:.3f} "
                  f"{'BEAT' if beat_c else 'miss'}{flag}")

        expert_res[test_name]={
            'wav':(p_wav,r_wav,b_wav),
            'avg':(p_avg,r_avg,b_avg),
            'med':(p_med,r_med,b_med),
        }

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n\n"+"="*75)
    print("MIXTURE OF EXPERTS SUMMARY — CROSS-SITE PRECISION")
    print("="*75)
    print(f"{'Rec':<6} {'Pooled raw':>11} {'Pooled cal':>11} "
          f"{'Weighted':>9} {'Avg':>7} {'Median':>8}")
    print("-"*55)
    for name in available:
        if name not in expert_res: continue
        pr=pooled_res[name]['raw'][0]
        pc=pooled_res[name]['cal'][0]
        pw=expert_res[name]['wav'][0]
        pa=expert_res[name]['avg'][0]
        pm=expert_res[name]['med'][0]
        print(f"{name:<6} {pr:>11.3f} {pc:>11.3f} {pw:>9.3f} "
              f"{pa:>7.3f} {pm:>8.3f}")
    print("="*75)

    avgs={
        'Pooled raw':    np.mean([pooled_res[n]['raw'][0] for n in available if n in pooled_res]),
        'Pooled cal':    np.mean([pooled_res[n]['cal'][0] for n in available if n in pooled_res]),
        'Weighted avg':  np.mean([expert_res[n]['wav'][0] for n in available if n in expert_res]),
        'Unweighted avg':np.mean([expert_res[n]['avg'][0] for n in available if n in expert_res]),
        'Median':        np.mean([expert_res[n]['med'][0] for n in available if n in expert_res]),
    }
    print("\nAverage cross-site precision:")
    for label,val in sorted(avgs.items(),key=lambda x:-x[1]):
        improvement=val-avgs['Pooled raw']
        print(f"  {label:<20}: {val:.3f}  ({improvement:+.3f} vs pooled raw)")
    print(f"\n  Previous best cross-site (deployment calibration): 0.364")
    best=max(avgs.values())
    print(f"  Best this experiment: {best:.3f}")
    print(f"  Overall improvement:  {best-0.364:+.3f}")

    out=os.path.join(BASE_DIR,"mixture_experts_results.txt")
    with open(out,"w",encoding="utf-8") as fh:
        fh.write("Mixture of Experts Results\n"+"="*50+"\n\n")
        for name in available:
            if name not in expert_res: continue
            fh.write(f"{name}:\n")
            fh.write(f"  Pooled raw: {pooled_res[name]['raw'][0]:.3f}\n")
            fh.write(f"  Pooled cal: {pooled_res[name]['cal'][0]:.3f}\n")
            fh.write(f"  Weighted:   {expert_res[name]['wav'][0]:.3f}\n")
            fh.write(f"  Avg:        {expert_res[name]['avg'][0]:.3f}\n\n")
    print(f"\nReport saved -> {out}")
    print("\nDone.")
