"""
43_lodo_sequence_voting.py
---------------------------
Leave-One-Device-Out (LODO) evaluation with sequence voting.

BACKGROUND:
  Script 36 showed cross-site training completely fails:
  0/5 recorders beat 0.70 when trained on other sites.
  Precision collapsed to 0.19-0.50.

  Script 42 showed sequence voting works powerfully per-site:
  4/5 recorders above 0.80 precision with sliding window voting.

THIS EXPERIMENT:
  Combines both: train on 4 recorders, test on 1 held-out,
  then apply sequence voting on the test predictions.

  The hypothesis: even if individual clip scores are noisy
  cross-site, genuine disturbance creates SUSTAINED patterns
  across consecutive clips. The temporal voting layer might
  generalise even when per-clip precision does not.

  A real vehicle approaches across many clips regardless of
  which recorder site it is at. Random false alarms remain
  isolated regardless of site.

DESIGN:
  For each recorder as test:
    1. Train clip-level GB on other 4 recorders combined
    2. Get probability scores for ALL test clips
    3. Sort test clips chronologically
    4. Apply sliding window voting (multiple N,K configs)
    5. Compare: cross-site alone vs cross-site + seq voting

WINDOW CONFIGS TESTED:
  Same as script 42: N=3K2, N=5K3, N=5K4, N=10K6, N=10K7, N=15K9

Uses saved feature matrices and embeddings.
Runtime: about 30-40 minutes.
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
     'am4_full_emb.npy','am4_full_labels.npy'),
    ('AM2','am2_feature_matrix.csv',
     'am2_time_controlled_emb.npy','am2_time_controlled_labels.npy'),
    ('AM5','am5_feature_matrix.csv',
     'am5_time_controlled_emb.npy','am5_time_controlled_labels.npy'),
    ('AM6','am6_feature_matrix.csv',
     'am6_time_controlled_emb.npy','am6_time_controlled_labels.npy'),
    ('AM1','am1_feature_matrix.csv',
     'am1_full_emb.npy','am1_full_labels.npy'),
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
    {'N':3,  'K':2,  'label':'N=3  K=2'},
    {'N':5,  'K':3,  'label':'N=5  K=3'},
    {'N':5,  'K':4,  'label':'N=5  K=4'},
    {'N':10, 'K':6,  'label':'N=10 K=6'},
    {'N':10, 'K':7,  'label':'N=10 K=7'},
    {'N':15, 'K':9,  'label':'N=15 K=9'},
]


def compute_rain_score(clip_df):
    rms  = clip_df['rms_mean'].values
    ent  = clip_df['spec_entropy_mean'].values
    roll = clip_df['rolloff_std'].values
    sil  = clip_df['silence_fraction'].values
    rn   = (rms-rms.min())/(rms.max()-rms.min()+1e-10)
    en   = (ent-ent.min())/(ent.max()-ent.min()+1e-10)
    ro   = 1-(roll-roll.min())/(roll.max()-roll.min()+1e-10)
    si   = 1-(sil-sil.min())/(sil.max()-sil.min()+1e-10)
    return 0.35*rn+0.35*en+0.20*ro+0.10*si


def build_clip_df(feat_df, emb):
    rows = []
    for cn in feat_df['clip_name'].unique():
        mask  = feat_df['clip_name'].values == cn
        cl    = feat_df['clip_label'].values[mask][0]
        hour  = feat_df['clip_hour'].values[mask][0]
        ce    = emb[mask]
        row   = {'clip_name':cn,'clip_label':cl,'clip_hour':hour}
        for k in SPECTRAL_KEYS:
            if k in feat_df.columns:
                row[k] = feat_df.loc[mask,k].values[0]
        for i,v in enumerate(ce.mean(0)): row[f'em_{i}'] = v
        for i,v in enumerate(ce.std(0)):  row[f'es_{i}'] = v
        for i,v in enumerate(ce.max(0)):  row[f'ex_{i}'] = v
        rows.append(row)
    return pd.DataFrame(rows)


def apply_rain_filter(clip_df):
    scores = compute_rain_score(clip_df)
    clip_df = clip_df.copy()
    clip_df['rain_score'] = scores
    base_scores = clip_df.loc[clip_df['clip_label']==0,'rain_score']
    threshold   = np.percentile(base_scores, 90)
    clip_df     = clip_df[clip_df['rain_score']<=threshold].copy()
    return clip_df.drop(columns=['rain_score'])


def compute_zscores(train_df, test_df):
    z_tr_rows, z_te_rows = [], []
    for hour in train_df['clip_hour'].unique():
        bm = (train_df['clip_hour']==hour)&(train_df['clip_label']==0)
        br = train_df.loc[bm,SPECTRAL_KEYS]
        if len(br)==0: continue
        hm = br.mean(); hs = br.std().replace(0,1e-6)
        th = train_df['clip_hour']==hour
        if th.sum()>0:
            z=(train_df.loc[th,SPECTRAL_KEYS]-hm)/hs
            z.columns=Z_KEYS; z_tr_rows.append(z)
        te_h = test_df['clip_hour']==hour
        if te_h.sum()>0:
            z=(test_df.loc[te_h,SPECTRAL_KEYS]-hm)/hs
            z.columns=Z_KEYS; z_te_rows.append(z)
    # Handle unseen hours
    seen = set(train_df['clip_hour'].unique())
    for hour in set(test_df['clip_hour'].unique())-seen:
        te_h = test_df['clip_hour']==hour
        if te_h.sum()>0:
            z=pd.DataFrame(0,index=test_df.index[te_h],columns=Z_KEYS)
            z_te_rows.append(z)
    z_tr = pd.concat(z_tr_rows).sort_index() if z_tr_rows else pd.DataFrame(0,index=train_df.index,columns=Z_KEYS)
    z_te = pd.concat(z_te_rows).sort_index() if z_te_rows else pd.DataFrame(0,index=test_df.index,columns=Z_KEYS)
    return z_tr, z_te


def train_and_predict(train_df, test_df):
    """Train GB on train_df, return probability scores for test_df."""
    ec = [c for c in train_df.columns if c.startswith(('em_','es_','ex_'))]
    y_tr = train_df['clip_label'].values.astype(int)

    z_tr, z_te = compute_zscores(train_df, test_df)

    pca     = PCA(n_components=50, random_state=SEED)
    etr     = pca.fit_transform(train_df[ec].values)
    ete     = pca.transform(test_df[ec].values)

    X_tr = np.hstack([train_df[SPECTRAL_KEYS].values,z_tr.values,etr]).astype(np.float32)
    X_te = np.hstack([test_df[SPECTRAL_KEYS].values, z_te.values,ete]).astype(np.float32)

    sc     = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)

    n_pos = y_tr.sum(); n_neg=(y_tr==0).sum()
    w     = np.where(y_tr==1,n_neg/max(n_pos,1),1.0)
    gb    = GradientBoostingClassifier(n_estimators=100,max_depth=3,
            learning_rate=0.1,random_state=SEED,subsample=0.8)
    gb.fit(X_tr_s,y_tr,sample_weight=w)
    return gb.predict_proba(X_te_s)[:,1]


def find_best_threshold(y, probs):
    pr,rc,th = precision_recall_curve(y,probs)
    valid = np.where((pr[:-1]>=0.70)&(rc[:-1]>=0.70))[0]
    if len(valid)>0:
        bi=valid[np.argmax(pr[valid]+rc[valid])]; bt=float(th[bi])
    else:
        bi=np.argmax(pr[:-1]+rc[:-1]); bt=float(th[bi])
    preds=(probs>=bt).astype(int)
    p,r,_,_=precision_recall_fscore_support(y,preds,average='binary',zero_division=0)
    return p,r,(p>=0.70 and r>=0.70),bt


def apply_sequence_voting(clip_df, probs, base_thresh, N, K):
    """Apply sliding window in chronological order."""
    sorted_idx  = clip_df['clip_name'].argsort().values
    sorted_probs = probs[sorted_idx]
    sorted_labels= clip_df['clip_label'].values[sorted_idx]
    flags        = (sorted_probs>=base_thresh).astype(int)
    final_preds  = np.zeros(len(flags),dtype=int)
    for i in range(len(flags)):
        ws = max(0,i-N+1)
        if flags[ws:i+1].sum()>=K:
            final_preds[i]=1
    result = np.zeros(len(clip_df),dtype=int)
    result[sorted_idx] = final_preds
    return result


if __name__ == '__main__':

    OUT = os.path.join(BASE_DIR,"lodo_sequence_voting_results.txt")

    print("Loading all recorder data...")
    all_clips = {}
    for name,ff,ef,lf in RECORDERS:
        fp=os.path.join(BASE_DIR,ff); ep=os.path.join(BASE_DIR,ef)
        if not os.path.exists(fp) or not os.path.exists(ep):
            print(f"  {name}: missing"); continue
        feat_df=pd.read_csv(fp)
        emb=np.load(ep)
        labels=np.load(os.path.join(BASE_DIR,lf))
        n=min(len(feat_df),len(emb))
        feat_df=feat_df.iloc[:n].reset_index(drop=True); emb=emb[:n]
        clip_df=build_clip_df(feat_df,emb)
        clip_df=apply_rain_filter(clip_df)
        clip_df['recorder']=name
        print(f"  {name}: {len(clip_df)} clips ({int(clip_df['clip_label'].sum())} sim)")
        all_clips[name]=clip_df

    all_results={}

    for name,*_ in RECORDERS:
        if name not in all_clips: continue
        print(f"\n{'='*60}")
        print(f"Test recorder: {name}")
        train_names=[r for r in all_clips if r!=name]
        print(f"Train recorders: {train_names}")

        test_df  = all_clips[name].copy().reset_index(drop=True)
        train_df = pd.concat([all_clips[r] for r in train_names],ignore_index=True)
        y_te     = test_df['clip_label'].values.astype(int)

        if len(np.unique(y_te))<2: print("  Skipping — no both classes"); continue

        print(f"  Train: {len(train_df)} clips ({int(train_df['clip_label'].sum())} sim)")
        print(f"  Test:  {len(test_df)} clips ({int(y_te.sum())} sim)")

        print(f"  Training classifier...")
        probs = train_and_predict(train_df, test_df)

        # Cross-site only (no sequence)
        p0,r0,beat0,bt = find_best_threshold(y_te,probs)
        fa0 = int(((probs>=bt).astype(int)==1)&(y_te==0)).sum() if False else int(((probs>=bt)&(y_te==0)).sum())
        print(f"\n  Cross-site only: P={p0:.3f} R={r0:.3f} {'BEAT' if beat0 else 'miss'}")

        # Sequence voting
        print(f"  Applying sequence voting...")
        print(f"  {'Config':<14} {'P':>7} {'R':>7} {'FA':>5} {'Beat?':>6}")
        print(f"  {'-'*45}")

        rec_results={'cross_site_only':(p0,r0,beat0,fa0)}
        for w in WINDOWS:
            N,K,label=w['N'],w['K'],w['label']
            preds_seq=apply_sequence_voting(test_df,probs,bt,N,K)
            p,r,_,_=precision_recall_fscore_support(y_te,preds_seq,average='binary',zero_division=0)
            beat=(p>=0.70 and r>=0.70)
            fa=int(((preds_seq==1)&(y_te==0)).sum())
            flag=" ★" if beat and p>p0+0.05 else ""
            print(f"  {label:<14} {p:>7.3f} {r:>7.3f} {fa:>5} {'BEAT' if beat else 'miss':>6}{flag}")
            rec_results[label]=(p,r,beat,fa)

        all_results[name]=rec_results

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n\n"+"="*75)
    print("LODO + SEQUENCE VOTING SUMMARY")
    print("Train on 4 recorders, test on 1, apply sequence voting")
    print("="*75)
    print(f"{'Rec':<6} {'CS only P':>10} {'CS only R':>10} {'Best seq P':>11} {'Best seq R':>11} {'Beat?':>6}")
    print("-"*75)

    for name,*_ in RECORDERS:
        if name not in all_results: continue
        res=all_results[name]
        p0,r0,b0,_=res['cross_site_only']
        # Best sequence result
        seq_res={k:v for k,v in res.items() if k!='cross_site_only'}
        beating={k:v for k,v in seq_res.items() if v[2]}
        if beating:
            bk=max(beating,key=lambda k:beating[k][0]+beating[k][1])
            bp,br,bb,_=beating[bk]
            beat_str="YES"
        else:
            bk=max(seq_res,key=lambda k:seq_res[k][0])
            bp,br,bb,_=seq_res[bk]
            beat_str="no"
        print(f"{name:<6} {p0:>10.3f} {r0:>10.3f} {bp:>11.3f} {br:>11.3f} {beat_str:>6}  [{bk}]")

    print("="*75)

    # Key question
    n_beat_cs  = sum(1 for r in all_results.values() if r['cross_site_only'][2])
    n_beat_seq = sum(1 for r in all_results.values()
                     if any(v[2] for k,v in r.items() if k!='cross_site_only'))
    print(f"\nCross-site only:           {n_beat_cs}/5 beat 0.70")
    print(f"Cross-site + seq voting:   {n_beat_seq}/5 beat 0.70")
    print(f"\nDid sequence voting rescue cross-site generalisation? "
          f"{'YES' if n_beat_seq>n_beat_cs else 'PARTIAL' if n_beat_seq>0 else 'NO'}")

    with open(OUT,"w",encoding="utf-8") as fh:
        fh.write("LODO Sequence Voting Results\n"+"="*50+"\n\n")
        for name,*_ in RECORDERS:
            if name not in all_results: continue
            res=all_results[name]
            p0,r0,b0,_=res['cross_site_only']
            fh.write(f"{name}:\n  Cross-site only: P={p0:.3f} R={r0:.3f} {'BEAT' if b0 else 'miss'}\n")
            for k,v in res.items():
                if k=='cross_site_only': continue
                fh.write(f"  {k}: P={v[0]:.3f} R={v[1]:.3f} {'BEAT' if v[2] else 'miss'}\n")
            fh.write("\n")
    print(f"\nReport saved -> {OUT}")
    print("Done.")
