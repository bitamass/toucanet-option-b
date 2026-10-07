"""
54_fewshot_adaptation.py
-------------------------
Few-shot cross-site adaptation benchmark.

SAMMY'S KEY SUGGESTION:
  Stop asking whether zero-shot works (it doesn't).
  Ask: how many locally labelled examples are needed?

  The scientifically useful question is:
  "How many locally labelled disturbance clips from a new site
  are sufficient to reach deployment target P>=0.70 R>=0.70?"

  This produces a calibration curve:
  Performance vs number of local examples from new site.

DESIGN:
  For each held-out recorder (test site):
    1. Train BASE REPRESENTATION on other 4 recorders
    2. From test site: provide 0,1,2,5,10,20,50 labelled examples
       (sim + matched baseline) — called SUPPORT SET
    3. Test on REMAINING test site clips (held out from support set)
    4. Repeat 5 times with different random support set selections

  FOUR ADAPTATION METHODS compared at each calibration size:

  A. Logistic calibration
     Use support set to fit a logistic regression on top of
     the base model's probability scores. Cheapest adaptation.

  B. Prototype classifier
     Average Perch embeddings of support disturbance clips = disturbance prototype
     Average Perch embeddings of support baseline clips = baseline prototype
     Classify by cosine distance. No retraining. Elegant and robust.

  C. Fine-tune final layer
     Keep base model frozen. Fit a new classifier head on support set features.
     More flexible than logistic calibration.

  D. Local GB model
     Fit a full gradient boosting model on support set only.
     Most powerful but most likely to overfit with tiny support sets.

PRIMARY RESULT:
  Table and graph of precision vs number of support examples per method.
  Key finding: at what calibration size does the system first beat 0.70?
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.metrics import (precision_recall_fscore_support,
                              precision_recall_curve)

BASE_DIR = r"C:\Users\BitaMassoudi\KasmirWorldProjects\toucanet-option-b"
SEED     = 42
N_TRIALS = 5   # repeat with different random support sets

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

# Calibration sizes: number of DISTURBANCE clips from new site
CALIB_SIZES = [0, 1, 2, 5, 10, 20, 50]


def build_clip_df(feat_df, emb):
    rows=[]
    for cn in feat_df['clip_name'].unique():
        mask=feat_df['clip_name'].values==cn
        cl=feat_df['clip_label'].values[mask][0]
        hour=feat_df['clip_hour'].values[mask][0]
        ce=emb[mask]
        row={'clip_name':cn,'clip_label':cl,'clip_hour':hour}
        for k in SPECTRAL_KEYS:
            if k in feat_df.columns:
                row[k]=feat_df.loc[mask,k].values[0]
        for i,v in enumerate(ce.mean(0)): row[f'em_{i}']=v
        for i,v in enumerate(ce.std(0)):  row[f'es_{i}']=v
        for i,v in enumerate(ce.max(0)):  row[f'ex_{i}']=v
        rows.append(row)
    return pd.DataFrame(rows)


def get_features(train_df, test_df):
    """Standard features with z-scores and PCA."""
    Z_KEYS_=[f"z_{k}" for k in SPECTRAL_KEYS]
    ec=[c for c in train_df.columns if c.startswith(('em_','es_','ex_'))]

    # Z-scores from training baseline
    z_tr_rows,z_te_rows=[],[]
    for h in train_df['clip_hour'].unique():
        bm=(train_df['clip_hour']==h)&(train_df['clip_label']==0)
        br=train_df.loc[bm,SPECTRAL_KEYS]
        if len(br)==0: continue
        hm=br.mean(); hs=br.std().replace(0,1e-6)
        th=train_df['clip_hour']==h
        if th.sum()>0:
            z=(train_df.loc[th,SPECTRAL_KEYS]-hm)/hs
            z.columns=Z_KEYS_; z_tr_rows.append(z)
        te_h=test_df['clip_hour']==h
        if te_h.sum()>0:
            z=(test_df.loc[te_h,SPECTRAL_KEYS]-hm)/hs
            z.columns=Z_KEYS_; z_te_rows.append(z)
    seen=set(train_df['clip_hour'].unique())
    for h in set(test_df['clip_hour'].unique())-seen:
        te_h=test_df['clip_hour']==h
        if te_h.sum()>0:
            z_te_rows.append(pd.DataFrame(0,index=test_df.index[te_h],columns=Z_KEYS_))

    z_tr=pd.concat(z_tr_rows).sort_index() if z_tr_rows else pd.DataFrame(0,index=train_df.index,columns=Z_KEYS_)
    z_te=pd.concat(z_te_rows).sort_index() if z_te_rows else pd.DataFrame(0,index=test_df.index,columns=Z_KEYS_)

    pca=PCA(n_components=50,random_state=SEED)
    etr=pca.fit_transform(train_df[ec].values)
    ete=pca.transform(test_df[ec].values)

    X_tr=np.hstack([train_df[SPECTRAL_KEYS].values,z_tr.values,etr]).astype(np.float32)
    X_te=np.hstack([test_df[SPECTRAL_KEYS].values, z_te.values,ete]).astype(np.float32)

    # Also return raw Perch embeddings for prototype method
    emb_tr=train_df[ec].values.astype(np.float32)
    emb_te=test_df[ec].values.astype(np.float32)

    return X_tr,X_te,emb_tr,emb_te,pca,z_tr.columns.tolist()


def train_base_model(X_tr, y_tr):
    """Train GB on training sites."""
    sc=StandardScaler(); X_s=sc.fit_transform(X_tr)
    n_pos=y_tr.sum(); n_neg=(y_tr==0).sum()
    w=np.where(y_tr==1,n_neg/max(n_pos,1),1.0)
    gb=GradientBoostingClassifier(n_estimators=100,max_depth=3,
       learning_rate=0.1,random_state=SEED,subsample=0.8)
    gb.fit(X_s,y_tr,sample_weight=w)
    return gb,sc


def evaluate(y, probs):
    pr,rc,th=precision_recall_curve(y,probs)
    valid=np.where((pr[:-1]>=0.70)&(rc[:-1]>=0.70))[0]
    if len(valid)>0:
        bi=valid[np.argmax(pr[valid]+rc[valid])]; bt=float(th[bi])
    else:
        bi=np.argmax(pr[:-1]+rc[:-1]); bt=float(th[bi])
    preds=(probs>=bt).astype(int)
    p,r,_,_=precision_recall_fscore_support(y,preds,average='binary',zero_division=0)
    return p,r,(p>=0.70 and r>=0.70)


def method_A_logistic(base_probs_support, y_support,
                       base_probs_test, min_samples=2):
    """Logistic calibration on base model scores."""
    if len(y_support)<min_samples or len(np.unique(y_support))<2:
        return base_probs_test  # fallback to uncalibrated
    X_s=base_probs_support.reshape(-1,1)
    X_t=base_probs_test.reshape(-1,1)
    lr=LogisticRegression(random_state=SEED,max_iter=500)
    lr.fit(X_s,y_support)
    return lr.predict_proba(X_t)[:,1]


def method_B_prototype(emb_support, y_support, emb_test, min_samples=1):
    """
    Cosine distance to class prototypes.
    disturbance prototype = mean of support disturbance embeddings
    baseline prototype = mean of support baseline embeddings
    score = cos_sim(test, dist_proto) - cos_sim(test, base_proto)
    """
    pos_idx=np.where(y_support==1)[0]
    neg_idx=np.where(y_support==0)[0]
    if len(pos_idx)==0: return np.zeros(len(emb_test))

    dist_proto=emb_support[pos_idx].mean(axis=0,keepdims=True)
    if len(neg_idx)>0:
        base_proto=emb_support[neg_idx].mean(axis=0,keepdims=True)
    else:
        # No baseline support — use zero vector
        base_proto=np.zeros_like(dist_proto)

    # Normalise all
    dist_proto_n=normalize(dist_proto)
    base_proto_n=normalize(base_proto+1e-10)
    emb_test_n=normalize(emb_test)

    sim_dist=(emb_test_n @ dist_proto_n.T).flatten()
    sim_base=(emb_test_n @ base_proto_n.T).flatten()

    # Score = how much closer to disturbance than to baseline
    scores=sim_dist-sim_base
    # Shift to [0,1] range for threshold finding
    scores=(scores-scores.min())/(scores.max()-scores.min()+1e-10)
    return scores


def method_C_finetune(X_support_full, y_support, X_test_full, min_samples=2):
    """
    Fine-tune final layer: fit logistic regression on full features
    from support set only.
    """
    if len(y_support)<min_samples or len(np.unique(y_support))<2:
        return None
    sc=StandardScaler(); X_s=sc.fit_transform(X_support_full)
    X_t=sc.transform(X_test_full)
    lr=LogisticRegression(C=1.0,random_state=SEED,max_iter=1000,
                          class_weight='balanced')
    lr.fit(X_s,y_support)
    return lr.predict_proba(X_t)[:,1]


def method_D_local_gb(X_support_full, y_support, X_test_full, min_samples=5):
    """
    Local GB model trained only on support set.
    Most powerful but most prone to overfitting with tiny sets.
    """
    if len(y_support)<min_samples or len(np.unique(y_support))<2:
        return None
    sc=StandardScaler(); X_s=sc.fit_transform(X_support_full)
    X_t=sc.transform(X_test_full)
    n_pos=y_support.sum(); n_neg=(y_support==0).sum()
    w=np.where(y_support==1,n_neg/max(n_pos,1),1.0)
    gb=GradientBoostingClassifier(n_estimators=50,max_depth=2,
       learning_rate=0.1,random_state=SEED)
    gb.fit(X_s,y_support,sample_weight=w)
    return gb.predict_proba(X_t)[:,1]


if __name__ == '__main__':

    print("Few-Shot Cross-Site Adaptation Benchmark")
    print("="*65)
    print("How many local examples needed to reach P>=0.70 R>=0.70?")
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
        print(f"  {name}: {len(clip_df)} clips "
              f"({int(clip_df['clip_label'].sum())} sim)")

    available=list(all_clips.keys())
    methods=['A_logistic','B_prototype','C_finetune','D_local_gb']
    results={m:{n:{k:[] for k in CALIB_SIZES} for n in available}
             for m in methods}

    for test_name in available:
        train_names=[r for r in available if r!=test_name]
        test_df=all_clips[test_name].copy().reset_index(drop=True)
        train_df=pd.concat([all_clips[r] for r in train_names],
                            ignore_index=True)
        y_tr=train_df['clip_label'].values.astype(int)
        y_te=test_df['clip_label'].values.astype(int)
        if len(np.unique(y_te))<2: continue

        n_sim_test=int(y_te.sum())
        n_base_test=int((y_te==0).sum())

        print(f"\n{'='*60}")
        print(f"Test: {test_name}  ({n_sim_test} sim, {n_base_test} base)")

        # Get features
        X_tr,X_te,emb_tr,emb_te,pca,zkeys=get_features(train_df,test_df)

        # Train base model on training sites
        gb_base,sc_base=train_base_model(X_tr,y_tr)
        base_probs_te=gb_base.predict_proba(sc_base.transform(X_te))[:,1]
        base_p,base_r,base_beat=evaluate(y_te,base_probs_te)
        print(f"  Zero-shot baseline: P={base_p:.3f} R={base_r:.3f} "
              f"{'BEAT' if base_beat else 'miss'}")

        # Store zero-shot result
        for m in methods:
            results[m][test_name][0].append(base_p)

        # Few-shot adaptation
        rng=np.random.default_rng(SEED)
        sim_idx=np.where(y_te==1)[0]
        base_idx=np.where(y_te==0)[0]

        print(f"\n  {'N_support':>10} {'A_logis':>9} {'B_proto':>9} "
              f"{'C_ftune':>9} {'D_localGB':>10}")
        print(f"  {'-'*50}")

        for n_calib in CALIB_SIZES:
            if n_calib == 0: continue
            if n_calib > n_sim_test:
                print(f"  {n_calib:>10} skipped (only {n_sim_test} sim clips)")
                continue

            trial_results={m:[] for m in methods}
            for trial in range(N_TRIALS):
                # Sample support set
                sup_sim=rng.choice(sim_idx,min(n_calib,len(sim_idx)),
                                   replace=False)
                sup_base=rng.choice(base_idx,min(n_calib,len(base_idx)),
                                    replace=False)
                support_idx=np.concatenate([sup_sim,sup_base])
                test_idx=np.array([i for i in range(len(test_df))
                                   if i not in set(support_idx)])
                if len(test_idx)==0: continue
                y_sup=y_te[support_idx]
                y_tst=y_te[test_idx]
                if len(np.unique(y_tst))<2: continue

                X_sup=X_te[support_idx]; X_tst=X_te[test_idx]
                emb_sup=emb_te[support_idx]; emb_tst=emb_te[test_idx]
                bp_sup=base_probs_te[support_idx]
                bp_tst=base_probs_te[test_idx]

                # Method A
                probs_A=method_A_logistic(bp_sup,y_sup,bp_tst)
                p_A,r_A,_=evaluate(y_tst,probs_A)
                trial_results['A_logistic'].append(p_A)

                # Method B
                probs_B=method_B_prototype(emb_sup,y_sup,emb_tst)
                p_B,r_B,_=evaluate(y_tst,probs_B)
                trial_results['B_prototype'].append(p_B)

                # Method C
                probs_C=method_C_finetune(X_sup,y_sup,X_tst)
                if probs_C is not None:
                    p_C,r_C,_=evaluate(y_tst,probs_C)
                    trial_results['C_finetune'].append(p_C)

                # Method D
                probs_D=method_D_local_gb(X_sup,y_sup,X_tst)
                if probs_D is not None:
                    p_D,r_D,_=evaluate(y_tst,probs_D)
                    trial_results['D_local_gb'].append(p_D)

            # Average across trials
            avg={m:np.mean(v) if v else 0.0 for m,v in trial_results.items()}
            for m in methods:
                results[m][test_name][n_calib].append(avg[m])

            print(f"  {n_calib:>10} "
                  f"{avg['A_logistic']:>9.3f} {avg['B_prototype']:>9.3f} "
                  f"{avg.get('C_finetune',0):>9.3f} {avg.get('D_local_gb',0):>10.3f}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n\n"+"="*75)
    print("FEW-SHOT ADAPTATION SUMMARY")
    print("Average precision across all 5 recorders at each calibration size")
    print("="*75)
    print(f"{'N support':>10} {'A logistic':>11} {'B prototype':>12} "
          f"{'C finetune':>11} {'D local GB':>11}")
    print("-"*75)

    for n_calib in CALIB_SIZES:
        avgs={}
        for m in methods:
            vals=[]
            for name in available:
                v=results[m][name][n_calib]
                if v: vals.extend(v)
            avgs[m]=np.mean(vals) if vals else 0.0
        beat_markers={m:"★" if avgs[m]>=0.70 else "" for m in methods}
        print(f"{n_calib:>10} "
              f"{avgs['A_logistic']:>10.3f}{beat_markers['A_logistic']} "
              f"{avgs['B_prototype']:>11.3f}{beat_markers['B_prototype']} "
              f"{avgs.get('C_finetune',0):>10.3f}{beat_markers['C_finetune']} "
              f"{avgs.get('D_local_gb',0):>10.3f}{beat_markers['D_local_gb']}")

    print("="*75)
    print("★ = avg precision across all recorders >= 0.70")

    # Find minimum calibration size to beat 0.70
    print("\nMinimum support examples needed to reach avg P>=0.70:")
    for m in methods:
        for n_calib in CALIB_SIZES:
            vals=[]
            for name in available:
                v=results[m][name][n_calib]
                if v: vals.extend(v)
            avg=np.mean(vals) if vals else 0.0
            if avg>=0.70:
                print(f"  {m}: {n_calib} examples")
                break
        else:
            print(f"  {m}: never reaches 0.70 in tested range")

    # Save
    out=os.path.join(BASE_DIR,"fewshot_adaptation_results.txt")
    with open(out,"w",encoding="utf-8") as fh:
        fh.write("Few-Shot Adaptation Results\n"+"="*50+"\n\n")
        for m in methods:
            fh.write(f"\n{m}:\n")
            for n_calib in CALIB_SIZES:
                vals=[]
                for name in available:
                    v=results[m][name][n_calib]
                    if v: vals.extend(v)
                avg=np.mean(vals) if vals else 0.0
                fh.write(f"  {n_calib} examples: avg P={avg:.3f}\n")
    print(f"\nReport saved -> {out}")
    print("\nDone.")
