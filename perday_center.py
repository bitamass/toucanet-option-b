"""perday_center.py -- does removing each day's background offset fix cross-day generalization?"""
import os, re, glob
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

SEED=0
DIR=os.path.dirname(os.path.abspath(__file__))
EMB=["{r}_time_controlled_emb.npy","{r}_full_emb.npy","{r}_embeddings.npy"]
LAB=["{r}_time_controlled_labels.npy","{r}_full_labels.npy","{r}_labels.npy"]
TRAINCAP=4000
TESTCAP=2000

def parse(n):
    m=re.search(r'(?:Audio[_ ]?Moth[_ ]?|AM)(\d+).*?(\d{8})',str(n),re.I)
    return (f"AM{int(m.group(1))}", m.group(2)) if m else ("AM?","NA")

def load(rnum):
    csv=os.path.join(DIR,f"am{rnum}_feature_matrix.csv")
    if not os.path.exists(csv): return None
    meta=pd.read_csv(csv, usecols=["clip_name","clip_label"])
    rtag=f"am{rnum}"; emb=lab=None
    for ec,lc in zip(EMB,LAB):
        ep,lp=os.path.join(DIR,ec.format(r=rtag)),os.path.join(DIR,lc.format(r=rtag))
        if os.path.exists(ep) and os.path.exists(lp):
            e=np.load(ep); l=np.load(lp); e=e.reshape(e.shape[0],-1)
            if e.shape[0]==len(meta): emb,lab=e,l; break
    if emb is None or float((lab==meta["clip_label"].values).mean())<0.99: return None
    pr=meta["clip_name"].apply(parse)
    df=pd.DataFrame(emb, columns=[f"e{i}" for i in range(emb.shape[1])])
    df["recorder"]=[a for a,_ in pr]
    df["session"]=pd.Series([a for a,_ in pr])+"|"+pd.Series([b for _,b in pr])
    df["label"]=meta["clip_label"].astype(int).values
    return df

def matched(df, seed=0):
    rng=np.random.default_rng(seed); keep=[]
    for _,g in df.groupby("session"):
        pos=g.index[g.label==1].to_numpy(); neg=g.index[g.label==0].to_numpy()
        k=min(len(pos),len(neg))
        if k==0: continue
        keep+=list(rng.choice(pos,k,replace=False))+list(rng.choice(neg,k,replace=False))
    return df.loc[keep].reset_index(drop=True)

def run(pool, feats, centered):
    X=pool[feats].values.astype(np.float32).copy()
    y=pool["label"].values; sess=pool["session"].values; rec=pool["recorder"].values
    if centered:
        for s in np.unique(sess):
            m=sess==s
            X[m]=X[m]-X[m].mean(axis=0, keepdims=True)
    rng=np.random.default_rng(SEED); rows=[]
    for s in np.unique(sess):
        te=np.where(sess==s)[0]; tr=np.where(sess!=s)[0]
        if y[te].sum()<3 or (y[te]==0).sum()<3 or y[tr].sum()==0: continue
        if len(tr)>TRAINCAP: tr=rng.choice(tr,TRAINCAP,replace=False)
        if len(te)>TESTCAP: te=rng.choice(te,TESTCAP,replace=False)
        mdl=HistGradientBoostingClassifier(random_state=SEED).fit(X[tr],y[tr])
        try: a=roc_auc_score(y[te], mdl.predict_proba(X[te])[:,1])
        except Exception: continue
        rows.append((rec[te[0]], a))
    return pd.DataFrame(rows, columns=["recorder","auc"])

def main():
    recs=[1,2,4,5,6]
    raw={r:load(r) for r in recs}; raw={r:d for r,d in raw.items() if d is not None}
    feats=[c for c in next(iter(raw.values())).columns if c.startswith("e") and c[1:].isdigit()]
    pool=pd.concat([matched(raw[r],SEED) for r in raw], ignore_index=True)
    print(f"pool {len(pool)} clips, {pool.session.nunique()} recorder-days, {len(feats)} dims\n")
    print("running RAW ..."); R=run(pool,feats,False)
    print("running CENTERED ..."); C=run(pool,feats,True)
    print("\n"+"="*46)
    print("LEAVE-ONE-DAY-OUT AUC:  raw  vs  per-day centered")
    print("="*46)
    print(f"{'recorder':<10}{'RAW':>8}{'CENTERED':>11}{'gain':>8}")
    for r in sorted(set(R.recorder)|set(C.recorder)):
        ra=R[R.recorder==r].auc.mean(); ca=C[C.recorder==r].auc.mean()
        print(f"{r:<10}{ra:>8.3f}{ca:>11.3f}{ca-ra:>+8.3f}")
    print("-"*37)
    print(f"{'OVERALL':<10}{R.auc.mean():>8.3f}{C.auc.mean():>11.3f}{C.auc.mean()-R.auc.mean():>+8.3f}")
    print("\nBig positive gain -> per-day background shift WAS the bottleneck;")
    print("centering exposes the disturbance and cross-day detection improves.")

if __name__=="__main__":
    main()