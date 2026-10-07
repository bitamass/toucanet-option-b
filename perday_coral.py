"""perday_coral.py -- does CORAL covariance alignment fix cross-day generalization?"""
import os, re, glob
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

SEED=0
DIR=os.path.dirname(os.path.abspath(__file__))
EMB=["{r}_time_controlled_emb.npy","{r}_full_emb.npy","{r}_embeddings.npy"]
LAB=["{r}_time_controlled_labels.npy","{r}_full_labels.npy","{r}_labels.npy"]
NPCA=100; TRAINCAP=4000; TESTCAP=2000; EPS=1e-3

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

def sym_sqrt(C, inv=False):
    w,V=np.linalg.eigh(C); w=np.clip(w,EPS,None)
    d=1.0/np.sqrt(w) if inv else np.sqrt(w)
    return (V*d)@V.T

def coral_map(Xs, Xt):
    mus=Xs.mean(0,keepdims=True); mut=Xt.mean(0,keepdims=True)
    Sc=Xs-mus; Tc=Xt-mut
    Cs=np.cov(Sc,rowvar=False)+EPS*np.eye(Sc.shape[1])
    Ct=np.cov(Tc,rowvar=False)+EPS*np.eye(Tc.shape[1])
    A=sym_sqrt(Cs,inv=True)@sym_sqrt(Ct)
    return Sc@A+mut

def main():
    recs=[1,2,4,5,6]
    raw={r:load(r) for r in recs}; raw={r:d for r,d in raw.items() if d is not None}
    feats=[c for c in next(iter(raw.values())).columns if c.startswith("e") and c[1:].isdigit()]
    pool=pd.concat([matched(raw[r],SEED) for r in raw], ignore_index=True)
    print(f"pool {len(pool)} clips, {pool.session.nunique()} recorder-days, PCA->{NPCA} dims\n")
    y=pool["label"].values; sess=pool["session"].values; rec=pool["recorder"].values
    Xfull=pool[feats].values.astype(np.float32)
    rng=np.random.default_rng(SEED); rows=[]
    print("running leave-one-day-out (no-adapt vs CORAL) ...")
    for s in np.unique(sess):
        te=np.where(sess==s)[0]; tr=np.where(sess!=s)[0]
        if y[te].sum()<3 or (y[te]==0).sum()<3 or y[tr].sum()==0: continue
        if len(tr)>TRAINCAP: tr=rng.choice(tr,TRAINCAP,replace=False)
        if len(te)>TESTCAP: te=rng.choice(te,TESTCAP,replace=False)
        nc=min(NPCA,len(tr)-1,Xfull.shape[1]); pca=PCA(n_components=nc, random_state=SEED).fit(Xfull[tr])
        Ztr=pca.transform(Xfull[tr]); Zte=pca.transform(Xfull[te])
        m1=HistGradientBoostingClassifier(random_state=SEED).fit(Ztr,y[tr])
        a1=roc_auc_score(y[te], m1.predict_proba(Zte)[:,1])
        Ztr_c=coral_map(Ztr,Zte)
        m2=HistGradientBoostingClassifier(random_state=SEED).fit(Ztr_c,y[tr])
        a2=roc_auc_score(y[te], m2.predict_proba(Zte)[:,1])
        rows.append((rec[te[0]],a1,a2))
    R=pd.DataFrame(rows,columns=["recorder","noadapt","coral"])
    print("\n"+"="*48)
    print("LEAVE-ONE-DAY-OUT AUC:  no-adapt (PCA)  vs  CORAL")
    print("="*48)
    print(f"{'recorder':<10}{'NO-ADAPT':>10}{'CORAL':>9}{'gain':>8}")
    for r in sorted(set(R.recorder)):
        na=R[R.recorder==r].noadapt.mean(); ca=R[R.recorder==r].coral.mean()
        print(f"{r:<10}{na:>10.3f}{ca:>9.3f}{ca-na:>+8.3f}")
    print("-"*37)
    print(f"{'OVERALL':<10}{R.noadapt.mean():>10.3f}{R.coral.mean():>9.3f}{R.coral.mean()-R.noadapt.mean():>+8.3f}")
    print("\nBig CORAL gain -> covariance alignment recovers cross-day detection.")
    print("Little gain -> cross-day shift is deeper than second-order statistics.")

if __name__=="__main__":
    main()