"""why_am2.py -- composition + all-pairs transfer to explain AM2."""
import os, re
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

SEED=0
DIR=os.path.dirname(os.path.abspath(__file__))
EMB=["{r}_time_controlled_emb.npy","{r}_full_emb.npy","{r}_embeddings.npy"]
LAB=["{r}_time_controlled_labels.npy","{r}_full_labels.npy","{r}_labels.npy"]

def parse(n):
    m=re.search(r'(?:Audio[_ ]?Moth[_ ]?|AM)(\d+).*?(\d{8})_(\d{6})',str(n),re.I)
    return (f"AM{int(m.group(1))}",m.group(2)) if m else ("AM?","NA")

def load(rnum):
    csv=os.path.join(DIR,f"am{rnum}_feature_matrix.csv")
    if not os.path.exists(csv): return None,None
    meta=pd.read_csv(csv, usecols=["clip_name","clip_label","sim_type"])
    rtag=f"am{rnum}"; emb=lab=None
    for ec,lc in zip(EMB,LAB):
        ep,lp=os.path.join(DIR,ec.format(r=rtag)),os.path.join(DIR,lc.format(r=rtag))
        if os.path.exists(ep) and os.path.exists(lp):
            e=np.load(ep); l=np.load(lp); e=e.reshape(e.shape[0],-1)
            if e.shape[0]==len(meta): emb,lab=e,l; break
    if emb is None or float((lab==meta["clip_label"].values).mean())<0.99: return None,None
    pr=meta["clip_name"].apply(parse)
    df=pd.DataFrame(emb, columns=[f"e{i}" for i in range(emb.shape[1])])
    df["date"]=pd.Series([a for a,_ in pr])+"|"+pd.Series([b for _,b in pr])
    df["label"]=meta["clip_label"].astype(int).values
    return df, meta

def matched(df, seed=0):
    rng=np.random.default_rng(seed); keep=[]
    for _,g in df.groupby("date"):
        pos=g.index[g.label==1].to_numpy(); neg=g.index[g.label==0].to_numpy()
        k=min(len(pos),len(neg))
        if k==0: continue
        keep+=list(rng.choice(pos,k,replace=False))+list(rng.choice(neg,k,replace=False))
    return df.loc[keep].copy()

def main():
    recs=[1,2,4,5,6]; D={}; M={}
    for r in recs:
        d,m=load(r)
        if d is not None: D[r],M[r]=d,m
    feats=[c for c in next(iter(D.values())).columns if c.startswith("e") and c[1:].isdigit()]
    print("="*64); print("1) DISTURBANCE COMPOSITION BY RECORDER"); print("="*64)
    print(f"{'recorder':<10}{'baseline':>10}{'Vehicle':>10}{'Human':>8}{'%pos':>8}")
    for r in D:
        st=M[r]["sim_type"].astype(str)
        veh=st.str.contains("Vehicle").sum(); hum=st.str.contains("Human").sum()
        base=(M[r]["clip_label"]==0).sum(); pct=100*M[r]["clip_label"].mean()
        print(f"AM{r:<8}{base:>10}{veh:>10}{hum:>8}{pct:>7.1f}%")
    print("\n"+"="*64); print("2) ALL-PAIRS TRANSFER (train=row, test=col, honest AUC)"); print("="*64)
    pools={r:matched(D[r],SEED) for r in D}
    models={r:HistGradientBoostingClassifier(random_state=SEED).fit(pools[r][feats].values,pools[r]["label"].values) for r in D}
    cols=list(D.keys())
    print("train\\test  " + "".join(f"AM{c:<6}" for c in cols))
    for rtr in cols:
        line=f"AM{rtr:<8}"
        for rte in cols:
            pool=pools[rte]; y=pool["label"].values
            if rtr==rte:
                aucs=[]
                for s in np.unique(pool["date"].values):
                    te=pool["date"].values==s; tr=~te
                    if y[te].sum()<3 or (y[te]==0).sum()<3 or y[tr].sum()==0: continue
                    mm=HistGradientBoostingClassifier(random_state=SEED).fit(pool[feats].values[tr],y[tr])
                    try: aucs.append(roc_auc_score(y[te],mm.predict_proba(pool[feats].values[te])[:,1]))
                    except Exception: pass
                a=np.nanmean(aucs) if aucs else np.nan
            else:
                try: a=roc_auc_score(y, models[rtr].predict_proba(pool[feats].values)[:,1])
                except Exception: a=np.nan
            line+=f"{a:>6.2f}" if a==a else f"{'--':>6}"
        print(line)
    print("\nDiagonal = honest self-performance. Off-diagonal = transfer.")
    print("Look for blocks of recorders that transfer to each other.")

if __name__=="__main__":
    main()