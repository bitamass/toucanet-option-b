"""teach_weak.py -- can the strong recorders teach the weak ones? Leave-one-recorder-out transfer."""
import os, re, glob
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

SEED=0
DIR=os.path.dirname(os.path.abspath(__file__))
EMB=["{r}_time_controlled_emb.npy","{r}_full_emb.npy","{r}_embeddings.npy"]
LAB=["{r}_time_controlled_labels.npy","{r}_full_labels.npy","{r}_labels.npy"]
STRONG=[2,6]
MAXPOOL=2500
TRAINCAP=5000

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
    df["recording"]=meta["clip_name"].values
    df["day"]=[b for _,b in pr]
    df["label"]=meta["clip_label"].astype(int).values
    return df

def matched(df, seed=0, cap=None):
    rng=np.random.default_rng(seed); keep=[]
    for _,g in df.groupby("day"):
        pos=g.index[g.label==1].to_numpy(); neg=g.index[g.label==0].to_numpy()
        k=min(len(pos),len(neg))
        if k==0: continue
        keep+=list(rng.choice(pos,k,replace=False))+list(rng.choice(neg,k,replace=False))
    out=df.loc[keep].reset_index(drop=True)
    if cap and len(out)>cap:
        out=out.iloc[rng.choice(len(out),cap,replace=False)].reset_index(drop=True)
    return out

def cap_rows(X,y,cap,seed=0):
    if len(y)<=cap: return X,y
    rng=np.random.default_rng(seed); idx=rng.choice(len(y),cap,replace=False)
    return X[idx],y[idx]

def self_auc(pool, feats):
    X=pool[feats].values; y=pool["label"].values; g=pool["recording"].values
    if pool["recording"].nunique()<3 or y.sum()<10: return np.nan
    aucs=[]
    for tr,te in GroupKFold(3).split(X,y,g):
        if y[te].sum()<3 or (y[te]==0).sum()<3: continue
        m=HistGradientBoostingClassifier(random_state=SEED).fit(X[tr],y[tr])
        aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:,1]))
    return np.mean(aucs) if aucs else np.nan

def main():
    recs=[1,2,4,5,6]
    raw={r:load(r) for r in recs}; raw={r:d for r,d in raw.items() if d is not None}
    feats=[c for c in next(iter(raw.values())).columns if c.startswith("e") and c[1:].isdigit()]
    pools={r:matched(raw[r], SEED, MAXPOOL) for r in raw}
    for r in pools: print(f"AM{r}: pool {len(pools[r])} clips, {int(pools[r].label.sum())} pos")
    print(f"\n{'target':<8}{'self':>8}{'from-all':>10}{'from-strong':>13}")
    for T in pools:
        s=self_auc(pools[T], feats)
        others=[pools[r] for r in pools if r!=T]
        Xo=np.vstack([p[feats].values for p in others]); yo=np.concatenate([p["label"].values for p in others])
        Xo,yo=cap_rows(Xo,yo,TRAINCAP,SEED)
        m=HistGradientBoostingClassifier(random_state=SEED).fit(Xo,yo)
        a_all=roc_auc_score(pools[T]["label"].values, m.predict_proba(pools[T][feats].values)[:,1])
        strong=[pools[r] for r in pools if r in STRONG and r!=T]
        if strong:
            Xs=np.vstack([p[feats].values for p in strong]); ys=np.concatenate([p["label"].values for p in strong])
            Xs,ys=cap_rows(Xs,ys,TRAINCAP,SEED)
            ms=HistGradientBoostingClassifier(random_state=SEED).fit(Xs,ys)
            a_str=roc_auc_score(pools[T]["label"].values, ms.predict_proba(pools[T][feats].values)[:,1])
        else: a_str=np.nan
        ss=f"{s:.3f}" if s==s else "  --"; st=f"{a_str:.3f}" if a_str==a_str else "  --"
        print(f"AM{T:<6}{ss:>8}{a_all:>10.3f}{st:>13}")
    print("\nself = honest within-recorder ceiling (GroupKFold by recording)")
    print("from-all/strong = trained on OTHER recorders, tested on this one (unseen)")
    print("from-* nears self on a weak recorder -> the group teaches it.")
    print("from-* ~0.5 -> the signal does not transfer to that recorder.")

if __name__=="__main__":
    main()