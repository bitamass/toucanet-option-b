"""perch_transfer.py -- is AM2's signal general (data-limited others) or is AM2 special?"""
import os, re
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score

SEED = 0
DIR = os.path.dirname(os.path.abspath(__file__))
EMB_CANDIDATES = ["{r}_time_controlled_emb.npy", "{r}_full_emb.npy", "{r}_embeddings.npy"]
LAB_CANDIDATES = ["{r}_time_controlled_labels.npy", "{r}_full_labels.npy", "{r}_labels.npy"]

def parse(name):
    m = re.search(r'(?:Audio[_ ]?Moth[_ ]?|AM)(\d+).*?(\d{8})_(\d{6})', str(name), re.I)
    if m: return f"AM{int(m.group(1))}", m.group(2)
    return "AM?","NA"

def load_recorder(rnum):
    csv=os.path.join(DIR,f"am{rnum}_feature_matrix.csv")
    if not os.path.exists(csv): return None
    meta=pd.read_csv(csv, usecols=["clip_name","clip_label"])
    rtag=f"am{rnum}"; emb=lab=None
    for ec,lc in zip(EMB_CANDIDATES,LAB_CANDIDATES):
        ep,lp=os.path.join(DIR,ec.format(r=rtag)),os.path.join(DIR,lc.format(r=rtag))
        if os.path.exists(ep) and os.path.exists(lp):
            e=np.load(ep); l=np.load(lp); e=e.reshape(e.shape[0],-1)
            if e.shape[0]==len(meta): emb,lab=e,l; break
    if emb is None or float((lab==meta["clip_label"].values).mean())<0.99: return None
    pr=meta["clip_name"].apply(parse)
    df=pd.DataFrame(emb, columns=[f"e{i}" for i in range(emb.shape[1])])
    df["recorder"]=[a for a,_ in pr]
    df["date"]=df["recorder"]+"|"+pd.Series([b for _,b in pr])
    df["label"]=meta["clip_label"].astype(int).values
    return df

def matched_pool(df, seed=0):
    rng=np.random.default_rng(seed); keep=[]
    for _,g in df.groupby("date"):
        pos=g.index[g.label==1].to_numpy(); neg=g.index[g.label==0].to_numpy()
        k=min(len(pos),len(neg))
        if k==0: continue
        keep+=list(rng.choice(pos,k,replace=False))+list(rng.choice(neg,k,replace=False))
    return df.loc[keep].copy()

def metrics(y,pred,proba):
    try: auc=roc_auc_score(y,proba)
    except Exception: auc=np.nan
    return precision_score(y,pred,zero_division=0), recall_score(y,pred,zero_division=0), auc

def sess_out(pool, feats, seed=0):
    y=pool["label"].values; rows=[]
    for s in np.unique(pool["date"].values):
        te=pool["date"].values==s; tr=~te
        if y[te].sum()<3 or (y[te]==0).sum()<3 or y[tr].sum()==0: continue
        m=HistGradientBoostingClassifier(random_state=seed).fit(pool[feats].values[tr],y[tr])
        pb=m.predict_proba(pool[feats].values[te])[:,1]; pd_=m.predict(pool[feats].values[te])
        rows.append(metrics(y[te],pd_,pb))
    return np.array(rows) if rows else None

def main():
    data={r:load_recorder(r) for r in [1,2,4,5,6]}
    data={r:d for r,d in data.items() if d is not None}
    feats=[c for c in next(iter(data.values())).columns if c.startswith("e") and c[1:].isdigit()]
    for r,d in data.items(): print(f"AM{r}: {len(d)} clips, {int(d.label.sum())} positive")
    print()
    print("="*60); print("A) TRANSFER: train on AM2, test honestly on each recorder"); print("="*60)
    train=matched_pool(data[2],SEED)
    model=HistGradientBoostingClassifier(random_state=SEED).fit(train[feats].values, train["label"].values)
    print(f"{'test recorder':<15}{'AUC':>8}{'P':>8}{'R':>8}")
    for r,d in data.items():
        if r==2: continue
        pool=matched_pool(d,SEED)
        if pool.label.sum()<3 or (pool.label==0).sum()<3: print(f"AM{r:<13} too few"); continue
        pb=model.predict_proba(pool[feats].values)[:,1]; pd_=model.predict(pool[feats].values)
        P,R,A=metrics(pool["label"].values,pd_,pb)
        print(f"AM{r:<13}{A:>8.3f}{P:>8.3f}{R:>8.3f}")
    ref=sess_out(matched_pool(data[2],SEED),feats,SEED)
    if ref is not None: print(f"\n(ref) AM2 self {np.nanmean(ref[:,2]):>8.3f}{ref[:,0].mean():>8.3f}{ref[:,1].mean():>8.3f}")
    print("\n"+"="*60); print("B) DATA-SIZE CHECK: AM2 session-out at reduced size"); print("="*60)
    full2=matched_pool(data[2],SEED); npos=int(full2.label.sum()); rng=np.random.default_rng(SEED)
    print(f"{'AM2 pos kept':<15}{'AUC':>8}{'P':>8}{'R':>8}")
    for target in [npos,1500,800,400]:
        if target>npos: continue
        idx=[]
        for _,g in full2.groupby("date"):
            pos=g.index[g.label==1].to_numpy(); neg=g.index[g.label==0].to_numpy()
            frac=min(1.0,target/npos); kp=int(round(len(pos)*frac)); kn=int(round(len(neg)*frac))
            if kp>0: idx+=list(rng.choice(pos,kp,replace=False))
            if kn>0: idx+=list(rng.choice(neg,kn,replace=False))
        sub=full2.loc[idx]; res=sess_out(sub,feats,SEED)
        if res is not None: print(f"{int(sub.label.sum()):<15}{np.nanmean(res[:,2]):>8.3f}{res[:,0].mean():>8.3f}{res[:,1].mean():>8.3f}")
    print("\nReading it:")
    print("  A) AM2 transfers (AUC>0.6) -> signal general, others lacked data.")
    print("     AM2 fails to transfer -> AM2 genuinely different.")
    print("  B) AUC falls toward 0.5 as size shrinks -> the win was data volume.")

if __name__=="__main__":
    main()