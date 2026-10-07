"""perch_test.py -- honest matched-control session-out test on full 1536-dim Perch embeddings."""
import os, re, glob
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
    if m:
        return f"AM{int(m.group(1))}", m.group(2), pd.to_datetime(m.group(2)+m.group(3), format="%Y%m%d%H%M%S")
    return "AM?","NA", pd.NaT

def load_recorder(rnum):
    csv = os.path.join(DIR, f"am{rnum}_feature_matrix.csv")
    if not os.path.exists(csv): return None
    meta = pd.read_csv(csv, usecols=["clip_name","clip_label","clip_hour"])
    rtag = f"am{rnum}"; emb=lab=used=None
    for ec, lc in zip(EMB_CANDIDATES, LAB_CANDIDATES):
        ep, lp = os.path.join(DIR, ec.format(r=rtag)), os.path.join(DIR, lc.format(r=rtag))
        if os.path.exists(ep) and os.path.exists(lp):
            e=np.load(ep); l=np.load(lp); e=e.reshape(e.shape[0],-1)
            if e.shape[0]==len(meta): emb,lab,used=e,l,os.path.basename(ep); break
    if emb is None:
        print(f"  AM{rnum}: no embedding file matches {len(meta)} clips -> skipped"); return None
    align=float((lab==meta["clip_label"].values).mean())
    if align<0.99:
        print(f"  AM{rnum}: {used} labels only {align:.0%} aligned -> skipped"); return None
    pr=meta["clip_name"].apply(parse)
    df=pd.DataFrame(emb, columns=[f"e{i}" for i in range(emb.shape[1])])
    df["recorder"]=[a for a,_,_ in pr]
    df["date"]=df["recorder"]+"|"+pd.Series([b for _,b,_ in pr])
    df["label"]=meta["clip_label"].astype(int).values
    print(f"  AM{rnum}: {used}  ({emb.shape[0]}x{emb.shape[1]})  aligned {align:.0%}")
    return df

def matched_pool(df, seed=0):
    rng=np.random.default_rng(seed); keep=[]
    for _, g in df.groupby("date"):
        pos=g.index[g.label==1].to_numpy(); neg=g.index[g.label==0].to_numpy()
        k=min(len(pos),len(neg))
        if k==0: continue
        keep+=list(rng.choice(pos,k,replace=False))+list(rng.choice(neg,k,replace=False))
    return df.loc[keep].copy()

def session_out(df, feats, seed=0):
    X=df[feats].values; y=df["label"].values; grp=df["date"].values; rec=df["recorder"].values; rows=[]
    for s in np.unique(grp):
        te=grp==s; tr=~te
        if y[te].sum()<3 or (y[te]==0).sum()<3 or y[tr].sum()==0: continue
        m=HistGradientBoostingClassifier(random_state=seed).fit(X[tr],y[tr])
        pred=m.predict(X[te])
        try: auc=roc_auc_score(y[te], m.predict_proba(X[te])[:,1])
        except Exception: auc=np.nan
        rows.append(dict(session=s, recorder=rec[te][0],
            precision=precision_score(y[te],pred,zero_division=0),
            recall=recall_score(y[te],pred,zero_division=0), auc=auc))
    return pd.DataFrame(rows)

def main():
    print("Loading Perch embeddings and checking alignment:")
    parts=[load_recorder(r) for r in [1,2,4,5,6]]
    parts=[d for d in parts if d is not None]
    if not parts: print("\nNo aligned recorders."); return
    df=pd.concat(parts, ignore_index=True)
    feats=[c for c in df.columns if c.startswith("e") and c[1:].isdigit()]
    print(f"\ncombined: {len(df)} clips, {len(feats)}-dim Perch, {int(df.label.sum())} positive, {df.recorder.nunique()} recorders\n")
    matched=matched_pool(df, SEED)
    print(f"matched pool: {len(matched)} clips (pos={int(matched.label.sum())}, neg={int((matched.label==0).sum())})\n")
    print("="*60); print("HONEST SESSION-OUT ON FULL PERCH EMBEDDINGS"); print("="*60)
    res=session_out(matched, feats, SEED)
    if res.empty: print("no scorable folds"); return
    print(f"overall mean AUC: {res.auc.mean():.3f}   mean P: {res.precision.mean():.3f}   mean R: {res.recall.mean():.3f}\n")
    pr=res.groupby("recorder")[["precision","recall","auc"]].mean().round(3)
    pr["meets_0.70"]=(pr.precision>=0.70)&(pr.recall>=0.70)
    print(pr.to_string())
    print("\nCompare to 20-PC features (~0.5). AUC clearly >0.6 = Perch carries signal the PCs lost.")
    res.to_csv(os.path.join(DIR,"perch_sessionout.csv"), index=False)
    print(f"saved -> {os.path.join(DIR,'perch_sessionout.csv')}")

if __name__=="__main__":
    main()