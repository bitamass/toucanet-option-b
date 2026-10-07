"""am2_leakcheck.py -- is AM2's signal real, or segment-level leakage?
Compares three hold-out grains on AM2 only:
  day       : leave-one-recorder-day-out (what we did)
  recording : leave-one-.wav-out (all ~11 segments of a clip stay together)
  bout      : leave-one-bout-out (>60-min gap boundaries)
If AUC stays high under 'recording'/'bout', segment leakage is ruled out.
"""
import os, re
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
SEED=0; DIR=os.path.dirname(os.path.abspath(__file__))

def parse(n):
    m=re.search(r'(?:Audio[_ ]?Moth[_ ]?|AM)(\d+).*?(\d{8})_(\d{6})',str(n),re.I)
    if m: return m.group(2), pd.to_datetime(m.group(2)+m.group(3),format="%Y%m%d%H%M%S")
    return "NA", pd.NaT

meta=pd.read_csv(os.path.join(DIR,"am2_feature_matrix.csv"), usecols=["clip_name","clip_label"])
emb=np.load(os.path.join(DIR,"am2_time_controlled_emb.npy")); emb=emb.reshape(emb.shape[0],-1)
lab=np.load(os.path.join(DIR,"am2_time_controlled_labels.npy"))
assert emb.shape[0]==len(meta) and (lab==meta.clip_label.values).mean()>0.99, "alignment problem"

pr=meta["clip_name"].apply(parse)
df=pd.DataFrame(emb, columns=[f"e{i}" for i in range(emb.shape[1])])
df["recording"]=meta["clip_name"].values
df["day"]=[a for a,_ in pr]
ts=pd.Series([b for _,b in pr])
df["label"]=meta["clip_label"].astype(int).values

# bouts: sort recordings in time, cut on >60min gaps
starts=pd.DataFrame({"rec":meta["clip_name"].values,"ts":ts}).drop_duplicates("rec").sort_values("ts")
gap=starts["ts"].diff().dt.total_seconds().div(60).fillna(0)
rec2bout=dict(zip(starts["rec"], (gap>60).cumsum()))
df["bout"]=meta["clip_name"].map(rec2bout).astype(str)

feats=[c for c in df.columns if c.startswith("e") and c[1:].isdigit()]

def matched(d, seed=0):
    rng=np.random.default_rng(seed); keep=[]
    for _,g in d.groupby("day"):
        pos=g.index[g.label==1].to_numpy(); neg=g.index[g.label==0].to_numpy()
        k=min(len(pos),len(neg))
        if k==0: continue
        keep+=list(rng.choice(pos,k,replace=False))+list(rng.choice(neg,k,replace=False))
    return d.loc[keep].copy()

def cv(d, group_col):
    y=d["label"].values; g=d[group_col].values; aucs=[]; leak=0
    for gv in np.unique(g):
        te=g==gv; tr=~te
        if y[te].sum()<3 or (y[te]==0).sum()<3 or y[tr].sum()==0: continue
        # leakage guard: no recording may appear on both sides
        if group_col!="recording":
            if set(d["recording"].values[te]) & set(d["recording"].values[tr]): leak+=1
        m=HistGradientBoostingClassifier(random_state=SEED).fit(d[feats].values[tr],y[tr])
        try: aucs.append(roc_auc_score(y[te], m.predict_proba(d[feats].values[te])[:,1]))
        except Exception: pass
    return (np.nanmean(aucs) if aucs else np.nan), len(aucs), leak

pool=matched(df, SEED)
print(f"AM2 matched pool: {len(pool)} clips, {int(pool.label.sum())} pos, "
      f"{pool.recording.nunique()} recordings, {pool.day.nunique()} days, {pool.bout.nunique()} bouts\n")
print(f"{'hold-out grain':<14}{'AUC':>8}{'folds':>7}{'recordings split?':>20}")
for col in ["day","bout","recording"]:
    a,n,leak=cv(pool,col)
    note = "OK" if (col=="recording" or leak==0) else f"{leak} folds LEAK"
    print(f"{col:<14}{a:>8.3f}{n:>7}{note:>20}")
print("\nIf AUC stays ~0.8 for 'recording' and 'bout' -> real signal, not segment leakage.")
print("If it drops toward 0.5 -> the day-level number was inflated by duplicate segments.")