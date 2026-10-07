"""am2_grouped_cv.py -- is AM2's signal real, or segment leakage? Grouped recording-level CV."""
import os, re
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score

SEED=0; K=5
DIR=os.path.dirname(os.path.abspath(__file__))

def parse_day(n):
    m=re.search(r'(\d{8})_\d{6}', str(n)); return m.group(1) if m else "NA"

meta=pd.read_csv(os.path.join(DIR,"am2_feature_matrix.csv"), usecols=["clip_name","clip_label"])
emb=np.load(os.path.join(DIR,"am2_time_controlled_emb.npy")); emb=emb.reshape(emb.shape[0],-1)
lab=np.load(os.path.join(DIR,"am2_time_controlled_labels.npy"))
assert emb.shape[0]==len(meta) and (lab==meta.clip_label.values).mean()>0.99, "alignment problem"

df=pd.DataFrame(emb, columns=[f"e{i}" for i in range(emb.shape[1])])
df["recording"]=meta["clip_name"].values
df["day"]=meta["clip_name"].apply(parse_day).values
df["label"]=meta["clip_label"].astype(int).values
feats=[c for c in df.columns if c.startswith("e") and c[1:].isdigit()]

rng=np.random.default_rng(SEED); keep=[]
for _,g in df.groupby("day"):
    pos=g.index[g.label==1].to_numpy(); neg=g.index[g.label==0].to_numpy()
    k=min(len(pos),len(neg))
    if k==0: continue
    keep+=list(rng.choice(pos,k,replace=False))+list(rng.choice(neg,k,replace=False))
pool=df.loc[keep].reset_index(drop=True)
print(f"AM2 matched pool: {len(pool)} clips, {int(pool.label.sum())} pos, {pool.recording.nunique()} recordings, {pool.day.nunique()} days\n")

X=pool[feats].values; y=pool["label"].values; groups=pool["recording"].values

def run(splitter, use_groups):
    aucs=[]; leaks=0
    it=splitter.split(X,y,groups) if use_groups else splitter.split(X,y)
    for tr,te in it:
        if use_groups and (set(groups[tr]) & set(groups[te])): leaks+=1
        if y[te].sum()<3 or (y[te]==0).sum()<3: continue
        m=HistGradientBoostingClassifier(random_state=SEED).fit(X[tr],y[tr])
        aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:,1]))
    return np.mean(aucs), np.std(aucs), len(aucs), leaks

print("="*58); print(f"AM2  |  {K}-fold CV on full Perch embeddings"); print("="*58)
na,ns,nf,_=run(StratifiedKFold(K, shuffle=True, random_state=SEED), False)
ga,gs,gf,gl=run(GroupKFold(K), True)
print(f"{'evaluation':<28}{'AUC':>8}{'sd':>7}{'folds':>7}")
print(f"{'NAIVE (segments shuffled)':<28}{na:>8.3f}{ns:>7.3f}{nf:>7}")
print(f"{'GROUPED by recording':<28}{ga:>8.3f}{gs:>7.3f}{gf:>7}" + (f"   [{gl} folds leaked!]" if gl else "   [no recording leaked]"))
print(f"\ninflation from segment leakage: {na-ga:+.3f}")
if ga>=0.70: print("VERDICT: grouped AUC stays high -> AM2 signal is REAL (not segment leakage).")
elif ga>=0.60: print("VERDICT: grouped AUC moderate -> weak but plausibly real signal on AM2.")
else: print("VERDICT: grouped AUC near chance -> the earlier AM2 result was largely leakage.")