"""shap_by_session.py -- does the model rely on the same acoustic cues across sessions?"""
import os, re, glob
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr

SEED = 0
DIR = os.path.dirname(os.path.abspath(__file__))
META = {"clip_name","clip_label","clip_hour","sim_type","n_segments","recorder","date","session","label"}

def parse(n):
    m=re.search(r'(?:Audio[_ ]?Moth[_ ]?|AM)(\d+).*?(\d{8})',str(n),re.I)
    return (f"AM{int(m.group(1))}", m.group(2)) if m else ("AM?","NA")

def load():
    paths=[p for p in sorted(glob.glob(os.path.join(DIR,"am*_feature_matrix.csv")))
           if re.search(r'am\d+_feature_matrix\.csv$', os.path.basename(p), re.I)]
    frames=[]
    for p in paths:
        df=pd.read_csv(p)
        pr=df["clip_name"].apply(parse)
        meta=pd.DataFrame({"recorder":[a for a,_ in pr],"date":[b for _,b in pr],
                           "label":df["clip_label"].astype(int).values}, index=df.index)
        meta["session"]=meta["recorder"]+"|"+meta["date"]
        frames.append(pd.concat([df,meta],axis=1))
    return pd.concat(frames, ignore_index=True)

def hand_features(df):
    num=[c for c in df.columns if c not in META and pd.api.types.is_numeric_dtype(df[c])]
    return [c for c in num if not c.startswith("z_") and not c.startswith("emb_pc")]

def matched(df, seed=0):
    rng=np.random.default_rng(seed); keep=[]
    for _,g in df.groupby("session"):
        pos=g.index[g.label==1].to_numpy(); neg=g.index[g.label==0].to_numpy()
        k=min(len(pos),len(neg))
        if k==0: continue
        keep+=list(rng.choice(pos,k,replace=False))+list(rng.choice(neg,k,replace=False))
    return df.loc[keep].copy()

def main():
    df=load(); feats=hand_features(df)
    pool=matched(df, SEED)
    print(f"matched pool: {len(pool)} clips, {len(feats)} hand features, {pool.session.nunique()} sessions\n")
    model=HistGradientBoostingClassifier(random_state=SEED).fit(pool[feats].values, pool["label"].values)
    imp={}
    for s,g in pool.groupby("session"):
        if g.label.sum()<5 or (g.label==0).sum()<5: continue
        r=permutation_importance(model, g[feats].values, g["label"].values,
                                 n_repeats=5, random_state=SEED, scoring="roc_auc")
        imp[s]=r.importances_mean
    if len(imp)<2: print("Not enough sessions."); return
    sessions=list(imp.keys()); I=np.vstack([imp[s] for s in sessions])
    print("="*66); print("TOP 5 FEATURES BY SESSION"); print("="*66)
    for s in sessions:
        order=np.argsort(imp[s])[::-1][:5]
        print(f"{s:<16} " + ", ".join(feats[i] for i in order))
    print("\n"+"="*66); print("CROSS-SESSION CONSISTENCY"); print("="*66)
    corrs=[]
    for i in range(len(sessions)):
        for j in range(i+1,len(sessions)):
            rho=spearmanr(I[i],I[j]).correlation
            if rho==rho: corrs.append(rho)
    top10=[set(np.argsort(v)[::-1][:10]) for v in I]; jac=[]
    for i in range(len(sessions)):
        for j in range(i+1,len(sessions)):
            u=top10[i]|top10[j]; jac.append(len(top10[i]&top10[j])/len(u) if u else np.nan)
    print(f"mean pairwise rank correlation of importances: {np.nanmean(corrs):+.3f}")
    print(f"mean top-10 feature overlap (Jaccard):          {np.nanmean(jac):.3f}")
    gi=np.argsort(I.mean(0))[::-1][:8]
    print("\nmost important overall:", ", ".join(feats[i] for i in gi))
    print("\nReading it:")
    print("  High correlation/overlap (~0.5+) -> same cues everywhere = consistent signal.")
    print("  Low/near-zero -> each session leans on different features = session quirks.")

if __name__=="__main__":
    main()