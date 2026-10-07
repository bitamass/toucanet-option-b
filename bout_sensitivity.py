"""bout_sensitivity.py -- leave-one-variable-out sensitivity with matched controls."""
import os, re, glob
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score

GAP_MIN = 60
SEED = 0
DIR = os.path.dirname(os.path.abspath(__file__))
META = {"clip_name","clip_label","clip_hour","sim_type","n_segments",
        "recorder","date","ts","bout","session","label"}

def parse(name):
    m = re.search(r'(?:Audio[_ ]?Moth[_ ]?|AM)(\d+).*?(\d{8})_(\d{6})', str(name), re.I)
    if m:
        return f"AM{int(m.group(1))}", m.group(2), pd.to_datetime(m.group(2)+m.group(3), format="%Y%m%d%H%M%S")
    return "AM?","NA", pd.NaT

def load():
    paths = [p for p in sorted(glob.glob(os.path.join(DIR,"am*_feature_matrix.csv")))
             if re.search(r'am\d+_feature_matrix\.csv$', os.path.basename(p), re.I)]
    frames=[]
    for p in paths:
        df = pd.read_csv(p)
        pr = df["clip_name"].apply(parse)
        meta = pd.DataFrame({"recorder":[a for a,_,_ in pr],"date":[b for _,b,_ in pr],
            "ts":[c for _,_,c in pr],"label":df["clip_label"].astype(int).values}, index=df.index)
        frames.append(pd.concat([df, meta], axis=1))
    d = pd.concat(frames, ignore_index=True).copy()
    d["bout"]=""
    for rec, idx in d.groupby("recorder").groups.items():
        sub = d.loc[idx].sort_values("ts")
        starts = sub.drop_duplicates("clip_name")[["clip_name","ts"]].sort_values("ts")
        gap = starts["ts"].diff().dt.total_seconds().div(60).fillna(0)
        name2bout = dict(zip(starts["clip_name"], (gap>GAP_MIN).cumsum()))
        d.loc[sub.index,"bout"] = rec + "|b" + sub["clip_name"].map(name2bout).astype(str)
    d["date"] = d["recorder"] + "|" + d["date"]
    return d

def feats_of(df):
    num = [c for c in df.columns if c not in META and pd.api.types.is_numeric_dtype(df[c])]
    return [c for c in num if not c.startswith("z_")]

def matched_pool(df, seed=0):
    rng = np.random.default_rng(seed); keep=[]
    for _, g in df.groupby("date"):
        pos=g.index[g.label==1].to_numpy(); neg=g.index[g.label==0].to_numpy()
        k=min(len(pos),len(neg))
        if k==0: continue
        keep+=list(rng.choice(pos,k,replace=False))+list(rng.choice(neg,k,replace=False))
    return df.loc[keep].copy()

def holdout(df, feats, col, seed=0):
    X=df[feats].values; y=df["label"].values; grp=df[col].values; rows=[]
    for gv in np.unique(grp.astype(str)):
        te=grp==gv; tr=~te
        if y[te].sum()<3 or (y[te]==0).sum()<3 or y[tr].sum()==0 or (y[tr]==0).sum()==0: continue
        m=HistGradientBoostingClassifier(random_state=seed).fit(X[tr],y[tr])
        pred=m.predict(X[te])
        try: auc=roc_auc_score(y[te], m.predict_proba(X[te])[:,1])
        except Exception: auc=np.nan
        rows.append(dict(group=gv, precision=precision_score(y[te],pred,zero_division=0),
            recall=recall_score(y[te],pred,zero_division=0), auc=auc))
    return pd.DataFrame(rows)

def main():
    d = load()
    print(f"loaded {len(d)} clips  |  recorders={d.recorder.nunique()}  dates={d.date.nunique()}  "
          f"bouts={d.bout.nunique()}  sim_types={d.sim_type.nunique()}\n")
    bp = d.groupby("bout")["label"].mean()
    print(f"pure bouts (all one class): {(((bp==0)|(bp==1)).mean()):.0%} of {d.bout.nunique()}")
    print("-> pure bouts can't be scored alone, so we use matched controls.\n")
    matched = matched_pool(d, SEED); feats = feats_of(d)
    print(f"matched pool: {len(matched)} clips (pos={int(matched.label.sum())}, "
          f"neg={int((matched.label==0).sum())}), {len(feats)} features\n")
    print("="*60); print("SENSITIVITY: honest AUC holding out each variable"); print("="*60)
    print(f"{'hold-out variable':<20}{'folds':>6}{'mean_AUC':>10}{'mean_P':>9}{'mean_R':>9}")
    for label,col in [("recorder","recorder"),("date (rec-day)","date"),("bout","bout"),("sim_type","sim_type")]:
        res=holdout(matched,feats,col,SEED)
        if res.empty: print(f"{label:<20}{'--':>6}   no scorable folds"); continue
        print(f"{label:<20}{len(res):>6}{res.auc.mean():>10.3f}{res.precision.mean():>9.3f}{res.recall.mean():>9.3f}")
    print("\n0.50 = chance, 0.70 = target. Consistent ~0.5 across rows = robust (no hidden signal).")
    holdout(matched,feats,"bout",SEED).to_csv(os.path.join(DIR,"bout_sensitivity.csv"),index=False)
    print(f"saved bout detail -> {os.path.join(DIR,'bout_sensitivity.csv')}")

if __name__=="__main__":
    main()