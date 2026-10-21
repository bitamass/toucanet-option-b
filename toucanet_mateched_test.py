"""
toucanet_matched_test.py
================================================================
The recovery experiment. Two things at once:

1. MATCHED CONTROLS.  For every disturbance clip we keep an equal number
   of baseline clips from the SAME recorder-day and SAME hour. After this,
   session identity and hour can no longer separate positives from
   negatives (each cell is ~50/50), so a model can ONLY score by hearing
   the disturbance. This removes the leak at its source.

2. FEATURE-SET HEAD-TO-HEAD.  Runs the honest leave-one-session-out test
   on the matched data for three feature sets:
     - hand : the ~40 hand-crafted acoustic features
     - emb  : the 20 embedding PCs (emb_pc*)
     - both : hand + emb
   If ANY of these gives AUC clearly above 0.5 on held-out sessions,
   there is real, non-leaked disturbance signal to build on.

Reads am*_feature_matrix.csv from this script's folder. No edits needed.
================================================================
"""

import os, re, glob
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score

SEED = 0
MATCH_BY_HOUR = True          # True = recorder-day + hour; False = recorder-day only
RATIO = 1                     # negatives : positives per cell (1 = strict 1:1)

DIR = os.path.dirname(os.path.abspath(__file__))
META = {"clip_name", "clip_label", "clip_hour", "sim_type", "n_segments",
        "recorder", "session", "label"}


def parse(name):
    m = re.search(r'(?:Audio[_ ]?Moth[_ ]?|AM)(\d+).*?(\d{8})_\d{6}', str(name), re.I)
    if m:
        return f"AM{int(m.group(1))}", m.group(2)
    rec = re.search(r'(\d+)', str(name)); day = re.search(r'(\d{8})', str(name))
    return (f"AM{int(rec.group(1))}" if rec else "AM?"), (day.group(1) if day else "NA")


def load():
    paths = [p for p in sorted(glob.glob(os.path.join(DIR, "am*_feature_matrix.csv")))
             if re.search(r'am\d+_feature_matrix\.csv$', os.path.basename(p), re.I)]
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        pr = df["clip_name"].apply(parse)
        rec = pd.Series([a for a, _ in pr], index=df.index)
        day = pd.Series([b for _, b in pr], index=df.index)
        meta = pd.DataFrame({"recorder": rec, "session": rec + "|" + day,
                             "label": df["clip_label"].astype(int)}, index=df.index)
        frames.append(pd.concat([df, meta], axis=1))
    return pd.concat(frames, ignore_index=True)


def feature_groups(df):
    numeric = [c for c in df.columns
               if c not in META and pd.api.types.is_numeric_dtype(df[c])]
    emb = [c for c in numeric if c.startswith("emb_pc")]
    hand = [c for c in numeric
            if not c.startswith("emb_pc") and not c.startswith("z_")]
    return {"hand": hand, "emb": emb, "both": hand + emb}


def matched_subset(df, by_hour=True, ratio=1, seed=0):
    rng = np.random.default_rng(seed)
    keys = ["session", "clip_hour"] if by_hour else ["session"]
    keep = []
    for _, g in df.groupby(keys):
        pos = g.index[g.label == 1].to_numpy()
        neg = g.index[g.label == 0].to_numpy()
        k = min(len(pos), len(neg))
        if k == 0:
            continue                      # cell has only one class -> drop (it's the confounded part)
        keep += list(rng.choice(pos, k, replace=False))
        keep += list(rng.choice(neg, min(len(neg), ratio * k), replace=False))
    return df.loc[keep].copy()


def session_out(df, feats, seed=0):
    X = df[feats].values; y = df["label"].values
    sess = df["session"].values; rec = df["recorder"].values
    rows = []
    for s in np.unique(sess):
        te = sess == s; tr = ~te
        if y[te].sum() < 3 or (y[te] == 0).sum() < 3 or y[tr].sum() == 0:
            continue
        m = HistGradientBoostingClassifier(random_state=seed).fit(X[tr], y[tr])
        pred = m.predict(X[te])
        try:
            auc = roc_auc_score(y[te], m.predict_proba(X[te])[:, 1])
        except Exception:
            auc = np.nan
        rows.append(dict(session=s, recorder=rec[te][0],
                         precision=precision_score(y[te], pred, zero_division=0),
                         recall=recall_score(y[te], pred, zero_division=0), auc=auc))
    return pd.DataFrame(rows)


def main():
    df = load()
    groups = feature_groups(df)
    print(f"loaded {len(df)} clips, {int(df.label.sum())} positive, "
          f"{df.session.nunique()} sessions\n")

    matched = matched_subset(df, MATCH_BY_HOUR, RATIO, SEED)
    lvl = "recorder-day + hour" if MATCH_BY_HOUR else "recorder-day"
    print(f"MATCHED CONTROLS ({lvl}, {RATIO}:1)")
    print(f"  matched clips: {len(matched)}  "
          f"(pos={int(matched.label.sum())}, neg={int((matched.label==0).sum())})")
    print(f"  sessions with both classes after matching: "
          f"{matched.groupby('session').label.nunique().eq(2).sum()}\n")

    print("=" * 62)
    print("HONEST SESSION-OUT ON MATCHED DATA  (AUC 0.5 = no signal)")
    print("=" * 62)
    print(f"{'feature set':<8} {'mean_AUC':>9} {'mean_P':>8} {'mean_R':>8} "
          f"{'>=0.70':>7}")
    summary = {}
    for name, feats in groups.items():
        res = session_out(matched, feats, SEED)
        if res.empty:
            print(f"{name:<8}  no usable folds"); continue
        pr = res.groupby("recorder")[["precision", "recall"]].mean()
        meets = int(((pr.precision >= 0.70) & (pr.recall >= 0.70)).sum())
        summary[name] = res
        print(f"{name:<8} {res.auc.mean():>9.3f} {res.precision.mean():>8.3f} "
              f"{res.recall.mean():>8.3f} {meets:>4}/{pr.shape[0]}")

    # per-recorder detail for the best feature set (by mean AUC)
    if summary:
        best = max(summary, key=lambda k: summary[k].auc.mean())
        print(f"\nper-recorder detail  [{best}]:")
        d = summary[best].groupby("recorder")[["precision", "recall", "auc"]].mean().round(3)
        d["meets_0.70"] = (d.precision >= 0.70) & (d.recall >= 0.70)
        print(d.to_string())
        out = os.path.join(DIR, "matched_sessionout.csv")
        summary[best].to_csv(out, index=False)
        print(f"\nsaved best per-session results -> {out}")

    print("\nHow to read:")
    print("  AUC still ~0.5 across all sets -> the disturbance signal does not")
    print("     survive once session/hour context is removed (clean negative).")
    print("  AUC clearly >0.5 for some set -> real signal there; build on it.")


if __name__ == "__main__":
    main()