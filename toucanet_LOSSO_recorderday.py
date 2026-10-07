"""
run_session_confound_toucanet.py
================================================================
Adapts the session-confound analysis to the real toucanet data.

Reads the per-recorder feature matrices (am1/2/4/5/6_feature_matrix.csv),
parses recorder + session + timestamp out of `clip_name`
(e.g. "Audio_Moth_1_20250318_181418.wav"), uses `clip_label` as the
target (1 = disturbance, 0 = baseline), and runs:

  STEP 0  Audit          - session x label cross-tab
  STEP 1  Predictability - can session / recorder be predicted from features?
  STEP 2  LOSSO          - leave-one-session-out (each held-out session
                           supplies its own positives AND negatives)

A "session" here = one recording file (recorder + full timestamp), so all
segments from the same clip stay together -> no within-clip leakage.
================================================================
"""

import os
import re
import glob
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# =====================================================================
# CONFIG
# =====================================================================
CFG = {
    # folder holding the am*_feature_matrix.csv files (default: this script's dir)
    "data_dir": None,
    "file_glob": "am*_feature_matrix.csv",

    "label_col": "clip_label",         # 1 = disturbance, 0 = baseline
    "clip_name_col": "clip_name",      # used to parse recorder / session / time

    # which feature set to test:
    #   "raw"  -> acoustic + embedding PCs (what a classifier really uses)
    #   "z"    -> the z_* normalized features + embedding PCs
    #   "all"  -> everything numeric
    "feature_set": "raw",

    "model": "gb",            # "gb" or "lr"
    "min_test_pos": 3,
    "min_test_neg": 3,
    "random_state": 0,
}

# metadata / non-feature columns to never treat as features
META_COLS = {"clip_name", "clip_label", "clip_hour", "sim_type", "n_segments",
             "recorder", "session", "timestamp", "label"}


# =====================================================================
# Load + parse the real data
# =====================================================================
def parse_clip_name(name):
    """'Audio_Moth_1_20250318_181418.wav' -> ('AM1', '20250318')  (recorder-day)"""
    base = str(name)
    m = re.search(r'(?:Audio[_ ]?Moth[_ ]?|AM)(\d+).*?(\d{8})_\d{6}', base, re.I)
    if m:
        return f"AM{int(m.group(1))}", m.group(2)   # session grain = day
    rec = re.search(r'(\d+)', base)
    day = re.search(r'(\d{8})', base)
    return (f"AM{int(rec.group(1))}" if rec else "AM?"), (day.group(1) if day else base)


def load_all(cfg):
    data_dir = cfg["data_dir"] or os.path.dirname(os.path.abspath(__file__))
    paths = sorted(glob.glob(os.path.join(data_dir, cfg["file_glob"])))
    # keep only the plain per-recorder matrices, skip *_full_* / *_time_controlled_*
    paths = [p for p in paths
             if re.search(r'am\d+_feature_matrix\.csv$', os.path.basename(p), re.I)]
    if not paths:
        raise SystemExit(f"No feature matrices matched in {data_dir}")
    print("loading:")
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        rec_from_file = re.search(r'am(\d+)_', os.path.basename(p), re.I)
        parsed = df[cfg["clip_name_col"]].apply(parse_clip_name)
        recorder = pd.Series([r for r, _ in parsed], index=df.index)
        session_time = pd.Series([t for _, t in parsed], index=df.index)
        if rec_from_file is not None:
            recorder = recorder.where(recorder != "AM?",
                                      f"AM{int(rec_from_file.group(1))}")
        meta = pd.DataFrame({
            "recorder": recorder,
            "session_time": session_time,
            "session": recorder + "|" + session_time,
            "label": df[cfg["label_col"]].astype(int),
        }, index=df.index)
        df = pd.concat([df, meta], axis=1)
        print(f"  {os.path.basename(p):32s} rows={len(df):6d}  "
              f"pos={int(df['label'].sum()):5d}  sessions={df['session'].nunique()}")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def pick_features(df, cfg):
    numeric = [c for c in df.columns
               if c not in META_COLS and pd.api.types.is_numeric_dtype(df[c])]
    emb = [c for c in numeric if c.startswith("emb_pc")]
    zc = [c for c in numeric if c.startswith("z_")]
    raw = [c for c in numeric if not c.startswith("z_") and not c.startswith("emb_pc")]
    if cfg["feature_set"] == "z":
        cols = zc + emb
    elif cfg["feature_set"] == "all":
        cols = numeric
    else:  # raw
        cols = raw + emb
    return cols


def make_model(cfg):
    if cfg["model"] == "lr":
        return LogisticRegression(max_iter=2000, class_weight="balanced")
    return HistGradientBoostingClassifier(random_state=cfg["random_state"])


# =====================================================================
# STEP 0 -- Audit
# =====================================================================
def audit(df):
    print("=" * 68)
    print("STEP 0  DATA AUDIT  (session x label)")
    print("=" * 68)
    g = df.groupby("session")["label"].agg(n="count", pos="sum")
    g["neg"] = g["n"] - g["pos"]
    g = g.sort_values("pos", ascending=False)
    sess_with_pos = int((g["pos"] > 0).sum())
    sess_with_both = int(((g["pos"] > 0) & (g["neg"] > 0)).sum())
    total_pos = int(g["pos"].sum())
    conc = g["pos"].head(3).sum() / total_pos if total_pos else 0
    print(f"sessions (recordings) total .. {len(g)}")
    print(f"sessions with positives ...... {sess_with_pos}   (= LOSSO folds)")
    print(f"sessions with BOTH pos & neg . {sess_with_both}   (usable folds)")
    print(f"total positive segments ...... {total_pos}")
    print(f"positives in top-3 sessions .. {conc:.0%}\n")
    print("per-session counts (top 15):")
    print(g.head(15).to_string(), "\n")
    if sess_with_pos < 5:
        print(f"  [!] Only {sess_with_pos} sim sessions -> LOSSO will be noisy; "
              "lean on per-session results.")
    if sess_with_both < sess_with_pos:
        print(f"  [!] {sess_with_pos - sess_with_both} sim session(s) have no "
              "baseline segments -> precision undefined there.")
    # recorder-level view
    print("\nby recorder:")
    r = df.groupby("recorder")["label"].agg(n="count", pos="sum")
    r["sessions"] = df.groupby("recorder")["session"].nunique()
    r["pos_sessions"] = df[df.label == 1].groupby("recorder")["session"].nunique()
    print(r.fillna(0).to_string(), "\n")
    return g


# =====================================================================
# STEP 1 -- Predictability probes
# =====================================================================
def probe(df, feats, target, cfg, name):
    y = df[target].astype("category").cat.codes.values
    X = df[feats].values
    cls, cnt = np.unique(y, return_counts=True)
    if len(cls) < 2 or cnt.min() < 2:
        print(f"  {name}: not enough per-class data -- skipped."); return
    chance = cnt.max() / cnt.sum()
    k = 3 if cnt.min() >= 3 else 2
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=cfg["random_state"])
    accs = [ (make_model(cfg).fit(X[tr], y[tr]).predict(X[te]) == y[te]).mean()
             for tr, te in skf.split(X, y) ]
    acc = float(np.mean(accs)); lift = acc - chance
    flag = "  <-- lots of identity info" if lift > 0.15 else ""
    print(f"  {name:<24} acc={acc:.3f}  chance={chance:.3f}  lift=+{lift:.3f}{flag}")


def predictability(df, feats, cfg):
    print("=" * 68)
    print("STEP 1  PREDICTABILITY PROBES")
    print("=" * 68)
    print("  High lift over chance => that identity is encoded in the features.\n")
    probe(df, feats, "session", cfg, "predict SESSION")
    probe(df, feats, "recorder", cfg, "predict RECORDER")
    print()


# =====================================================================
# STEP 2 -- Leave-one-session-out
# =====================================================================
def losso(df, feats, cfg):
    print("=" * 68)
    print("STEP 2  LEAVE-ONE-SESSION-OUT")
    print("=" * 68)
    X = df[feats].values; y = df["label"].values
    sess = df["session"].values; rec = df["recorder"].values
    pos_sessions = [s for s in np.unique(sess) if y[sess == s].sum() >= cfg["min_test_pos"]]
    rows = []
    for s in pos_sessions:
        te = sess == s; tr = ~te
        n_pos, n_neg = int(y[te].sum()), int((y[te] == 0).sum())
        if n_neg < cfg["min_test_neg"] or y[tr].sum() == 0:
            rows.append(dict(session=s, recorder=rec[te][0], n_pos=n_pos, n_neg=n_neg,
                             precision=np.nan, recall=np.nan, f1=np.nan, auc=np.nan,
                             note="no/low within-session negatives"))
            continue
        m = make_model(cfg).fit(X[tr], y[tr])
        pred = m.predict(X[te])
        try:
            auc = roc_auc_score(y[te], m.predict_proba(X[te])[:, 1])
        except Exception:
            auc = np.nan
        rows.append(dict(session=s, recorder=rec[te][0], n_pos=n_pos, n_neg=n_neg,
                         precision=precision_score(y[te], pred, zero_division=0),
                         recall=recall_score(y[te], pred, zero_division=0),
                         f1=f1_score(y[te], pred, zero_division=0), auc=auc, note=""))
    res = pd.DataFrame(rows)
    if res.empty:
        print("  No usable folds.\n"); return res
    show = res.copy()
    for c in ["precision", "recall", "f1", "auc"]:
        show[c] = show[c].round(3)
    print("per held-out session:")
    print(show.to_string(index=False), "\n")
    ok = res.dropna(subset=["precision"])
    if not ok.empty:
        print("summary across held-out sessions (unweighted mean +/- sd):")
        for c in ["precision", "recall", "f1", "auc"]:
            print(f"    {c:<10} {ok[c].mean():.3f} +/- {ok[c].std(ddof=0):.3f}")
        print("\nper-recorder mean (matches AM1..AM6 / P,R>=0.70 framing):")
        pr = ok.groupby("recorder")[["precision", "recall"]].mean().round(3)
        pr["meets_0.70"] = (pr["precision"] >= 0.70) & (pr["recall"] >= 0.70)
        print(pr.to_string())
    return res


# =====================================================================
def main():
    df = load_all(CFG)
    feats = pick_features(df, CFG)
    print(f"\ntotal rows={len(df)}  features={len(feats)} ({CFG['feature_set']} set)  "
          f"positives={int(df['label'].sum())}\n")
    audit(df)
    predictability(df, feats, CFG)
    res = losso(df, feats, CFG)
    out = os.path.join(CFG["data_dir"] or os.path.dirname(os.path.abspath(__file__)),
                       "losso_recorderday.csv")
    res.to_csv(out, index=False)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()