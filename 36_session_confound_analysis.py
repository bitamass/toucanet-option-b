"""
session_confound_analysis.py
================================================================
Tests whether the disturbance classifier is learning the actual
disturbance signal or just recognizing which recording SESSION a
clip came from. Implements the plan discussed with Sammy:

  STEP 0  Audit          - session x label cross-tab; is the data
                           structured so a fair test is even possible?
  STEP 1  Predictability - can session / recorder be predicted from
                           the 106 features? (negative-control probes)
  STEP 2  LOSSO          - leave-one-session-out evaluation, the
                           decisive test, with each held-out session
                           supplying its OWN positives AND negatives.
  (SHAP-by-session is intentionally deferred until STEP 2 flags a
   problem worth localizing.)

Run as-is to see it work on synthetic data, then set DATA_PATH and the
COLUMN NAMES in CONFIG to run on the real feature table.
================================================================
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# =====================================================================
# CONFIG  -- edit this block for the real data
# =====================================================================
CONFIG = {
    # None -> generate synthetic data so the script runs end-to-end.
    # Set to a .csv / .parquet path to use the real feature table.
    "data_path": None,

    # --- column names in your table -----------------------------------
    "recorder_col":  "recorder",       # e.g. AM1, AM2, ...
    "timestamp_col": "timestamp",      # local datetime of the clip
    "label_col":     "label",          # 1 = simulation/disturbance, 0 = background
    "sim_type_col":  None,             # OR: a "Sim Type" column; non-empty => positive
    "session_col":   None,             # set if a session id already exists; else built below
    "recording_id_col": None,          # set if a continuous-recording id exists

    # --- the 106 feature columns --------------------------------------
    # None -> auto-detect (all numeric cols that aren't ids/labels).
    "feature_cols": None,

    # --- session definition -------------------------------------------
    # "recorder_day"      : recorder + calendar date          (default)
    # "recording_id"      : use recording_id_col directly
    # "recorder_hour_bin" : recorder + N-hour block (set session_hours)
    "session_def": "recorder_day",
    "session_hours": 6,

    # --- model ---------------------------------------------------------
    "model": "gb",            # "gb" (HistGradientBoosting) or "lr"
    "min_test_pos": 3,        # skip LOSSO folds with fewer positives than this
    "min_test_neg": 3,        # skip folds without enough negatives
    "random_state": 0,
}


# =====================================================================
# Synthetic data -- lets you see the outputs before wiring real data.
# Deliberately builds in a session confound so the probes light up.
# =====================================================================
def make_synthetic(n_sessions=10, n_recorders=5, clips_per_session=120,
                    n_features=106, sim_sessions=(1, 2, 5), seed=0):
    rng = np.random.default_rng(seed)
    rows, feats = [], []
    start = pd.Timestamp("2026-03-01 04:00:00")
    for s in range(n_sessions):
        rec = f"AM{(s % n_recorders) + 1}"
        day = start + pd.Timedelta(days=s)
        # each session has its own acoustic "fingerprint" (the confound)
        session_shift = rng.normal(0, 1.0, n_features)
        has_sim = s in sim_sessions
        for c in range(clips_per_session):
            ts = day + pd.Timedelta(minutes=int(rng.integers(0, 60 * 12)))
            # positives only exist in sim sessions, ~35% of their clips
            is_pos = int(has_sim and rng.random() < 0.35)
            base = rng.normal(0, 1.0, n_features) + session_shift
            if is_pos:
                # a genuine, weak disturbance signal on a few dims
                base[:8] += 0.9
            rows.append({"clip_id": f"{rec}_{s}_{c}", "recorder": rec,
                         "timestamp": ts, "label": is_pos})
            feats.append(base)
    df = pd.DataFrame(rows)
    F = pd.DataFrame(feats, columns=[f"f{i}" for i in range(n_features)])
    return pd.concat([df, F], axis=1)


# =====================================================================
# Loading + session construction
# =====================================================================
def load_data(cfg):
    if cfg["data_path"] is None:
        print(">> No data_path set -- generating synthetic data with a built-in "
              "session confound.\n")
        return make_synthetic(seed=cfg["random_state"])
    p = cfg["data_path"]
    df = pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)
    return df


def build_label(df, cfg):
    if cfg["sim_type_col"]:
        st = df[cfg["sim_type_col"]]
        df["label"] = (~(st.isna() | (st.astype(str).str.strip() == ""))).astype(int)
        cfg["label_col"] = "label"
    else:
        df[cfg["label_col"]] = df[cfg["label_col"]].astype(int)
    return df


def build_session(df, cfg):
    if cfg["session_col"]:
        df["session"] = df[cfg["session_col"]].astype(str)
        return df
    rec = df[cfg["recorder_col"]].astype(str)
    ts = pd.to_datetime(df[cfg["timestamp_col"]])
    kind = cfg["session_def"]
    if kind == "recording_id" and cfg["recording_id_col"]:
        df["session"] = df[cfg["recording_id_col"]].astype(str)
    elif kind == "recorder_hour_bin":
        h = cfg["session_hours"]
        block = ts.dt.floor(f"{h}h")
        df["session"] = rec + "|" + block.dt.strftime("%Y-%m-%d_%H")
    else:  # recorder_day (default)
        df["session"] = rec + "|" + ts.dt.strftime("%Y-%m-%d")
    return df


def get_feature_cols(df, cfg):
    if cfg["feature_cols"]:
        return cfg["feature_cols"]
    drop = {cfg["recorder_col"], cfg["timestamp_col"], cfg["label_col"],
            cfg.get("sim_type_col"), cfg.get("session_col"),
            cfg.get("recording_id_col"), "session", "clip_id", "label"}
    drop = {c for c in drop if c}
    cols = [c for c in df.columns
            if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
    return cols


def make_model(cfg):
    if cfg["model"] == "lr":
        return LogisticRegression(max_iter=2000, class_weight="balanced")
    return HistGradientBoostingClassifier(random_state=cfg["random_state"])


# =====================================================================
# STEP 0 -- Audit
# =====================================================================
def audit(df, cfg):
    print("=" * 68)
    print("STEP 0  DATA AUDIT  (session x label)")
    print("=" * 68)
    g = df.groupby("session")["label"].agg(n="count", pos="sum")
    g["neg"] = g["n"] - g["pos"]
    g["pos_frac"] = (g["pos"] / g["n"]).round(3)
    g = g.sort_values("pos", ascending=False)

    n_sessions = len(g)
    sess_with_pos = int((g["pos"] > 0).sum())
    sess_with_both = int(((g["pos"] > 0) & (g["neg"] > 0)).sum())
    total_pos = int(g["pos"].sum())
    top = g["pos"].head(3).sum()
    conc = (top / total_pos) if total_pos else 0

    print(f"sessions total ............... {n_sessions}")
    print(f"sessions with positives ...... {sess_with_pos}   "
          f"(= number of LOSSO folds)")
    print(f"sessions with BOTH pos & neg . {sess_with_both}   "
          f"(usable folds for within-session negatives)")
    print(f"total positives .............. {total_pos}")
    print(f"positives in top-3 sessions .. {conc:.0%}  (concentration)\n")
    print("per-session counts (top 12):")
    print(g.head(12).to_string(), "\n")

    # -- interpretation / warnings ------------------------------------
    if sess_with_pos < 5:
        print(f"  [!] Only {sess_with_pos} sim sessions -> LOSSO will be noisy. "
              "Report per-session results, not just the mean.")
    if sess_with_both < sess_with_pos:
        print(f"  [!] {sess_with_pos - sess_with_both} sim session(s) have NO "
              "background clips. Those folds can't supply within-session "
              "negatives -> precision there is undefined / confounded.")
    neg_only = int(((g["pos"] == 0) & (g["neg"] > 0)).sum())
    if neg_only and sess_with_both == 0:
        print("  [!] Positives and negatives live in DIFFERENT sessions. "
              "A pure 'which session?' detector would score ~perfectly on a "
              "naive split -- fix negative sampling before trusting any P/R.")
    print()
    return g


# =====================================================================
# STEP 1 -- Predictability probes (negative controls)
# =====================================================================
def predictability(df, feats, target_col, cfg, name):
    y = df[target_col].astype("category").cat.codes.values
    X = df[feats].values
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        print(f"  {name}: only one class present -- skipped."); return
    chance = counts.max() / counts.sum()          # majority-class baseline
    # need >=2 per class for stratified 3-fold
    k = 3 if counts.min() >= 3 else 2
    if counts.min() < 2:
        print(f"  {name}: some class has <2 samples -- skipped."); return
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=cfg["random_state"])
    accs = []
    for tr, te in skf.split(X, y):
        m = make_model(cfg)
        m.fit(X[tr], y[tr])
        accs.append((m.predict(X[te]) == y[te]).mean())
    acc = np.mean(accs)
    lift = acc - chance
    flag = "  <-- lots of identity info in features" if lift > 0.15 else ""
    print(f"  {name:<26} acc={acc:.3f}  chance={chance:.3f}  "
          f"lift=+{lift:.3f}{flag}")


def run_predictability(df, feats, cfg):
    print("=" * 68)
    print("STEP 1  PREDICTABILITY PROBES  (can features reveal session/recorder?)")
    print("=" * 68)
    print("  High lift over chance => that identity is encoded in the features")
    print("  and is available for the disturbance model to exploit.\n")
    predictability(df, feats, "session", cfg, "predict SESSION")
    predictability(df, feats, cfg["recorder_col"], cfg, "predict RECORDER")
    print()


# =====================================================================
# STEP 2 -- Leave-one-simulation-session-out (the decisive test)
# =====================================================================
def losso(df, feats, cfg):
    print("=" * 68)
    print("STEP 2  LEAVE-ONE-SESSION-OUT  (train on the rest, test on unseen "
          "session)")
    print("=" * 68)
    X = df[feats].values
    y = df["label"].values
    sess = df["session"].values
    rec = df[cfg["recorder_col"]].astype(str).values

    pos_sessions = [s for s in np.unique(sess)
                    if y[sess == s].sum() >= cfg["min_test_pos"]]
    rows = []
    for s in pos_sessions:
        te = sess == s
        tr = ~te
        n_pos, n_neg = int(y[te].sum()), int((y[te] == 0).sum())
        if n_neg < cfg["min_test_neg"]:
            rows.append({"session": s, "recorder": rec[te][0],
                         "n_pos": n_pos, "n_neg": n_neg,
                         "precision": np.nan, "recall": np.nan,
                         "f1": np.nan, "auc": np.nan,
                         "note": "no within-session negatives"})
            continue
        if y[tr].sum() == 0:
            continue
        m = make_model(cfg)
        m.fit(X[tr], y[tr])
        pred = m.predict(X[te])
        try:
            proba = m.predict_proba(X[te])[:, 1]
            auc = roc_auc_score(y[te], proba)
        except Exception:
            auc = np.nan
        rows.append({
            "session": s, "recorder": rec[te][0],
            "n_pos": n_pos, "n_neg": n_neg,
            "precision": precision_score(y[te], pred, zero_division=0),
            "recall": recall_score(y[te], pred, zero_division=0),
            "f1": f1_score(y[te], pred, zero_division=0),
            "auc": auc, "note": ""})

    res = pd.DataFrame(rows)
    if res.empty:
        print("  No usable folds. See the audit warnings above.\n")
        return res

    print("per held-out session:")
    show = res.copy()
    for c in ["precision", "recall", "f1", "auc"]:
        show[c] = show[c].round(3)
    print(show.to_string(index=False), "\n")

    ok = res.dropna(subset=["precision"])
    if not ok.empty:
        print("summary across held-out sessions (unweighted mean +/- sd):")
        for c in ["precision", "recall", "f1", "auc"]:
            print(f"    {c:<10} {ok[c].mean():.3f} +/- {ok[c].std(ddof=0):.3f}")
        # per-recorder view mirrors the P/R>=0.70 target table
        print("\nper-recorder mean (matches the AM1..AM6 target framing):")
        pr = ok.groupby("recorder")[["precision", "recall"]].mean().round(3)
        pr["meets_0.70"] = (pr["precision"] >= 0.70) & (pr["recall"] >= 0.70)
        print(pr.to_string())
        print("\n  Read: if precision/recall hold up here, performance survives on "
              "sessions\n  the model never trained on -> evidence it's the "
              "disturbance, not the session.\n  If they collapse vs the naive "
              "split, the earlier numbers leaned on session cues.\n")
    return res


# =====================================================================
def main():
    cfg = CONFIG
    df = load_data(cfg)
    df = build_label(df, cfg)
    df = build_session(df, cfg)
    feats = get_feature_cols(df, cfg)
    print(f"rows={len(df)}  features={len(feats)}  "
          f"positives={int(df['label'].sum())}  "
          f"session_def={cfg['session_def']}\n")

    audit(df, cfg)
    run_predictability(df, feats, cfg)
    res = losso(df, feats, cfg)

    # write results next to THIS script, so it lands in the same folder
    # regardless of which directory you launched python from.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(script_dir, "losso_results.csv")
    res.to_csv(out, index=False)
    print(f"saved per-session LOSSO results -> {out}")


if __name__ == "__main__":
    main()