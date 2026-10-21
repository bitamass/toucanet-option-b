"""
76_all_clips_model.py
---------------------
One model over ALL clips: is a bird heard less often when people are present,
after accounting for recorder, day and time of day?

Data: bioacoustics_dataset.csv (all 3-s clips, correct simulation labels).
Unit: 1-minute bins per recorder (kept if at least 15 of 20 clips recorded).
Outcomes (decided before running):
  PRIMARY   bird_frac  = share of clips in the minute with a bird (BirdNET conf >= 0.25)
  SECONDARY mean_conf  = average of each clip's top BirdNET confidence (0 if none)
Model (least squares):
  outcome ~ DURING + AFTER_0-20 + AFTER_20-40 + AFTER_40-60 min
            + recorder x day  + recorder x hour-of-day   (fixed effects)
PRIMARY analysis uses minutes 06:00-18:00 (diurnal birds); all hours reported too.

Uncertainty is judged two ways:
  - cluster-robust standard errors by recorder-day
  - a placebo test: each session is moved to the same clock time on another day
    at the same recorder (no real session within 2 h) and the model refit,
    300 times. The real effect is compared with this null distribution, which
    respects how bird activity drifts within a day.

Usage: python 76_all_clips_model.py "C:\\...\\bioacoustics_dataset.csv"
Writes: all_clips_model_results.txt
"""
import ast
import sys
import numpy as np
import pandas as pd

PATH, CONF, N_PERM, SEED = sys.argv[1], 0.25, 300, 0
rng = np.random.default_rng(SEED)

want = {"clip_name", "Recorder", "Timestamp", "species", "confidence", "Sim Type", "Sim Relative Time"}
df = pd.read_csv(PATH, usecols=lambda c: c in want, low_memory=False)
df["clip_name"] = df["clip_name"].astype(str)
df = df.drop_duplicates("clip_name")
df["ts"] = pd.to_datetime(df["Timestamp"].astype(str), format="%Y%m%d_%H%M%S", errors="coerce")
df = df.dropna(subset=["ts"])


def as_list(v):
    if isinstance(v, str) and v.strip().startswith("["):
        try:
            return list(ast.literal_eval(v))
        except (ValueError, SyntaxError):
            return []
    if v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() in ("", "nan", "None"):
        return []
    return [v]


def top_conf(row):
    best = 0.0
    for c in as_list(row["confidence"]):
        try:
            best = max(best, float(c))
        except (TypeError, ValueError):
            pass
    return best


print("Parsing BirdNET detections (takes a minute or two)...")
df["conf"] = df.apply(top_conf, axis=1)
df["bird"] = (df.conf >= CONF).astype(float)
df["mean_conf_clip"] = np.where(df.conf >= CONF, df.conf, 0.0)

# ---- sessions --------------------------------------------------------------
sim = df[df["Sim Type"].astype(str).str.strip() != "[]"].copy()
sim["rel"] = pd.to_numeric(sim["Sim Relative Time"].astype(str).str.strip("[]").str.split(",").str[0],
                           errors="coerce").fillna(0)
sim["event"] = sim.ts - pd.to_timedelta(sim.rel, unit="s")
sess = []
for r, g in sim.groupby("Recorder"):
    e = g.sort_values("event")
    grp = (e.event.diff() > pd.Timedelta("20min")).cumsum()
    for _, x in e.groupby(grp):
        sess.append(dict(rec=r, start=x.ts.min().floor("min"), end=x.ts.max().ceil("min"),
                         centre=x.event.median()))
sess = pd.DataFrame(sess)
print(f"  {len(df):,} clips, {len(sess)} sessions")

# ---- minute bins -------------------------------------------------------------
df["minute"] = df.ts.dt.floor("min")
m = df.groupby(["Recorder", "minute"]).agg(n=("bird", "size"), bird_frac=("bird", "mean"),
                                           mean_conf=("mean_conf_clip", "mean")).reset_index()
m = m[m.n >= 15].reset_index(drop=True)
m["rec_day"] = m.Recorder + "_" + m.minute.dt.strftime("%m%d")
m["rec_hour"] = m.Recorder + "_" + m.minute.dt.hour.astype(str)
m["hour"] = m.minute.dt.hour
print(f"  {len(m):,} usable minutes")

TERMS = ["during", "after_0_20", "after_20_40", "after_40_60"]


def indicators(frame, sessions):
    out = np.zeros((len(frame), 4))
    rec, t = frame.Recorder.values, frame.minute.values
    for s in sessions.itertuples():
        r = rec == s.rec
        bounds = [(s.start, s.end), (s.end, s.end + pd.Timedelta("20min")),
                  (s.end + pd.Timedelta("20min"), s.end + pd.Timedelta("40min")),
                  (s.end + pd.Timedelta("40min"), s.end + pd.Timedelta("60min"))]
        for k, (a, b) in enumerate(bounds):
            out[r & (t >= np.datetime64(a)) & (t < np.datetime64(b)), k] = 1
    return out


def demean(v, codes, sizes, tol=1e-8, max_iter=500):
    """Remove recorder-day and recorder-hour averages (two-way fixed effects)."""
    v = v.astype(float).copy()
    for _ in range(max_iter):
        change = 0.0
        for c, n in zip(codes, sizes):
            means = np.bincount(c, weights=v, minlength=n) / np.maximum(np.bincount(c, minlength=n), 1)
            v -= means[c]
            change = max(change, np.abs(means).max())
        if change < tol:
            break
    return v


def prep(frame):
    codes, sizes = [], []
    for col in ["rec_day", "rec_hour"]:
        c, u = pd.factorize(frame[col])
        codes.append(c)
        sizes.append(len(u))
    return codes, sizes, frame["rec_day"].values


def fit(P, y_r, ind, se=False):
    codes, sizes, clusters = P
    X = np.column_stack([demean(ind[:, k], codes, sizes) for k in range(ind.shape[1])])
    beta, *_ = np.linalg.lstsq(X, y_r, rcond=None)
    if not se:
        return beta
    resid = y_r - X @ beta
    XtX_inv = np.linalg.pinv(X.T @ X)
    meat = np.zeros((X.shape[1], X.shape[1]))
    for idx in pd.Series(np.arange(len(y_r))).groupby(clusters).indices.values():
        sc = X[idx].T @ resid[idx]
        meat += np.outer(sc, sc)
    V = XtX_inv @ meat @ XtX_inv
    return beta, np.sqrt(np.clip(np.diag(V), 0, None))


days_by_rec = {r: sorted(g.minute.dt.normalize().unique()) for r, g in m.groupby("Recorder")}


def placebo_sessions():
    rows = []
    for s in sess.itertuples():
        opts = []
        for d in days_by_rec.get(s.rec, []):
            off = d - s.start.normalize()
            if off == pd.Timedelta(0):
                continue
            c2 = s.centre + off
            if ((sess.rec == s.rec) & ((sess.centre - c2).abs() < pd.Timedelta("2h"))).any():
                continue
            opts.append(off)
        off = opts[rng.integers(len(opts))] if opts else None
        if off is not None:
            rows.append(dict(rec=s.rec, start=s.start + off, end=s.end + off))
    return pd.DataFrame(rows)


out = ["ALL-CLIPS MODEL: bird activity when people are present", "=" * 66,
       f"Sessions: {len(sess)}   minutes used: {len(m):,}", ""]
for label, mask in [("PRIMARY: daytime minutes 06:00-18:00", (m.hour >= 6) & (m.hour < 18)),
                    ("All hours", np.ones(len(m), bool))]:
    f = m[mask].reset_index(drop=True)
    P = prep(f)
    ind = indicators(f, sess)
    out.append(f"{label}   ({len(f):,} minutes, {int(ind[:, 0].sum())} 'during' minutes)")
    placebos = [indicators(f, placebo_sessions()) for _ in range(N_PERM)]
    for yname in ["bird_frac", "mean_conf"]:
        y_r = demean(f[yname].values, P[0], P[1])
        base = f[yname].mean()
        b, se = fit(P, y_r, ind, se=True)
        # placebo fits keep the real sessions in the model as controls
        null = np.array([fit(P, y_r, np.hstack([pl, ind]))[:4] for pl in placebos])
        out.append(f"  {yname}  (average level {base:.3f})")
        for k, term in enumerate(TERMS):
            p_perm = (np.sum(np.abs(null[:, k]) >= abs(b[k])) + 1) / (N_PERM + 1)
            lo, hi = b[k] - 1.96 * se[k], b[k] + 1.96 * se[k]
            out.append(f"    {term:<12} {b[k]:+.4f} ({100 * b[k] / base:+.0f}% of average)  "
                       f"95% CI {lo:+.4f} to {hi:+.4f}   placebo p={p_perm:.3f}")
        mde = 2.8 * null[:, 0].std(ddof=1)
        out.append(f"    smallest 'during' change detectable with 80% power: about {mde:.4f} "
                   f"({100 * mde / base:.0f}% of average)")
    out.append("")
print("Done.")
out.append("Negative = less bird activity than usual for that recorder, day and hour.")
txt = "\n".join(out)
print(txt)
open("all_clips_model_results.txt", "w").write(txt)
