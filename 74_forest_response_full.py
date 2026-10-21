"""
74_forest_response_full.py
--------------------------
Redo of scripts 70/71 on bioacoustics_dataset.csv: ALL clips (not only the
ones where BirdNET heard a bird) and its simulation labels, which disagree
with cleaned_df at recorders 1-5 (see script 73).

Because every 3-second clip is present, "bird activity" is now a true share:
  bird_frac = fraction of clips in which BirdNET heard any bird (conf >= 0.25,
              the same cut-off cleaned_df used). 1 - bird_frac = silence.
  species   = number of bird species heard (conf >= 0.25)

For each simulation session (rebuilt from bioacoustics labels):
  DURING = first to last labelled clip; BEFORE and AFTER = equal-length
  windows right before and right after. All three must be at least 80%
  recorded (no gaps from the recorder being off).
Placebo = the same clock windows at the same recorder on days with no
simulation within 2 h. A reaction to people should change more in real
sessions than in placebo windows.

Two pre-set groups are reported: ALL sessions, and DAYTIME sessions
(whole before-to-after span between 06:00 and 17:30).

Usage:
  python 74_forest_response_full.py "C:\\...\\bioacoustics_dataset.csv"
Writes: forest_full_sessions.csv, forest_full_results.txt
"""
import ast
import sys
import numpy as np
import pandas as pd
from scipy import stats

PATH = sys.argv[1]
CONF = 0.25
want = {"clip_name", "Recorder", "Timestamp", "species", "confidence", "Sim Type", "Sim Relative Time"}
df = pd.read_csv(PATH, usecols=lambda c: c in want, low_memory=False)
df["clip_name"] = df["clip_name"].astype(str)
df = df.drop_duplicates("clip_name")
df["ts"] = pd.to_datetime(df["Timestamp"].astype(str), format="%Y%m%d_%H%M%S", errors="coerce")
df = df.dropna(subset=["ts"]).sort_values(["Recorder", "ts"]).reset_index(drop=True)


def as_list(v):
    if isinstance(v, str) and v.strip().startswith("["):
        try:
            return list(ast.literal_eval(v))
        except (ValueError, SyntaxError):
            return []
    if v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() in ("", "nan", "None"):
        return []
    return [v]


def birds(row):
    sp, cf = as_list(row["species"]), as_list(row["confidence"])
    if len(cf) == 1 and len(sp) > 1:
        cf = cf * len(sp)
    out = set()
    for s, c in zip(sp, cf):
        try:
            if float(c) >= CONF and str(s).strip():
                out.add(str(s))
        except (TypeError, ValueError):
            pass
    return out


print("Parsing BirdNET detections...")
df["birdset"] = df.apply(birds, axis=1)
df["bird"] = df.birdset.str.len() > 0
print(f"  {len(df):,} clips, {df.bird.mean():.1%} with a bird detected")

# ---- sessions from bioacoustics labels ------------------------------------
sim = df[df["Sim Type"].astype(str).str.strip() != "[]"].copy()
sim["rel"] = pd.to_numeric(sim["Sim Relative Time"].astype(str).str.strip("[]").str.split(",").str[0],
                           errors="coerce").fillna(0)
sim["event"] = sim.ts - pd.to_timedelta(sim.rel, unit="s")
sess = []
for r, g in sim.groupby("Recorder"):
    e = g.sort_values("event")
    grp = (e.event.diff() > pd.Timedelta("20min")).cumsum()
    for _, x in e.groupby(grp):
        sess.append(dict(rec=r, start=x.ts.min(), end=x.ts.max() + pd.Timedelta("3s"),
                         centre=x.event.median()))
sess = pd.DataFrame(sess)
print(f"  {len(sess)} simulation sessions")

by_rec = {r: g for r, g in df.groupby("Recorder")}


def win(g, a, b):
    w = g[(g.ts >= a) & (g.ts < b)]
    expected = (b - a).total_seconds() / 3
    sp = set().union(*w.birdset) if len(w) else set()
    return dict(cov=len(w) / expected, bird_frac=w.bird.mean() if len(w) else np.nan, species=len(sp))


def triple(g, s, e):
    L = e - s
    b, d, a = win(g, s - L, s), win(g, s, e), win(g, e, e + L)
    if min(b["cov"], d["cov"], a["cov"]) < 0.8:
        return None
    return {f"{k}_{m}": v[m] for k, v in [("before", b), ("during", d), ("after", a)] for m in ["bird_frac", "species"]}


def daytime(s, e):
    L = e - s
    m0, m1 = (s - L), (e + L)
    return m0.date() == m1.date() and m0.hour * 60 + m0.minute >= 360 and m1.hour * 60 + m1.minute <= 1050


rows, plac = [], []
days = sorted(df.ts.dt.normalize().unique())
for x in sess.itertuples():
    g = by_rec[x.rec]
    t = triple(g, x.start, x.end)
    if t is None:
        continue
    rows.append(dict(rec=x.rec, start=x.start, end=x.end, day=daytime(x.start, x.end), **t))
    for d in days:
        off = d - x.start.normalize()
        if off == pd.Timedelta(0):
            continue
        s2, e2 = x.start + off, x.end + off
        near = sess[(sess.rec == x.rec) & ((sess.centre - (s2 + (e2 - s2) / 2)).abs() < pd.Timedelta("2h"))]
        if len(near):
            continue
        t2 = triple(g, s2, e2)
        if t2 is not None:
            plac.append(dict(rec=x.rec, day=daytime(s2, e2), **t2))

R, P = pd.DataFrame(rows), pd.DataFrame(plac)
R.to_csv("forest_full_sessions.csv", index=False)
out = [f"FOREST RESPONSE, FULL DATA (bioacoustics labels)", "=" * 64,
       f"Sessions with complete recordings: {len(R)} (daytime {int(R.day.sum())})   "
       f"placebo windows: {len(P)} (daytime {int(P.day.sum()) if len(P) else 0})", ""]
for label, rs, ps in [("ALL SESSIONS", R, P), ("DAYTIME SESSIONS", R[R.day], P[P.day] if len(P) else P)]:
    out.append(label)
    for m, name in [("bird_frac", "Share of clips with birds"), ("species", "Species heard")]:
        for when in ["during", "after"]:
            dr = (rs[f"{when}_{m}"] - rs[f"before_{m}"]).dropna()
            dp = (ps[f"{when}_{m}"] - ps[f"before_{m}"]).dropna() if len(ps) else pd.Series(dtype=float)
            p = stats.mannwhitneyu(dr, dp).pvalue if len(dr) >= 3 and len(dp) >= 3 else np.nan
            fmt = "{:+.3f}" if m == "bird_frac" else "{:+.1f}"
            out.append(f"  {name:<26} {when:>6}-before: sessions median {fmt.format(dr.median())} "
                       f"(lower in {int((dr < 0).sum())}/{len(dr)})  placebo median "
                       f"{fmt.format(dp.median()) if len(dp) else 'n/a'} (n={len(dp)})  p={p:.3f}")
    out.append("")
out.append("Per recorder, median change in share of clips with birds (during - before):")
for r, g in R.groupby("rec"):
    out.append(f"  {r}: {len(g)} sessions, {(g.during_bird_frac - g.before_bird_frac).median():+.3f}")
txt = "\n".join(out)
print(txt)
open("forest_full_results.txt", "w").write(txt)
