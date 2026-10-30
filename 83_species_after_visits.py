r"""
83_species_after_visits.py  —  runs locally (PyCharm), reads the CSV only, no audio
----------------------------------------------------------------------------------
Scripts 79-82: the bird band is ~3 dB louder for an hour after visits, even with
noisy clips removed, and the clips sound like animals. Which animals?

Uses the BirdNET species already in bioacoustics_dataset.csv (conf >= 0.25) and the
SAME visits and normal-day (placebo) windows as scripts 79-82 (60-min after window).

A. Species after visits. For each species and each visit:
     change = share of clips with the species in the hour AFTER minus the BEFORE period
     diff   = visit change - median change of that visit's own normal days
              (same recorder, same clock time, same length)
   Species are ranked by how consistently they rise (sign test), with a
   false-discovery-rate correction across species, and by how many recorders agree.
   Also reported: any-bird share and species per clip.
B. Walking-only minutes. Clips inside a visit whose simulation label is walking only
   (no vehicle / chainsaw / gunshot), compared with the before period and with the
   same minutes on the visit's normal days. Does the forest change while people
   are only walking?

Usage:  .venv1\Scripts\python.exe 83_species_after_visits.py "C:\Users\BitaMassoudi\Downloads\bioacoustics_dataset.csv"
Writes: species_after_visits.csv, species_results.txt  (in toucanet_soundscape\)
"""
import ast, os, sys
import numpy as np
import pandas as pd
from scipy import stats

RECORDERS = ["Audio_Moth_1", "Audio_Moth_2", "Audio_Moth_3",
             "Audio_Moth_4", "Audio_Moth_5", "Audio_Moth_6"]      # same as 77-82
MAX_PLACEBO_PER_VISIT, SEED = 4, 0
AFTER = pd.Timedelta("60min")
CONF, MIN_VISITS = 0.25, 8
CSV = sys.argv[1]
OUT = os.path.join(os.getcwd(), "toucanet_soundscape")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(SEED)

# ---------------- labels and sessions ----------------
print("Reading labels...")
cols = {"clip_name", "Recorder", "Timestamp", "Sim Type", "Sim Relative Time", "species", "confidence"}
df = pd.read_csv(CSV, usecols=lambda c: c in cols, low_memory=False).drop_duplicates("clip_name")
df["ts"] = pd.to_datetime(df["Timestamp"].astype(str), format="%Y%m%d_%H%M%S", errors="coerce")
df = df.dropna(subset=["ts"])
sim = df[df["Sim Type"].astype(str).str.strip() != "[]"].copy()
sim["rel"] = pd.to_numeric(sim["Sim Relative Time"].astype(str).str.strip("[]").str.split(",").str[0],
                           errors="coerce").fillna(0)
sim["event"] = sim.ts - pd.to_timedelta(sim.rel, unit="s")
sess = []
for r, g in sim.groupby("Recorder"):
    e = g.sort_values("event")
    for _, x in e.groupby((e.event.diff() > pd.Timedelta("20min")).cumsum()):
        sess.append(dict(rec=r, start=x.ts.min(), end=x.ts.max() + pd.Timedelta("3s"), centre=x.event.median()))
sess = pd.DataFrame(sess)


def covered(g, a, b):
    return ((g.ts >= a) & (g.ts < b)).sum() / ((b - a).total_seconds() / 3)


windows = []   # (kind, rec, start, end)
for rec in RECORDERS:
    g = df[df.Recorder == rec]
    days = sorted(g.ts.dt.normalize().unique())
    for x in sess[sess.rec == rec].itertuples():
        L = x.end - x.start
        ok = lambda s, e: all(covered(g, a, b) >= 0.8 for a, b in [(s - L, s), (s, e), (e, e + AFTER)])
        clash = lambda s, e: ((sess.rec == rec) & (sess.end > s - L) & (sess.start < e + AFTER)).any()
        if not ok(x.start, x.end):
            continue
        if ((sess.rec == rec) & (sess.start >= x.end) & (sess.start < x.end + AFTER)).any():
            continue   # another visit within the hour after
        windows.append(("visit", rec, x.start, x.end))
        opts = []
        for d in days:
            off = d - x.start.normalize()
            if off == pd.Timedelta(0):
                continue
            c2 = x.centre + off
            if ((sess.rec == rec) & ((sess.centre - c2).abs() < pd.Timedelta("2h"))).any():
                continue
            if ok(x.start + off, x.end + off) and not clash(x.start + off, x.end + off):
                opts.append(off)
        for off in rng.permutation(opts)[:MAX_PLACEBO_PER_VISIT] if opts else []:
            windows.append(("placebo", rec, x.start + off, x.end + off))
W = pd.DataFrame(windows, columns=["kind", "rec", "start", "end"])
print(W.groupby(["rec", "kind"]).size().to_string())


print(f"  visits {(W.kind == 'visit').sum()}, normal-day windows {(W.kind == 'placebo').sum()}  (scripts 79-82 had 25 and 28)")


# ---------------- species per clip ----------------
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
    for s_, c in zip(sp, cf):
        try:
            if float(c) >= CONF and str(s_).strip():
                out.add(str(s_).split("_")[-1])      # common name
        except (TypeError, ValueError):
            pass
    return out


print("Parsing BirdNET species (a minute or two)...")
df = df[df.Recorder.isin(RECORDERS)].copy()
df["birds"] = df.apply(birds, axis=1)
df["n_sp"] = df.birds.str.len()
df["any_bird"] = (df.n_sp > 0).astype(float)
counts = pd.Series([b for s_ in df.birds for b in s_]).value_counts()
SPECIES = list(counts[counts >= 200].index)            # skip very rare species
print(f"  {len(SPECIES)} species with at least 200 detections")
by_rec = {r: g.sort_values("ts") for r, g in df.groupby("Recorder")}


def shares(clips):
    if len(clips) == 0:
        return None
    d = {"any_bird": clips.any_bird.mean(), "species_per_clip": clips.n_sp.mean()}
    for sp in SPECIES:
        d[sp] = clips.birds.apply(lambda s_: sp in s_).mean()
    return d


def window_change(w):
    g = by_rec[w.rec]
    L = w.end - w.start
    before = g[(g.ts >= w.start - L) & (g.ts < w.start)]
    after = g[(g.ts >= w.end) & (g.ts < w.end + AFTER)]
    if len(before) < 0.5 * L.total_seconds() / 3 or len(after) < 0.5 * AFTER.total_seconds() / 3:
        return None
    b, a = shares(before), shares(after)
    return {k: a[k] - b[k] for k in a}


# ---------------- A. species after visits, matched ----------------
W["tod"] = W.start - W.start.dt.normalize()
W["dur"] = W.end - W.start
print("Comparing each visit with its own normal days...")
changes = {i: window_change(w) for i, w in W.iterrows()}
rows = []
for i, v in W[W.kind == "visit"].iterrows():
    cv = changes[i]
    pl = W[(W.kind == "placebo") & (W.rec == v.rec) & (W.tod == v.tod) & (W.dur == v.dur)]
    cps = [changes[j] for j in pl.index if changes[j] is not None]
    if cv is None or not cps:
        continue
    for k in cv:
        base = np.median([c[k] for c in cps])
        present = cv[k] != 0 or any(c[k] != 0 for c in cps)
        rows.append(dict(visit=i, rec=v.rec, start=v.start, measure=k, diff=cv[k] - base, present=present))
D = pd.DataFrame(rows)
D.to_csv(os.path.join(OUT, "species_after_visits.csv"), index=False)
n_visits = D.visit.nunique()

res = []
for k, g in D[D.present].groupby("measure"):
    d = g["diff"]
    up, down = int((d > 0).sum()), int((d < 0).sum())
    if up + down < (5 if k in ("any_bird", "species_per_clip") else MIN_VISITS):
        continue
    p = stats.binomtest(up, up + down, 0.5).pvalue
    recs_up = int((g.groupby("rec")["diff"].median() > 0).sum())
    res.append(dict(measure=k, n=up + down, up=up, down=down, median_diff=d.median(), p=p,
                    recs_up=recs_up, recs=g.rec.nunique()))
R = pd.DataFrame(res)
sp = R[~R.measure.isin(["any_bird", "species_per_clip"])].copy()
if len(sp):
    sp = sp.sort_values("p")
    m = len(sp)
    q = (sp.p * m / np.arange(1, m + 1)).iloc[::-1].cummin().iloc[::-1].clip(upper=1)
    sp["q"] = q.values

out = ["WHICH ANIMALS CHANGE AFTER VISITS?  (BirdNET species, hour after vs before, each visit vs its own normal days)",
       "=" * 100,
       f"Visits with matched normal days: {n_visits}.  diff = change in share of clips with the species, visit minus normal days.", ""]
for k, label in [("any_bird", "Any bird"), ("species_per_clip", "Species per clip")]:
    r = R[R.measure == k]
    if len(r):
        r = r.iloc[0]
        out.append(f"{label:<18} median diff {r.median_diff:+.3f}   up in {r.up}/{r.n} visits   p={r.p:.3f}")
out += ["", f"Species that rise most consistently (heard at >= {MIN_VISITS} visits; q = false-discovery-rate corrected):",
        f"  {'species':<32} {'up/visits':>9} {'median diff':>12} {'p':>7} {'q':>6}  recorders up"]
for r in sp.head(15).itertuples():
    out.append(f"  {r.measure[:32]:<32} {f'{r.up}/{r.n}':>9} {r.median_diff:>+12.3f} {r.p:>7.3f} {r.q:>6.2f}  {r.recs_up}/{r.recs}")
dn = sp.sort_values(["down", "p"], ascending=[False, True]).query("down > up").head(5)
if len(dn):
    out += ["", "Species that tend to DROP after visits:"]
    for r in dn.itertuples():
        out.append(f"  {r.measure[:32]:<32} {f'{r.down}/{r.n} down':>11} {r.median_diff:>+12.3f}  p={r.p:.3f}")

# ---------------- B. walking-only minutes during visits ----------------
LOUD = ("vehicle", "chainsaw", "gun")
simw = sim.copy()
simw["walk_only"] = ~simw["Sim Type"].astype(str).str.lower().str.contains("|".join(LOUD))
wrows = []
for i, v in W[W.kind == "visit"].iterrows():
    g = by_rec[v.rec]
    L = v.end - v.start
    before = g[(g.ts >= v.start - L) & (g.ts < v.start)]
    wt = simw[(simw.Recorder == v.rec) & (simw.ts >= v.start) & (simw.ts < v.end) & simw.walk_only].ts
    if len(wt) < 20 or len(before) < 50:
        continue
    walk = g[g.ts.isin(wt)]
    pl = W[(W.kind == "placebo") & (W.rec == v.rec) & (W.tod == v.tod) & (W.dur == v.dur)]
    pdiffs = []
    for _, p_ in pl.iterrows():
        off = p_.start - v.start
        pb = g[(g.ts >= p_.start - L) & (g.ts < p_.start)]
        pw = g[g.ts.isin(wt + off)]
        if len(pw) >= 20 and len(pb) >= 50:
            pdiffs.append(pw.any_bird.mean() - pb.any_bird.mean())
    if pdiffs:
        wrows.append(dict(rec=v.rec, n_walk_clips=len(walk),
                          diff=(walk.any_bird.mean() - before.any_bird.mean()) - np.median(pdiffs)))
WK = pd.DataFrame(wrows)
out += ["", "B. Walking-only minutes DURING visits (any-bird share vs before, minus the same on normal days)"]
if len(WK) >= 3:
    up, n = int((WK["diff"] > 0).sum()), int((WK["diff"] != 0).sum())
    p = stats.binomtest(up, n, 0.5).pvalue if n else np.nan
    out.append(f"  visits with walking-only stretches: {len(WK)}   median diff {WK['diff'].median():+.3f}   "
               f"higher in {up}/{n}   p={p:.3f}")
else:
    out.append(f"  only {len(WK)} visits with enough walking-only clips; too few to test")
out += ["", "How to read it:",
        "  - A species with a high 'up/visits', small q and several recorders up is a candidate responder.",
        "  - Alarm-calling species rising suggests a disturbance reaction; many species rising suggests a rebound.",
        "  - B tells whether birds change while people are only walking (the quiet-poacher case)."]
txt = "\n".join(out)
print(txt)
open(os.path.join(OUT, "species_results.txt"), "w", encoding="utf-8").write(txt)
print(f"\nSaved to {OUT}")
