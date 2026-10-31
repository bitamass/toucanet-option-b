"""
85_route_approach.py  —  run in Google Colab after scripts 77/79 (reads no new audio)
------------------------------------------------------------------------------------
Main question: does the forest react BEFORE people arrive?
The team walks a route past the recorders, about 10-15 minutes apart. So while the
team is at the previous recorder, the next one has not been reached yet. For each
visit at recorder B:
  approach period = from the start of the team's previous session (at another
                    recorder, same day, within the hour before) until B's visit starts
                    (limited to the part of B's BEFORE window that was measured)
We compare B's soundscape and BirdNET detections in that approach period with the
SAME clock minutes at B on its own normal days (same recorder, clock time, length).
Absolute levels, matched by recorder and time, so the daily rhythm cancels out.
  diff > 0 : louder / more birds while people are approaching than on normal days
Also reported: the part of B's before window BEFORE the approach (people further away).

Colab:  upload this file, then
   from google.colab import drive; drive.mount("/content/drive")
   %run /content/85_route_approach.py
"""
import os, ast, time
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from scipy import signal, stats
try:
    import soundfile as sf
    def load(p):
        return sf.read(p, dtype="float32", always_2d=False)
except ImportError:
    from scipy.io import wavfile
    def load(p):
        sr, y = wavfile.read(p)
        return y.astype("float32") / (32768.0 if y.dtype == np.int16 else 1.0), sr

# ---------------- settings ----------------
RECORDERS = ["Audio_Moth_1", "Audio_Moth_2", "Audio_Moth_3",
             "Audio_Moth_4", "Audio_Moth_5", "Audio_Moth_6"]
BASE = "/content/drive/MyDrive/Audio Moth Data"
CSV = os.path.join(BASE, "bioacoustics_dataset.csv")
AUDIO = os.path.join(BASE, "Segmented Foldered")
OUT = "/content/drive/MyDrive/toucanet_soundscape"   # your own Drive
MAX_PLACEBO_PER_VISIT, THREADS, SEED = 4, 32, 0
AFTER = pd.Timedelta("60min")   # how far past each visit to look
BIN = 10                         # minutes per time-course step

import sys
if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
    # Terminal mode:  python 77_soundscape_colab.py "G:\My Drive\Audio Moth Data"
    BASE = sys.argv[1]
    CSV = os.path.join(BASE, "bioacoustics_dataset.csv")
    AUDIO = os.path.join(BASE, "Segmented Foldered")
    OUT = os.path.join(os.getcwd(), "toucanet_soundscape")
    THREADS = 8
else:
    from google.colab import drive
    drive.mount("/content/drive")
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


# ---------------- approach periods ----------------
import ast
CONF = 0.25
meas = df.merge(pd.read_csv(os.path.join(OUT, "clip_measures.csv")), on="clip_name")


def nbirds(row):
    sp, cf = row["species"], row["confidence"]
    try:
        sp = ast.literal_eval(sp) if isinstance(sp, str) and sp.startswith("[") else ([] if pd.isna(sp) else [sp])
        cf = ast.literal_eval(cf) if isinstance(cf, str) and cf.startswith("[") else ([] if pd.isna(cf) else [cf])
    except (ValueError, SyntaxError):
        return 0
    if len(cf) == 1 and len(sp) > 1:
        cf = cf * len(sp)
    return sum(1 for c in cf if float(c) >= CONF)


meas["any_bird"] = (meas.apply(nbirds, axis=1) > 0).astype(float)
M = ["bird_db", "total_db", "high_db", "low_db", "any_bird"]
W["tod"] = W.start - W.start.dt.normalize()
W["dur"] = W.end - W.start


def level(rec, a, b):
    x = meas[(meas.Recorder == rec) & (meas.ts >= a) & (meas.ts < b)]
    return x[M].median() if len(x) >= 20 else None


rows = []
for i, v in W[W.kind == "visit"].iterrows():
    L = v.end - v.start
    prev = sess[(sess.rec != v.rec) & (sess.start < v.start) & (sess.start >= v.start - pd.Timedelta("60min"))]
    if len(prev) == 0:
        continue
    a = max(prev.start.min(), v.start - L)          # approach start, inside the measured before window
    if v.start - a < pd.Timedelta("2min"):
        continue
    pl = W[(W.kind == "placebo") & (W.rec == v.rec) & (W.tod == v.tod) & (W.dur == v.dur)]
    for part, (s0, s1) in [("approach", (a, v.start)), ("earlier", (v.start - L, a))]:
        if s1 - s0 < pd.Timedelta("2min"):
            continue
        lv = level(v.rec, s0, s1)
        lp = [level(v.rec, s0 + (p.start - v.start), s1 + (p.start - v.start)) for _, p in pl.iterrows()]
        lp = [x for x in lp if x is not None]
        if lv is None or not lp:
            continue
        base = pd.concat(lp, axis=1).median(axis=1)
        rows.append(dict(rec=v.rec, start=v.start, part=part, minutes=(s1 - s0).total_seconds() / 60,
                         **{m: lv[m] - base[m] for m in M}))
D = pd.DataFrame(rows, columns=["rec", "start", "part", "minutes"] + M)
D.to_csv(os.path.join(OUT, "route_approach.csv"), index=False)

out = ["DOES THE FOREST CHANGE WHILE PEOPLE ARE APPROACHING?", "=" * 72,
       "Recorder B while the team is at the previous recorder (before reaching B), vs the same minutes on B's normal days.",
       "dB measures in dB; any_bird = share of clips with a BirdNET bird. Positive = higher than normal days.", ""]
for part, label in [("approach", "APPROACH (team at the previous recorder, on its way)"),
                    ("earlier", "EARLIER (before the approach, team further away)")]:
    d = D[D.part == part]
    out.append(f"{label}:  {len(d)} visits, median {d.minutes.median():.0f} min each" if len(d) else f"{label}: none")
    for m in M:
        x = d[m].dropna()
        up, dn = int((x > 0).sum()), int((x < 0).sum())
        p = stats.binomtest(up, up + dn, 0.5).pvalue if up + dn >= 3 else np.nan
        unit = "" if m == "any_bird" else " dB"
        out.append(f"  {m:<9} median {x.median():+.3f}{unit}   higher in {up}/{up + dn}   p={p:.3f}")
    out.append("")
out += ["How to read it:",
        "  - APPROACH clearly different (small p, most visits one way) and EARLIER not -> the forest reacts before people arrive.",
        "  - Both similar -> no early-warning signal in these measures.",
        "  - Watch low_db: a rise there means the people themselves are already audible (not a forest response)."]
txt = "\n".join(out)
print(txt)
open(os.path.join(OUT, "route_approach_results.txt"), "w").write(txt)
print(f"\nSaved to {OUT}")
