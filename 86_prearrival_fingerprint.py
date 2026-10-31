"""
86_prearrival_fingerprint.py  —  run in Google Colab (reads a sample of audio, ~40-60 min)
----------------------------------------------------------------------------------------
Main question: does anything change in the soundscape BEFORE people arrive?
Averages over whole windows can hide brief changes (alarm calls, insects falling
silent), so this looks minute by minute and pitch by pitch around arrival.

For the matched visits (same as scripts 81-84) and their own normal days, it reads a
sample of clips in 5-minute steps from 30 minutes before the visit starts to 10
minutes after, and measures the spectrum in pitch groups:
  0-1 kHz (river, wind, voices)  1-2  2-4  4-6  6-8 (birds)  8-11  11-14 (insects)  14-16
For each step: level on the visit day minus the same clock minutes on its normal days,
then minus that same difference in the first step (-30 to -25 min).
  -> how much each pitch group has changed, compared with half an hour earlier,
     beyond what normally happens at that time.
A change that starts BEFORE minute 0 is the early-warning signal we are looking for.

Colab:  upload this file, then   %run /content/86_prearrival_fingerprint.py
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
cols = {"clip_name", "Recorder", "Timestamp", "Sim Type", "Sim Relative Time"}
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


# ---------------- sample clips around arrival ----------------
from scipy import signal
from concurrent.futures import ThreadPoolExecutor
STEPS = list(range(-30, 10, 5))          # step starts, minutes relative to visit start
PER_STEP = 20
GROUPS = [(0, 1), (1, 2), (2, 4), (4, 6), (6, 8), (8, 11), (11, 14), (14, 16)]
W["tod"] = W.start - W.start.dt.normalize()
W["dur"] = W.end - W.start
rng2 = np.random.default_rng(2)
plan, need = {}, set()                    # (window index, step) -> clip names
use = []
for i, v in W[W.kind == "visit"].iterrows():
    pl = W[(W.kind == "placebo") & (W.rec == v.rec) & (W.tod == v.tod) & (W.dur == v.dur)]
    if len(pl) == 0:
        continue
    use.append((i, list(pl.index)))
    for j, w in [(i, v)] + list(pl.iterrows()):
        g = df[df.Recorder == w.rec]
        for st in STEPS:
            a = w.start + pd.Timedelta(minutes=st)
            x = g[(g.ts >= a) & (g.ts < a + pd.Timedelta("5min"))].clip_name
            if len(x) >= 10:
                pick = list(x.sample(min(PER_STEP, len(x)), random_state=int(rng2.integers(1e9))))
                plan[(j, st)] = pick
                need |= set(pick)
print(f"Matched visits: {len(use)}   clips to read: {len(need):,}")

print("Finding the clips (a few minutes)...")
paths = {}
recs = sorted({W.loc[i, "rec"] for i, _ in use})
for rec in recs:
    for root, _, files in os.walk(os.path.join(AUDIO, rec.replace("_", " "))):
        for f in files:
            if f in need:
                paths[f] = os.path.join(root, f)
print(f"  found {len(paths):,} of {len(need):,}")


def groups(name):
    try:
        y, sr = load(paths[name])
        if y.ndim > 1:
            y = y.mean(axis=1)
        f, P = signal.welch(y, fs=sr, nperseg=2048)
        return np.array([P[(f >= a * 1000) & (f < b * 1000)].mean() for a, b in GROUPS])
    except Exception:
        return None


print("Measuring...")
names = [n for n in need if n in paths]
spec = {}
with ThreadPoolExecutor(THREADS) as ex:
    for k, (n, s_) in enumerate(zip(names, ex.map(groups, names)), 1):
        if s_ is not None:
            spec[n] = s_
        if k % 1000 == 0:
            print(f"  {k:,}/{len(names):,}")


def lvl(j, st):
    arr = [spec[n] for n in plan.get((j, st), []) if n in spec]
    return 10 * np.log10(np.mean(arr, axis=0) + 1e-15) if len(arr) >= 8 else None


res = {st: [] for st in STEPS}
for i, pls in use:
    d = {}
    for st in STEPS:
        lv = lvl(i, st)
        lp = [x for x in (lvl(j, st) for j in pls) if x is not None]
        if lv is not None and lp:
            d[st] = lv - np.median(lp, axis=0)
    if STEPS[0] in d:
        for st in d:
            res[st].append(d[st] - d[STEPS[0]])

names_g = [f"{a}-{b}k" for a, b in GROUPS]
rows = []
out = ["SOUNDSCAPE AROUND ARRIVAL, BY PITCH  (change since -30 min, beyond normal days; median dB)", "=" * 92,
       "minute 0 = people reach the recorder.  * = changed the same way in at least 75% of visits", "",
       "  minutes     " + "".join(f"{g:>9}" for g in names_g) + "   visits"]
for st in STEPS:
    arr = np.array(res[st])
    if len(arr) < 5:
        continue
    med = np.median(arr, axis=0)
    frac_up = (arr > 0).mean(axis=0)
    cells = "".join(f"{m:+8.2f}{'*' if (f >= 0.75 or f <= 0.25) and st != STEPS[0] else ' '}" for m, f in zip(med, frac_up))
    tag = "before" if st < 0 else "arrive"
    out.append(f"  {st:+3d} to {st + 5:+3d} {cells}   {len(arr):>3}  {tag}")
    for g, m, f in zip(names_g, med, frac_up):
        rows.append(dict(step_start=st, group=g, median_change_db=m, share_up=f, visits=len(arr)))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "prearrival_fingerprint.csv"), index=False)
out += ["", "How to read it:",
        "  - Look along the rows from -30 to 0. Stars or growing numbers BEFORE 0 in one pitch group = an early-warning change.",
        "  - A drop in 11-14k before arrival would mean insects falling silent as people approach.",
        "  - A rise in 0-1k / 1-2k only from 0 onwards is the people themselves arriving (expected)."]
txt = "\n".join(out)
print(txt)
open(os.path.join(OUT, "prearrival_fingerprint_results.txt"), "w").write(txt)
print(f"\nSaved to {OUT}")
