"""
82_noise_filtered.py  —  run in Google Colab after scripts 79/81 (reads no audio)
--------------------------------------------------------------------------------
Script 81 found the soundscape about 2 dB louder in the hour after visits, but some
of the largest rises came with a jump in the LOW band (voices, footsteps, engines,
wind) - i.e. probably human or weather noise, not animals.

This script removes "noisy" clips and re-tests:
  noisy clip = low_db more than NOISE_DB above the median low_db of that window's
               BEFORE period (so each window is judged against its own baseline)
Then, with noisy clips removed from before AND after:
  A. share of after-clips flagged noisy: visits vs placebo  (how much unlabelled
     human/weather noise follows a visit)
  B. matched test (each visit vs its own normal days), raw vs filtered
  C. whole-hour test (all visits vs all placebo windows), raw vs filtered
If the bird-band rise survives filtering, it is more likely an animal response.

Colab:  upload this file, then   %run /content/82_noise_filtered.py
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


# ---------------- noise filter and tests ----------------
NOISE_DB = 3.0
meas = df.merge(pd.read_csv(os.path.join(OUT, "clip_measures.csv")), on="clip_name")
M = ["bird_db", "total_db", "high_db"]


def change(w, filt):
    L = w.end - w.start
    g = meas[(meas.Recorder == w.rec) & (meas.ts >= w.start - L) & (meas.ts < w.end + AFTER)]
    before, after = g[g.ts < w.start], g[g.ts >= w.end]
    if len(before) < 0.5 * L.total_seconds() / 3 or len(after) < 0.5 * AFTER.total_seconds() / 3:
        return None
    thr = before.low_db.median() + NOISE_DB
    noisy_share = float((after.low_db > thr).mean())
    if filt:
        before, after = before[before.low_db <= thr], after[after.low_db <= thr]
        if len(before) < 20 or len(after) < 60:
            return None
    return dict(noisy_share=noisy_share, **{m: after[m].median() - before[m].median() for m in M})


W["tod"] = W.start - W.start.dt.normalize()
W["dur"] = W.end - W.start
rows = []
for w in W.itertuples():
    for filt in (False, True):
        c = change(w, filt)
        if c:
            rows.append(dict(i=w.Index, kind=w.kind, rec=w.rec, tod=w.tod, dur=w.dur, filt=filt, **c))
D = pd.DataFrame(rows)
D.to_csv(os.path.join(OUT, "noise_filtered_windows.csv"), index=False)

out = ["NOISE-FILTERED SOUNDSCAPE TEST", "=" * 72,
       f"Noisy clip = low band more than {NOISE_DB:.0f} dB above that window's own before-period median.", ""]
raw = D[~D.filt]
v, p = raw[raw.kind == "visit"].noisy_share, raw[raw.kind == "placebo"].noisy_share
pv = stats.mannwhitneyu(v, p).pvalue if len(v) >= 3 and len(p) >= 3 else np.nan
out += ["A. Share of clips in the hour after that are noisy",
        f"  visits {100 * v.median():.1f}% (n={len(v)})   placebo {100 * p.median():.1f}% (n={len(p)})   p={pv:.3f}", ""]

for filt, name in [(False, "RAW (all clips)"), (True, "FILTERED (noisy clips removed)")]:
    d = D[D.filt == filt]
    out.append(f"{name}")
    for m in M:
        diffs = []
        for vv in d[d.kind == "visit"].itertuples():
            pl = d[(d.kind == "placebo") & (d.rec == vv.rec) & (d.tod == vv.tod) & (d.dur == vv.dur)]
            if len(pl):
                diffs.append(getattr(vv, m) - pl[m].median())
        diffs = pd.Series(diffs, dtype=float).dropna()
        pm = stats.wilcoxon(diffs).pvalue if len(diffs) >= 5 else np.nan
        a, b = d[d.kind == "visit"][m].dropna(), d[d.kind == "placebo"][m].dropna()
        pu = stats.mannwhitneyu(a, b).pvalue if len(a) >= 3 and len(b) >= 3 else np.nan
        out.append(f"  {m:<9} B. matched: {diffs.median():+5.2f} dB, louder in {int((diffs > 0).sum())}/{len(diffs)}, p={pm:.3f}"
                   f"   |  C. all: visits {a.median():+5.2f} vs placebo {b.median():+5.2f} dB, p={pu:.3f}")
    out.append("")
out += ["How to read it:",
        "  - A: a higher noisy share after visits = unlabelled human/vehicle/wind noise follows visits.",
        "  - If the bird_db rise stays after filtering, it is more likely the animals.",
        "  - If it disappears after filtering, the rise was mostly that noise."]
txt = "\n".join(out)
print(txt)
open(os.path.join(OUT, "noise_filtered_results.txt"), "w").write(txt)
print(f"\nSaved to {OUT}")
