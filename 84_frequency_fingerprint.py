"""
84_frequency_fingerprint.py  —  run in Google Colab (reads a sample of audio, ~20-30 min)
---------------------------------------------------------------------------------------
The bird band (2-8 kHz) is ~3 dB louder for an hour after visits, but BirdNET finds no
more birds or species (script 83). What IS the extra sound? Its frequency fingerprint
tells us:
  - insects / frogs   -> sharp rises at a few fixed pitches (narrow peaks)
  - more bird calling -> a broad rise spread across 2-8 kHz
  - river / water     -> a smooth rise, strongest at lower frequencies

For the matched visits (same as scripts 81-82) and their own normal days, it samples
clips from BEFORE (up to 30) and the HOUR AFTER (up to 60), skipping noisy clips
(low band > 3 dB above the window's before level, as in script 82), and measures the
average spectrum in 500 Hz steps from 0 to 16 kHz.
  change = after minus before (dB), per 500 Hz step
  diff   = visit change minus the median change of its own normal days
Across visits: median diff per step, how many visits went up, and a sign-test p.

Saves fingerprint.csv, fingerprint.png and fingerprint_results.txt to toucanet_soundscape.
Colab:  upload this file, then   %run /content/84_frequency_fingerprint.py
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


# ---------------- pick matched visits and sample clips ----------------
from scipy import signal
from concurrent.futures import ThreadPoolExecutor
N_BEFORE, N_AFTER, STEP, FMAX, NOISE_DB = 30, 60, 500, 16000, 3.0
meas = df.merge(pd.read_csv(os.path.join(OUT, "clip_measures.csv")), on="clip_name")
W["tod"] = W.start - W.start.dt.normalize()
W["dur"] = W.end - W.start
pick_rng = np.random.default_rng(1)
use, samples = [], {}
for i, v in W[W.kind == "visit"].iterrows():
    pl = W[(W.kind == "placebo") & (W.rec == v.rec) & (W.tod == v.tod) & (W.dur == v.dur)]
    if len(pl) == 0:
        continue
    for j, w in [(i, v)] + list(pl.iterrows()):
        L = w.end - w.start
        g = meas[meas.Recorder == w.rec]
        before = g[(g.ts >= w.start - L) & (g.ts < w.start)]
        after = g[(g.ts >= w.end) & (g.ts < w.end + AFTER)]
        if len(before) < 20 or len(after) < 60:
            continue
        thr = before.low_db.median() + NOISE_DB
        before, after = before[before.low_db <= thr], after[after.low_db <= thr]
        samples[j] = {
            "before": list(before.clip_name.sample(min(N_BEFORE, len(before)), random_state=int(pick_rng.integers(1e9)))),
            "after": list(after.clip_name.sample(min(N_AFTER, len(after)), random_state=int(pick_rng.integers(1e9))))}
    use.append((i, list(pl.index)))
need = {n for s_ in samples.values() for part in s_.values() for n in part}
print(f"Matched visits: {len(use)}   clips to read: {len(need):,}")

print("Finding the clips (a few minutes)...")
paths = {}
for rec in sorted({W.loc[i, "rec"] for i in samples}):
    for root, _, files in os.walk(os.path.join(AUDIO, rec.replace("_", " "))):
        for f in files:
            if f in need:
                paths[f] = os.path.join(root, f)
print(f"  found {len(paths):,} of {len(need):,}")

edges = np.arange(0, FMAX + STEP, STEP)


def spectrum(name):
    try:
        y, sr = load(paths[name])
        if y.ndim > 1:
            y = y.mean(axis=1)
        f, P = signal.welch(y, fs=sr, nperseg=2048)
        return np.array([P[(f >= a) & (f < b)].mean() if ((f >= a) & (f < b)).any() else np.nan
                         for a, b in zip(edges[:-1], edges[1:])])
    except Exception:
        return None


print("Measuring spectra...")
names = [n for n in need if n in paths]
spec = {}
with ThreadPoolExecutor(THREADS) as ex:
    for k, (n, s_) in enumerate(zip(names, ex.map(spectrum, names)), 1):
        if s_ is not None:
            spec[n] = s_
        if k % 500 == 0:
            print(f"  {k:,}/{len(names):,}")


def part_db(ns):
    arr = [spec[n] for n in ns if n in spec]
    return 10 * np.log10(np.nanmean(arr, axis=0) + 1e-15) if len(arr) >= 10 else None


change = {}
for j, s_ in samples.items():
    b, a = part_db(s_["before"]), part_db(s_["after"])
    if b is not None and a is not None:
        change[j] = a - b
diffs = []
for i, pls in use:
    pc = [change[j] for j in pls if j in change]
    if i in change and pc:
        diffs.append(change[i] - np.median(pc, axis=0))
diffs = np.array(diffs)

# ---------------- report ----------------
lo, hi = edges[:-1] / 1000, edges[1:] / 1000
med = np.nanmedian(diffs, axis=0)
up = (diffs > 0).sum(axis=0)
n = (~np.isnan(diffs)).sum(axis=0)
p = np.array([stats.binomtest(int(u), int(k), 0.5).pvalue if k else np.nan for u, k in zip(up, n)])
F = pd.DataFrame(dict(khz_from=lo, khz_to=hi, median_diff_db=med, visits_up=up, visits=n, p=p))
F.to_csv(os.path.join(OUT, "fingerprint.csv"), index=False)

out = ["FREQUENCY FINGERPRINT OF THE AFTER-VISIT RISE  (noisy clips removed)", "=" * 72,
       f"Visits compared with their own normal days: {len(diffs)}", "",
       "  kHz        diff dB   up/visits   p       (each # = 0.5 dB louder after visits)"]
for r in F.itertuples():
    bar = "#" * int(max(0, min(20, round(r.median_diff_db / 0.5)))) if not np.isnan(r.median_diff_db) else ""
    neg = "-" * int(max(0, min(20, round(-r.median_diff_db / 0.5)))) if not np.isnan(r.median_diff_db) else ""
    out.append(f"  {r.khz_from:4.1f}-{r.khz_to:<4.1f}  {r.median_diff_db:+6.2f}   {r.visits_up:>3}/{r.visits:<3}    {r.p:5.3f}   {bar or neg}")
top = F.nlargest(3, "median_diff_db")
spread = (F[(F.khz_from >= 2) & (F.khz_to <= 8)].median_diff_db > 1).mean()
out += ["", "Largest rises: " + ", ".join(f"{a:.1f}-{b:.1f} kHz ({d:+.1f} dB)" for a, b, d in
                                           zip(top.khz_from, top.khz_to, top.median_diff_db)),
        f"Share of 2-8 kHz steps rising by more than 1 dB: {100 * spread:.0f}%", "",
        "How to read it:",
        "  - A few narrow steps with big rises and little elsewhere  -> insects or frogs (fixed-pitch choruses)",
        "  - Most 2-8 kHz steps rising together                       -> more bird calling",
        "  - Rise largest at low kHz and fading upwards               -> water / river or wind"]
txt = "\n".join(out)
print(txt)
open(os.path.join(OUT, "fingerprint_results.txt"), "w").write(txt)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4))
    mid = (lo + hi) / 2
    for d in diffs:
        ax.plot(mid, d, color="#BBBBBB", lw=0.8)
    ax.plot(mid, med, color="#C8641C", lw=2.5, label="median across visits")
    ax.axhline(0, color="#333333", lw=0.8)
    ax.axvspan(2, 8, color="#1F3A2E", alpha=0.07, label="bird band 2-8 kHz")
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Louder after visits vs normal days (dB)")
    ax.set_title("Which frequencies get louder in the hour after a visit")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fingerprint.png"), dpi=150)
    print("Saved fingerprint.png")
except Exception as e:
    print(f"(plot skipped: {e})")
print(f"\nSaved to {OUT}")
