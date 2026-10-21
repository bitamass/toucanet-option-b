"""
77_soundscape_colab.py  —  run in Google Colab, not on the laptop
-----------------------------------------------------------------
Soundscape check on the FULL audio (every 3-s clip, not only clips with birds).
Reads clips straight from the shared "Audio Moth Data / Segmented Foldered"
folder, measures only the clips inside the analysis windows, and saves small
CSVs back to your Drive. Read-only on the team folders.

Windows (same rules as scripts 74-76, labels from bioacoustics_dataset.csv):
  DURING = first to last labelled clip of a visit; BEFORE and AFTER = equal
  length right before and after. Placebo = same clock times at the same
  recorder on other days with no visit within 2 h (up to 4 per visit).

Per clip, standard soundscape measures:
  total_db  overall loudness
  low_db    50-1500 Hz   (wind, engines, voices)
  bird_db   2-8 kHz      (most bird song)
  high_db   8-16 kHz     (insects)
  ndsi      (2-8 kHz minus 1-2 kHz) / sum   -1 = human/engine noise, +1 = animals
  aci       acoustic complexity, 2-8 kHz (busier = higher)
  entropy   spectral entropy (lower = a few dominant sounds)

Read AFTER - BEFORE first: no people are present then, so their own noise
cannot drive the result.

SETUP (once): in Drive, right-click "Audio Moth Data" > Organize > Add shortcut
> My Drive. Then in Colab: paste this whole file into one cell and run it.
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
RECORDERS = ["Audio_Moth_6"]          # add "Audio_Moth_1", ... once this works
BASE = "/content/drive/MyDrive/Audio Moth Data"
CSV = os.path.join(BASE, "bioacoustics_dataset.csv")
AUDIO = os.path.join(BASE, "Segmented Foldered")
OUT = "/content/drive/MyDrive/toucanet_soundscape"   # your own Drive
MAX_PLACEBO_PER_VISIT, THREADS, SEED = 4, 16, 0

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
        ok = lambda s, e: all(covered(g, a, b) >= 0.8 for a, b in [(s - L, s), (s, e), (e, e + L)])
        if not ok(x.start, x.end):
            continue
        windows.append(("visit", rec, x.start, x.end))
        opts = []
        for d in days:
            off = d - x.start.normalize()
            if off == pd.Timedelta(0):
                continue
            c2 = x.centre + off
            if ((sess.rec == rec) & ((sess.centre - c2).abs() < pd.Timedelta("2h"))).any():
                continue
            if ok(x.start + off, x.end + off):
                opts.append(off)
        for off in rng.permutation(opts)[:MAX_PLACEBO_PER_VISIT] if opts else []:
            windows.append(("placebo", rec, x.start + off, x.end + off))
W = pd.DataFrame(windows, columns=["kind", "rec", "start", "end"])
print(W.groupby(["rec", "kind"]).size().to_string())

need = set()
for w in W.itertuples():
    L = w.end - w.start
    g = df[(df.Recorder == w.rec) & (df.ts >= w.start - L) & (df.ts < w.end + L)]
    need |= set(g.clip_name)
print(f"Clips to measure: {len(need):,}")

# ---------------- find files ----------------
print("Indexing audio folders (a few minutes the first time)...")
paths = {}
for rec in RECORDERS:
    folder = os.path.join(AUDIO, rec.replace("_", " "))
    for root, _, files in os.walk(folder):
        for f in files:
            if f in need:
                paths[f] = os.path.join(root, f)
print(f"  found {len(paths):,} of {len(need):,}")
if not paths:
    raise SystemExit("No clips found. Check the shortcut to 'Audio Moth Data' and the folder names.")


def band_db(f, P, lo, hi):
    m = (f >= lo) & (f < hi)
    return 10 * np.log10(P[m].mean() + 1e-15) if m.any() else np.nan


def measure(name):
    y, sr = load(paths[name])
    if y.ndim > 1:
        y = y.mean(axis=1)
    f, _, S = signal.spectrogram(y, fs=sr, nperseg=1024, noverlap=512)
    P = S.mean(axis=1)
    bird = (f >= 2000) & (f < 8000)
    Sb = S[bird] + 1e-15
    aci = float((np.abs(np.diff(Sb, axis=1)).sum(axis=1) / Sb.sum(axis=1)).sum())
    p = P / P.sum()
    ent = float(-(p * np.log2(p + 1e-15)).sum() / np.log2(len(p)))
    bio, anth = P[bird].sum(), P[(f >= 1000) & (f < 2000)].sum()
    return dict(clip_name=name, total_db=10 * np.log10(P.mean() + 1e-15), low_db=band_db(f, P, 50, 1500),
                bird_db=band_db(f, P, 2000, 8000), high_db=band_db(f, P, 8000, 16000),
                ndsi=(bio - anth) / (bio + anth + 1e-15), aci=aci, entropy=ent)


cache_file = os.path.join(OUT, "clip_measures.csv")
done = pd.read_csv(cache_file) if os.path.exists(cache_file) else pd.DataFrame(columns=["clip_name"])
todo = [n for n in paths if n not in set(done.clip_name)]
print(f"Measuring {len(todo):,} clips ({len(done):,} already saved)...")
t0, rows = time.time(), []


def safe(n):
    try:
        return measure(n)
    except Exception:
        return None


with ThreadPoolExecutor(THREADS) as ex:
    for i, r in enumerate(ex.map(safe, todo), 1):
        if r:
            rows.append(r)
        if i % 2000 == 0 or i == len(todo):
            done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
            done.to_csv(cache_file, index=False)
            rows = []
            print(f"  {i:,}/{len(todo):,}  ({(time.time() - t0) / 60:.1f} min)")

# ---------------- windows and tests ----------------
M = ["total_db", "low_db", "bird_db", "high_db", "ndsi", "aci", "entropy"]
meas = df.merge(done, on="clip_name")
res = []
for w in W.itertuples():
    L = w.end - w.start
    g = meas[meas.Recorder == w.rec]
    row = dict(kind=w.kind, rec=w.rec, start=w.start)
    for part, (a, b) in [("before", (w.start - L, w.start)), ("during", (w.start, w.end)), ("after", (w.end, w.end + L))]:
        x = g[(g.ts >= a) & (g.ts < b)]
        enough = len(x) >= 0.5 * (b - a).total_seconds() / 3
        for m in M:
            row[f"{part}_{m}"] = x[m].median() if enough else np.nan
    res.append(row)
R = pd.DataFrame(res)
R.to_csv(os.path.join(OUT, "soundscape_windows.csv"), index=False)

out = ["SOUNDSCAPE CHECK (full audio, bioacoustics labels)", "=" * 64,
       f"Recorders: {', '.join(RECORDERS)}   visits: {(R.kind == 'visit').sum()}   placebo windows: {(R.kind == 'placebo').sum()}", ""]
for when in ["after", "during"]:
    out.append(f"{when.upper()} - BEFORE   (visits vs placebo)" + ("   <- read this first" if when == "after" else ""))
    for m in M:
        dv = (R[R.kind == "visit"][f"{when}_{m}"] - R[R.kind == "visit"][f"before_{m}"]).dropna()
        dp = (R[R.kind == "placebo"][f"{when}_{m}"] - R[R.kind == "placebo"][f"before_{m}"]).dropna()
        p = stats.mannwhitneyu(dv, dp).pvalue if len(dv) >= 3 and len(dp) >= 3 else np.nan
        out.append(f"  {m:<9} visits median {dv.median():+.3f} (lower in {int((dv < 0).sum())}/{len(dv)})   "
                   f"placebo median {dp.median():+.3f} (n={len(dp)})   p={p:.3f}")
    out.append("")
out.append("dB measures: negative = quieter than before. ndsi: negative = more engine/voice-band sound.")
txt = "\n".join(out)
print(txt)
open(os.path.join(OUT, "soundscape_results.txt"), "w").write(txt)
print(f"\nSaved to {OUT}")
