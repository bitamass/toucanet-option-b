"""
88_what_goes_quiet.py  —  run in Google Colab (reads a sample of audio, ~30-40 min)
-----------------------------------------------------------------------------------
Script 86 found the 6-8 kHz band dropping by ~3 dB from about 15 minutes before people
arrive. What goes quiet?

For the matched visits (same as scripts 81-87) and their own normal days, it compares
  EARLY = 30 to 15 minutes before arrival   vs   LATE = 15 to 0 minutes before arrival
A. Fine spectrum, 250 Hz steps from 4 to 10 kHz:
     change = LATE minus EARLY (dB); diff = visit change minus its normal days' change
   A few narrow steps dropping  -> insects (fixed-pitch callers)
   A broad drop across 6-8 kHz  -> birds (songs sweep across many pitches)
B. Pictures and clips to check by eye and ear, saved to toucanet_soundscape/listen_before/:
     - for the visits with the biggest 6-8 kHz drop: a spectrogram of EARLY vs LATE
       (10 clips each, 0-12 kHz, the 6-8 kHz band marked)
     - two typical clips from EARLY and two from LATE for every matched visit

Colab:  upload this file, then   %run /content/88_what_goes_quiet.py
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


# ---------------- sample clips ----------------
import shutil
from scipy import signal
from concurrent.futures import ThreadPoolExecutor
PER, STEP, F0, F1 = 40, 250, 4000, 10000
W["tod"] = W.start - W.start.dt.normalize()
W["dur"] = W.end - W.start
rng3 = np.random.default_rng(3)
pairs, plan, need = [], {}, set()
for i, v in W[W.kind == "visit"].iterrows():
    pl = W[(W.kind == "placebo") & (W.rec == v.rec) & (W.tod == v.tod) & (W.dur == v.dur)]
    if len(pl) == 0:
        continue
    pairs.append((i, list(pl.index)))
    for j, w in [(i, v)] + list(pl.iterrows()):
        g = df[df.Recorder == w.rec]
        for part, (a, b) in [("early", (-30, -15)), ("late", (-15, 0))]:
            x = g[(g.ts >= w.start + pd.Timedelta(minutes=a)) & (g.ts < w.start + pd.Timedelta(minutes=b))].sort_values("ts")
            if len(x) >= 20:
                pick = list(x.clip_name.sample(min(PER, len(x)), random_state=int(rng3.integers(1e9))))
                plan[(j, part)] = pick
                need |= set(pick)
                if j == i:   # extra clips for pictures: 10 consecutive from the middle of the period
                    mid = len(x) // 2
                    run = list(x.clip_name.iloc[max(0, mid - 5): mid + 5])
                    plan[(j, part + "_run")] = run
                    need |= set(run)
print(f"Matched visits: {len(pairs)}   clips to read: {len(need):,}")

print("Finding the clips (a few minutes)...")
paths = {}
for rec in sorted({W.loc[i, "rec"] for i, _ in pairs}):
    for root, _, files in os.walk(os.path.join(AUDIO, rec.replace("_", " "))):
        for f in files:
            if f in need:
                paths[f] = os.path.join(root, f)
print(f"  found {len(paths):,} of {len(need):,}")

edges = np.arange(F0, F1 + STEP, STEP)


def fine(name):
    try:
        y, sr = load(paths[name])
        if y.ndim > 1:
            y = y.mean(axis=1)
        f, P = signal.welch(y, fs=sr, nperseg=4096)
        band = np.array([P[(f >= a) & (f < b)].mean() for a, b in zip(edges[:-1], edges[1:])])
        return band, P[(f >= 6000) & (f < 8000)].mean()
    except Exception:
        return None


print("Measuring...")
names = [n for n in need if n in paths]
spec, b68 = {}, {}
with ThreadPoolExecutor(THREADS) as ex:
    for k, (n, r) in enumerate(zip(names, ex.map(fine, names)), 1):
        if r is not None:
            spec[n], b68[n] = r
        if k % 1000 == 0:
            print(f"  {k:,}/{len(names):,}")


def db(ns):
    arr = [spec[n] for n in ns if n in spec]
    return 10 * np.log10(np.mean(arr, axis=0) + 1e-15) if len(arr) >= 10 else None


def change(j):
    e, l = db(plan.get((j, "early"), [])), db(plan.get((j, "late"), []))
    return None if e is None or l is None else l - e


diffs, drop68 = [], {}
for i, pls in pairs:
    cv = change(i)
    pc = [c for c in (change(j) for j in pls) if c is not None]
    if cv is None or not pc:
        continue
    d = cv - np.median(pc, axis=0)
    diffs.append(d)
    lo_i, hi_i = np.searchsorted(edges, 6000), np.searchsorted(edges, 8000)
    drop68[i] = d[lo_i:hi_i].mean()
diffs = np.array(diffs)

# ---------------- A. report ----------------
med = np.median(diffs, axis=0)
down = (diffs < 0).sum(axis=0)
n = len(diffs)
p = np.array([stats.binomtest(int(k), n, 0.5).pvalue for k in down])
pd.DataFrame(dict(khz_from=edges[:-1] / 1000, khz_to=edges[1:] / 1000, median_diff_db=med,
                  visits_down=down, visits=n, p=p)).to_csv(os.path.join(OUT, "what_goes_quiet.csv"), index=False)
out = ["WHAT GOES QUIET BEFORE PEOPLE ARRIVE?  (15-0 min before vs 30-15 min before, beyond normal days)", "=" * 92,
       f"Matched visits: {n}.  Negative = quieter just before arrival.  Each - is 0.5 dB quieter.", "",
       "  kHz          diff dB   down/visits   p"]
for a, b, m, k, pp in zip(edges[:-1] / 1000, edges[1:] / 1000, med, down, p):
    bar = "-" * int(max(0, min(20, round(-m / 0.5)))) or ("+" * int(max(0, min(20, round(m / 0.5)))))
    tag = "  <- 6-8 kHz" if 6 <= a < 8 else ""
    out.append(f"  {a:5.2f}-{b:<5.2f}  {m:+6.2f}    {k:>3}/{n:<3}     {pp:5.3f}  {bar}{tag}")
in68 = (edges[:-1] >= 6000) & (edges[:-1] < 8000)
strong = in68 & (med < -1)
out += ["", f"6-8 kHz steps quieter by more than 1 dB: {int(strong.sum())} of {int(in68.sum())}"]
if strong.sum():
    out.append("Quietest steps: " + ", ".join(f"{a / 1000:.2f}-{(a + STEP) / 1000:.2f} kHz ({m:+.1f} dB)"
                                             for a, m in sorted(zip(edges[:-1][in68], med[in68]), key=lambda t: t[1])[:3]))
out += ["Reading: only a few neighbouring steps dropping -> likely an insect (one fixed pitch);",
        "         most of 6-8 kHz dropping together -> likely birds.", ""]

# ---------------- B. pictures and clips ----------------
listen = os.path.join(OUT, "listen_before")
os.makedirs(listen, exist_ok=True)
order = sorted(drop68, key=drop68.get)
copied = 0
for rank, i in enumerate(order, 1):
    v = W.loc[i]
    tag = f"{rank:02d}_{v.rec[-1]}_{v.start:%d%b_%H%M}"
    for part in ["early", "late"]:
        ns = [x for x in plan.get((i, part), []) if x in paths]
        for n_ in ns[:2]:
            shutil.copy(paths[n_], os.path.join(listen, f"{tag}_{part}_{n_}"))
            copied += 1
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for rank, i in enumerate(order[:4], 1):
        v = W.loc[i]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        for ax, part, title in zip(axes, ["early_run", "late_run"], ["30-15 min before", "15-0 min before"]):
            ys, sr = [], None
            for n_ in plan.get((i, part), []):
                if n_ in paths:
                    y, sr = load(paths[n_])
                    ys.append(y if y.ndim == 1 else y.mean(axis=1))
            if not ys:
                continue
            y = np.concatenate(ys)
            f, t, S = signal.spectrogram(y, fs=sr, nperseg=1024, noverlap=512)
            keep = f <= 12000
            ax.pcolormesh(t, f[keep] / 1000, 10 * np.log10(S[keep] + 1e-15), shading="auto", cmap="magma")
            ax.axhspan(6, 8, color="cyan", alpha=0.12)
            ax.set_title(title)
            ax.set_xlabel("seconds (10 clips)")
        axes[0].set_ylabel("kHz")
        fig.suptitle(f"{v.rec} {v.start:%d %b %H:%M}: 6-8 kHz change {drop68[i]:+.1f} dB vs normal days")
        fig.tight_layout()
        fig.savefig(os.path.join(listen, f"spectrogram_{rank:02d}_{v.rec[-1]}_{v.start:%d%b_%H%M}.png"), dpi=120)
        plt.close(fig)
    out.append(f"Spectrograms of the 4 visits with the biggest 6-8 kHz drop saved in {listen}")
except Exception as e:
    out.append(f"(spectrograms skipped: {e})")
out += [f"Listening clips: {copied} copied to {listen}",
        "  Names start with the visit's rank (01 = biggest 6-8 kHz drop). Play '_early_' then '_late_' for the same visit:",
        "  what can you hear in early that is missing in late?"]
txt = "\n".join(out)
print(txt)
open(os.path.join(OUT, "what_goes_quiet_results.txt"), "w").write(txt)
print(f"\nSaved to {OUT}")
