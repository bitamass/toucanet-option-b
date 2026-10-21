"""
72_am6_audio_check.py
---------------------
Audio check of the script-71 lead: does the soundscape at recorder 6 get
quieter during and after daytime simulations?

Only the clips inside the analysis windows are read (a few thousand files),
so it is light on a laptop. Run from a plain terminal, not PyCharm.

For every daytime session at AM6 (same sessions as script 71) and every
placebo window (same clock time, AM6, days with no simulation within 2 h):
  BEFORE = 30 min before the simulation window, DURING = the 30-min window,
  AFTER = 30 min after.
Per clip (3 s), three standard soundscape measures:
  bird_db  - sound energy in the 2-8 kHz bird band (dB)
  aci      - acoustic complexity index in 2-8 kHz (busier = higher)
  entropy  - spectral entropy (lower = fewer, more dominant sounds)
Per window: the median over its clips.

AFTER vs BEFORE is the cleanest comparison, because no people are present
in AFTER, so human noise cannot inflate or mask the measures.

Usage:
  python 72_am6_audio_check.py "cleaned_df (1).csv" "C:\\path\\to\\folder_with_wavs"
The audio folder is searched recursively for Audio_Moth_6_*.wav files.
Needs: pandas numpy scipy   (uses soundfile if installed, else scipy)
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats, signal

try:
    import soundfile as sf
    def load(p):
        y, sr = sf.read(p, dtype="float32", always_2d=False)
        return (y.mean(axis=1) if y.ndim > 1 else y), sr
except ImportError:
    from scipy.io import wavfile
    def load(p):
        sr, y = wavfile.read(p)
        y = y.astype("float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        return y / 32768.0, sr

CSV, AUDIO = sys.argv[1], sys.argv[2]
REC = "Audio_Moth_6"
H, F = pd.Timedelta("15min"), pd.Timedelta("30min")
LO, HI = 2000, 8000

# ---- sessions (same rules as script 71) -----------------------------------
df = pd.read_csv(CSV, low_memory=False).drop_duplicates("clip_name")
df = df[df["Human Activity"].astype(str).str.lower() != "speech"]
df["ts"] = pd.to_datetime(df["Timestamp"], format="%Y%m%d_%H%M%S")
sim = df[df["Sim Type"] != "[]"].copy()
sim["rel"] = sim["Sim Relative Time"].str.strip("[]").str.split(",").str[0].astype(float)
sim["event"] = sim.ts - pd.to_timedelta(sim.rel, unit="s")
cent = []
for r, g in sim.groupby("Recorder"):
    e = g.sort_values("event")
    grp = (e.event.diff() > pd.Timedelta("20min")).cumsum()
    for _, x in e.groupby(grp):
        cent.append((r, x.event.median()))
cent = pd.DataFrame(cent, columns=["r", "c"])
mins = lambda t: t.hour * 60 + t.minute
day = cent[[r == REC and mins(c - H - F) >= 360 and mins(c + H + F) <= 1050
            for r, c in cent.itertuples(index=False)]]
g6 = df[df.Recorder == REC]

def running(c):  # recorder must be logging clips within 3 h before AND after
    return ((g6.ts < c - H - F) & (g6.ts > c - pd.Timedelta("3h"))).any() and \
           ((g6.ts > c + H + F) & (g6.ts < c + pd.Timedelta("3h"))).any()


windows = [("session", c) for c in day.c if running(c)]
for c in day.c:
    tod = c - c.normalize()
    for d in sorted(g6.ts.dt.normalize().unique()):
        cc = d + tod
        if (abs(cent[cent.r == REC].c - cc) < pd.Timedelta("2h")).any():
            continue
        if ((g6.ts < cc - H - F) & (g6.ts > cc - pd.Timedelta("3h"))).any() and \
           ((g6.ts > cc + H + F) & (g6.ts < cc + pd.Timedelta("3h"))).any():
            windows.append(("placebo", cc))
print(f"AM6 daytime sessions: {len(day)}   placebo windows: {len(windows) - len(day)}")

# ---- find the needed wav files -------------------------------------------
need = set()
for _, c in windows:
    w = g6[(g6.ts >= c - H - F) & (g6.ts < c + H + F)]
    need |= set(w.clip_name)
paths = {}
for root, _, files in os.walk(AUDIO):
    for f in files:
        if f in need:
            paths[f] = os.path.join(root, f)
print(f"Clips needed: {len(need)}   found on disk: {len(paths)}")
if not paths:
    sys.exit("No matching wav files found. Check the audio folder path.")


def measures(path):
    y, sr = load(path)
    f, t, S = signal.spectrogram(y, fs=sr, nperseg=1024, noverlap=512)
    band = (f >= LO) & (f <= HI)
    Sb = S[band] + 1e-12
    bird_db = 10 * np.log10(Sb.mean())
    aci = float((np.abs(np.diff(Sb, axis=1)).sum(axis=1) / Sb.sum(axis=1)).sum())
    p = S.mean(axis=1) + 1e-12
    p = p / p.sum()
    entropy = float(-(p * np.log2(p)).sum() / np.log2(len(p)))
    return bird_db, aci, entropy


cache = {}
for i, (name, p) in enumerate(paths.items(), 1):
    try:
        cache[name] = measures(p)
    except Exception as e:
        print(f"  skip {name}: {e}")
    if i % 500 == 0:
        print(f"  measured {i}/{len(paths)} clips")


def window_median(c0, c1):
    names = g6[(g6.ts >= c0) & (g6.ts < c1)].clip_name
    vals = np.array([cache[n] for n in names if n in cache])
    return np.median(vals, axis=0) if len(vals) >= 5 else np.full(3, np.nan)


rows = []
for kind, c in windows:
    b, d, a = window_median(c - H - F, c - H), window_median(c - H, c + H), window_median(c + H, c + H + F)
    rows.append(dict(kind=kind, center=c, **{f"{k}_{m}": v for k, arr in
                     [("before", b), ("during", d), ("after", a)]
                     for m, v in zip(["bird_db", "aci", "entropy"], arr)}))
res = pd.DataFrame(rows)
res.to_csv("am6_audio_windows.csv", index=False)

out = ["AM6 AUDIO CHECK", "=" * 60]
for m, label in [("bird_db", "Bird-band loudness (dB)"), ("aci", "Acoustic complexity"),
                 ("entropy", "Spectral entropy")]:
    out.append(label)
    for x in ["during", "after"]:
        s = res[res.kind == "session"]; pl = res[res.kind == "placebo"]
        ds = (s[f"{x}_{m}"] - s[f"before_{m}"]).dropna()
        dp = (pl[f"{x}_{m}"] - pl[f"before_{m}"]).dropna()
        p = stats.mannwhitneyu(ds, dp).pvalue if len(ds) >= 3 and len(dp) >= 3 else np.nan
        out.append(f"  {x:>6} - before: sessions median {ds.median():+.3f} (lower in {int((ds < 0).sum())}/{len(ds)})"
                   f"   placebo median {dp.median():+.3f} (n={len(dp)})   p={p:.3f}")
    out.append("")
out.append("Read AFTER - BEFORE first: no people are present then.")
txt = "\n".join(out)
print(txt)
open("am6_audio_results.txt", "w").write(txt)
print("\nSaved am6_audio_results.txt and am6_audio_windows.csv")
