"""
87_weather_and_species_before.py  —  run in Google Colab after scripts 79-86 (no audio)
--------------------------------------------------------------------------------------
Two checks on the latest findings, using the same visits and normal-day windows:

A. WEATHER. Insect choruses and wind depend on temperature, humidity and wind, and
   matching by clock time does not control weather. Were visit days different from
   their matched normal days, before and after the visit? Does the after-visit rise
   (high band 8-16 kHz, bird band 2-8 kHz; from script 82, noisy clips removed)
   follow the weather difference?
   The hourly weather columns in the dataset may be shifted by several hours, so the
   script first finds the shift that puts the daily temperature peak in the early
   afternoon, and prints the daily cycle so you can check it.

B. WHO GOES QUIET BEFORE ARRIVAL? Script 86 found the 6-8 kHz band dropping from about
   15 minutes before people arrive. For each BirdNET species:
     change = share of clips with it 15-0 min before arrival minus 30-15 min before
     diff   = visit change minus the same on its own normal days
   Species are ranked by how consistently they drop.

Colab:  upload this file, then   %run /content/87_weather_and_species_before.py
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
cols = {"clip_name", "Recorder", "Timestamp", "Sim Type", "Sim Relative Time", "species", "confidence",
        "Datetime", "Temperature", "Humidity", "Windspeed", "Precipitation"}
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


import ast
W["tod"] = W.start - W.start.dt.normalize()
W["dur"] = W.end - W.start
pairs = []
for i, v in W[W.kind == "visit"].iterrows():
    pl = W[(W.kind == "placebo") & (W.rec == v.rec) & (W.tod == v.tod) & (W.dur == v.dur)]
    if len(pl):
        pairs.append((i, list(pl.index)))
out = ["WEATHER AND WHO GOES QUIET BEFORE ARRIVAL", "=" * 72, f"Matched visits: {len(pairs)}", ""]


def sign_line(name, d, unit=""):
    d = pd.Series(d, dtype=float).dropna()
    up, dn = int((d > 0).sum()), int((d < 0).sum())
    p = stats.binomtest(up, up + dn, 0.5).pvalue if up + dn >= 3 else np.nan
    return f"  {name:<28} median {d.median():+.3f}{unit}   higher in {up}/{up + dn}   p={p:.3f}"


# ---------------- A. weather ----------------
WX = ["Temperature", "Humidity", "Windspeed", "Precipitation"]
if not all(c in df.columns for c in WX + ["Datetime"]):
    out += ["A. WEATHER: the dataset has no weather columns, so this part is skipped.", ""]
else:
    wx = df[["Datetime"] + WX].copy()
    wx["Datetime"] = pd.to_datetime(wx.Datetime, errors="coerce")
    wx = wx.dropna(subset=["Datetime"]).groupby("Datetime")[WX].mean()
    best, best_err = 0, 99
    for k in range(-12, 13):
        hr = (wx.index - pd.Timedelta(hours=k)).hour
        peak = wx.Temperature.groupby(hr).mean().idxmax()
        err = abs(peak - 13.5)
        if err < best_err:
            best, best_err = k, err
    wx_local = wx.copy()
    wx_local.index = wx.index - pd.Timedelta(hours=best)
    cyc = wx_local.Temperature.groupby(wx_local.index.hour).mean()
    out += [f"A. WEATHER  (hourly weather shifted by {best:+d} h so the daily temperature peak is in the early afternoon)",
            "  daily temperature cycle, local hour: " + "  ".join(f"{h}h {t:.1f}" for h, t in cyc.items() if h % 3 == 0),
            "  -> check: warmest around 12-15h, coolest around 3-6h.", ""]

    def wmean(a, b):
        x = wx_local[(wx_local.index >= a.floor("h")) & (wx_local.index < b)]
        return x.mean() if len(x) else None

    rows = []
    for i, pls in pairs:
        v = W.loc[i]
        L = v.end - v.start
        rec = {}
        for part, (a, b) in [("before", (v.start - L, v.start)), ("after", (v.end, v.end + AFTER))]:
            vv = wmean(a, b)
            pp = [wmean(a + (W.loc[j].start - v.start), b + (W.loc[j].start - v.start)) for j in pls]
            pp = [x for x in pp if x is not None]
            if vv is None or not pp:
                continue
            base = pd.concat(pp, axis=1).median(axis=1)
            for c in WX:
                rec[f"{part}_{c}"] = vv[c] - base[c]
        rec["i"] = i
        rows.append(rec)
    Dw = pd.DataFrame(rows)
    out.append("  Visit day minus its own normal days (same recorder, same clock time):")
    for part in ["before", "after"]:
        for c, u in zip(WX, [" C", " %", " km/h", " mm"]):
            col = f"{part}_{c}"
            if col in Dw:
                out.append(sign_line(f"{part:<6} {c}", Dw[col], u))
    nf = os.path.join(OUT, "noise_filtered_windows.csv")
    if os.path.exists(nf):
        N = pd.read_csv(nf)
        N = N[N.filt == True]
        sd = []
        for i, pls in pairs:
            a = N[N.i == i]
            b = N[N.i.isin(pls)]
            if len(a) and len(b):
                sd.append(dict(i=i, high_db=a.high_db.iloc[0] - b.high_db.median(),
                               bird_db=a.bird_db.iloc[0] - b.bird_db.median()))
        C = pd.DataFrame(sd).merge(Dw, on="i")
        out += ["", "  Does the after-visit rise follow the weather difference? (Spearman, across visits)"]
        for snd in ["high_db", "bird_db"]:
            for c in ["after_Temperature", "after_Humidity", "after_Windspeed"]:
                if c in C and len(C) >= 6:
                    rho, p = stats.spearmanr(C[snd], C[c], nan_policy="omit")
                    out.append(f"    {snd:<8} vs {c:<18} rho={rho:+.2f}  p={p:.3f}  (n={len(C)})")
    out.append("")


# ---------------- B. species in the 15 minutes before arrival ----------------
def birds(row):
    def L(v):
        if isinstance(v, str) and v.strip().startswith("["):
            try:
                return list(ast.literal_eval(v))
            except (ValueError, SyntaxError):
                return []
        return [] if v is None or (isinstance(v, float) and np.isnan(v)) else [v]
    sp, cf = L(row["species"]), L(row["confidence"])
    if len(cf) == 1 and len(sp) > 1:
        cf = cf * len(sp)
    outs = set()
    for s_, c in zip(sp, cf):
        try:
            if float(c) >= 0.25:
                outs.add(str(s_).split("_")[-1])
        except (TypeError, ValueError):
            pass
    return outs


print("Parsing BirdNET species...")
recs = {W.loc[i, "rec"] for i, _ in pairs}
dd = df[df.Recorder.isin(recs)].copy()
dd["birds"] = dd.apply(birds, axis=1)
dd["any_bird"] = (dd.birds.str.len() > 0).astype(float)
counts = pd.Series([b for s_ in dd.birds for b in s_]).value_counts()
SPECIES = list(counts[counts >= 200].index)
by_rec = {r: g for r, g in dd.groupby("Recorder")}


def shares(rec, a, b):
    g = by_rec[rec]
    x = g[(g.ts >= a) & (g.ts < b)]
    if len(x) < 50:
        return None
    d = {"any_bird": x.any_bird.mean()}
    for sp in SPECIES:
        d[sp] = x.birds.apply(lambda s_: sp in s_).mean()
    return d


def pre_change(w):
    m30, m15 = w.start - pd.Timedelta("30min"), w.start - pd.Timedelta("15min")
    a, b = shares(w.rec, m30, m15), shares(w.rec, m15, w.start)
    return None if a is None or b is None else {k: b[k] - a[k] for k in a}


rows = []
for i, pls in pairs:
    cv = pre_change(W.loc[i])
    cps = [c for c in (pre_change(W.loc[j]) for j in pls) if c is not None]
    if cv is None or not cps:
        continue
    for k in cv:
        base = np.median([c[k] for c in cps])
        rows.append(dict(i=i, measure=k, diff=cv[k] - base,
                         present=cv[k] != 0 or any(c[k] != 0 for c in cps)))
Ds = pd.DataFrame(rows, columns=["i", "measure", "diff", "present"])
out += ["B. WHO GOES QUIET IN THE 15 MINUTES BEFORE ARRIVAL?  (vs 30-15 min before, beyond normal days)"]
if len(Ds):
    out.append(sign_line("any bird", Ds[Ds.measure == "any_bird"]["diff"]))
    res = []
    for k, g in Ds[(Ds.measure != "any_bird") & Ds.present].groupby("measure"):
        d = g["diff"]
        up, dn = int((d > 0).sum()), int((d < 0).sum())
        if up + dn >= 6:
            res.append(dict(sp=k, down=dn, n=up + dn, med=d.median(), p=stats.binomtest(dn, up + dn, 0.5).pvalue))
    R = pd.DataFrame(res)
    if len(R):
        R = R.sort_values(["p", "med"])
        out.append("  Species that drop most consistently (heard at >= 6 visits):")
        for r in R[R.down > R.n - R.down].head(10).itertuples():
            out.append(f"    {r.sp[:32]:<32} down in {r.down}/{r.n}   median {r.med:+.3f}   p={r.p:.3f}")
else:
    out.append("  not enough data")
out += ["", "How to read it:",
        "  A: weather differences with small p, or a strong rho, mean weather may explain part of the soundscape change.",
        "  B: species that drop in most visits are candidates for the early-warning signal (birds going quiet as people approach)."]
txt = "\n".join(out)
print(txt)
open(os.path.join(OUT, "weather_species_before_results.txt"), "w").write(txt)
print(f"\nSaved to {OUT}")
