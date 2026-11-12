r"""
94_wind_check.py  —  run in Google Colab AFTER scripts 92-93 (no audio; about a minute)
---------------------------------------------------------------------------------------
Script 93: around human-presence visits the forest above 2 kHz is ~1-2.5 dB quieter than on
normal days (birds 2-7 kHz, insects 10-13 kHz), in 11-13 of 15 visits. But visit hours were also
windier (+2 km/h), and wind alone can make birds and insects call less. This script checks wind:

 1. Quieting in the BIRD band (2-7 kHz) and INSECT band (10-13 kHz), visit minus normal days,
    -30 to +30 min, clips with BirdNET human tags removed:
      - all visits
      - low-wind visits only (wind within 1 km/h of their normal days)
      - the calmer half of visits (smallest wind difference)
      - against the single normal day with the most similar wind
 2. Does the quieting follow the wind difference across visits? (Spearman, and the quieting
    expected at zero wind difference from a simple regression, with a bootstrap interval)
 3. Time course: bird and insect band, visit minus normal days, 5-min steps from -40 to +40 min.
    (also wind on visit hours vs normal days, hour by hour, to see if the whole day was windier)

Results: wind_check_results.txt, wind_check_timecourse.csv, wind_check.png in toucanet_soundscape.

Colab:  from google.colab import drive; drive.mount("/content/drive")
        %run /content/94_wind_check.py
"""
import ast, os, sys, time
import numpy as np
import pandas as pd
from scipy import signal, stats
from concurrent.futures import ThreadPoolExecutor

try:
    import soundfile as sf
    def load(p):
        y, sr = sf.read(p, dtype="float32", always_2d=False)
        return (y.mean(axis=1) if y.ndim > 1 else y), sr
except ImportError:
    from scipy.io import wavfile
    def load(p):
        sr, y = wavfile.read(p)
        y = y.astype("float32") / (32768.0 if y.dtype == np.int16 else 1.0)
        return (y.mean(axis=1) if y.ndim > 1 else y), sr

# ---------------- settings ----------------
GAP = pd.Timedelta("40min")          # events closer than this at one recorder = one visit
SPAN = 40                            # minutes before and after the anchor
STEP = 5                             # minutes per step
BASE_STEPS = (-40, -35)              # baseline steps (40-30 min before)
PER_STEP, MAX_PLACEBO, THREADS, SEED = 12, 3, 32, 0       # same as script 89 -> same clips
GROUPS = [(0, 1), (1, 2), (2, 4), (4, 6), (6, 8), (8, 11), (11, 14), (14, 16)]
CONF = 0.25

if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
    BASE = sys.argv[1]                                     # local test: folder with CSV + audio
    OUT = os.path.join(os.getcwd(), "toucanet_soundscape")
else:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE = "/content/drive/MyDrive/Audio Moth Data"
    OUT = "/content/drive/MyDrive/toucanet_soundscape"
CSV = os.path.join(BASE, "bioacoustics_dataset.csv")
AUDIO = os.path.join(BASE, "Segmented Foldered")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(SEED)


def as_list(v):
    if isinstance(v, str) and v.strip().startswith("["):
        try:
            return list(ast.literal_eval(v))
        except (ValueError, SyntaxError):
            return []
    if v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() in ("", "nan", "None"):
        return []
    return [v]


# ---------------- events from the labels ----------------
print("Reading labels...")
cols = {"clip_name", "Recorder", "Timestamp", "Sim Type", "Sim Relative Time", "species", "confidence",
        "Datetime", "Temperature", "Humidity", "Windspeed", "Precipitation"}
df = pd.read_csv(CSV, usecols=lambda c: c in cols, low_memory=False).drop_duplicates("clip_name")
df["ts"] = pd.to_datetime(df["Timestamp"].astype(str), format="%Y%m%d_%H%M%S", errors="coerce")
df = df.dropna(subset=["ts"]).sort_values(["Recorder", "ts"]).reset_index(drop=True)

ev = []
for r in df[df["Sim Type"].astype(str).str.strip() != "[]"].itertuples():
    types = as_list(df.at[r.Index, "Sim Type"])
    rels = as_list(df.at[r.Index, "Sim Relative Time"])
    for t, rel in zip(types, rels):
        try:
            ev.append((r.Recorder, (r.ts - pd.Timedelta(seconds=float(rel))).round("3s"), str(t)))
        except (TypeError, ValueError):
            pass
E = pd.DataFrame(ev, columns=["rec", "t", "type"]).drop_duplicates().sort_values(["rec", "t"])
E["kind"] = np.where(E.type.str.lower().str.contains("presence"), "presence", "sound")
print(f"  logged events: {len(E)}  ({(E.kind == 'presence').sum()} human presence, {(E.kind == 'sound').sum()} played sound)")
print("  event types: " + "; ".join(f"{k} ({v})" for k, v in E.type.value_counts().items()))

visits = []
for rec, g in E.groupby("rec"):
    grp = (g.t.diff() > GAP).cumsum()
    for _, x in g.groupby(grp):
        visits.append(dict(rec=rec, t0=x.t.iloc[0], kind=x.kind.iloc[0], first_type=x.type.iloc[0],
                           n_events=len(x), has_sound=(x.kind == "sound").any()))
V = pd.DataFrame(visits)
by_rec = {r: g for r, g in df.groupby("Recorder")}


def covered(rec, a, b):
    g = by_rec[rec]
    return ((g.ts >= a) & (g.ts < b)).sum() / ((b - a).total_seconds() / 3)


span = pd.Timedelta(minutes=SPAN)
win = []
for v in V.itertuples():
    if covered(v.rec, v.t0 - span, v.t0 + span) < 0.8:
        continue
    win.append(dict(kind="visit", group=v.kind, rec=v.rec, t0=v.t0, vid=len(win)))
    vid = win[-1]["vid"]
    days = sorted(by_rec[v.rec].ts.dt.normalize().unique())
    opts = []
    for d in days:
        off = d - v.t0.normalize()
        if off == pd.Timedelta(0):
            continue
        t2 = v.t0 + off
        if (E[(E.rec == v.rec)].t.sub(t2).abs() < pd.Timedelta("2h")).any():
            continue
        if covered(v.rec, t2 - span, t2 + span) >= 0.8:
            opts.append(t2)
    for t2 in (rng.permutation(np.array(opts, dtype="datetime64[ns]"))[:MAX_PLACEBO] if opts else []):
        win.append(dict(kind="placebo", group=v.kind, rec=v.rec, t0=pd.Timestamp(t2), vid=vid))
Wn = pd.DataFrame(win)
nv = Wn[Wn.kind == "visit"].groupby("group").size()
npl = Wn[Wn.kind == "placebo"].groupby("vid").size()
print(f"  visits with full recording: {dict(nv)}; visits with >=1 normal day: {len(npl)}")


# ---------------- BirdNET tags for human-made sounds (for B) ----------------
HUMAN_TAGS = ("human", "engine", "gun", "siren", "dog", "power tool", "chainsaw", "vehicle", "fireworks")
df["human_tag"] = df["species"].apply(lambda v: any(any(h in str(s).lower() for h in HUMAN_TAGS) for s in as_list(v)))
by_rec = {r: g for r, g in df.groupby("Recorder")}
steps = list(range(-SPAN, SPAN, STEP))

# ---------------- same clip sample as script 89 ----------------
plan, need = {}, set()
for w in Wn.itertuples():
    g = by_rec[w.rec]
    for st in steps:
        a = w.t0 + pd.Timedelta(minutes=st)
        x = g[(g.ts >= a) & (g.ts < a + pd.Timedelta(minutes=STEP))]
        if len(x) >= 8:
            pick = list(x.clip_name.sample(min(PER_STEP, len(x)), random_state=int(rng.integers(1e9))))
            if w.group == "presence":                     # this script uses human-presence visits only
                plan[(w.Index, st)] = pick
                need |= set(pick)
human_tag = dict(zip(df.clip_name, df.human_tag))
print(f"Human-presence windows: clips to featurise: {len(need):,}")


# ---------------- features from script 92 ----------------
EDGES = np.linspace(0, 16000, 49)
HIGH_BANDS = EDGES[:-1] >= 2000
cache_file = os.path.join(OUT, "model_features_cache.npz")
if not os.path.exists(cache_file):
    sys.exit("model_features_cache.npz not found: run script 92 first.")
z = np.load(cache_file, allow_pickle=True)
feat = {k: dict(zip(z[f"{k}_names"], z[f"{k}_vals"].astype("float32"))) for k in ["spec", "pfull", "phigh"] if f"{k}_names" in z}
print(f"Features loaded: {len(feat['spec']):,} clips (of {len(need):,} planned)")


BIRD, INSECT = (2000, 7000), (10000, 13000)
centres = (EDGES[:-1] + EDGES[1:]) / 2
bmask = (centres >= BIRD[0]) & (centres < BIRD[1])
imask = (centres >= INSECT[0]) & (centres < INSECT[1])
vis = Wn[(Wn.kind == "visit") & (Wn.group == "presence")]
vis = vis[[((Wn.kind == "placebo") & (Wn.vid == v)).any() for v in vis.vid]]
MID = tuple(range(-30, 30, 5))
print(f"Human-presence visits with normal days: {len(vis)}")


def band_db(i, sts):
    """Mean dB (10 x mean log10 power) in the bird and insect bands over the given steps; human-tagged clips removed."""
    A = [feat["spec"][n] for st in sts for n in plan.get((i, st), []) if n in feat["spec"] and not human_tag.get(n, False)]
    if len(A) < max(6, 2 * len(sts)):
        return None
    A = 10 * np.mean(A, axis=0)
    return np.array([A[bmask].mean(), A[imask].mean()])


# ---------------- weather ----------------
WX = ["Temperature", "Humidity", "Windspeed"]
if not all(c in df.columns for c in WX + ["Datetime"]):
    sys.exit("No weather columns in the dataset; this check needs Windspeed.")
wx = df[["Datetime"] + WX].copy()
wx["Datetime"] = pd.to_datetime(wx.Datetime, errors="coerce")
wx = wx.dropna(subset=["Datetime"]).groupby("Datetime")[WX].mean()
best, best_err = 0, 99
for k in range(-12, 13):
    hr = (wx.index - pd.Timedelta(hours=k)).hour
    err = abs(wx.Temperature.groupby(hr).mean().idxmax() - 13.5)
    if err < best_err:
        best, best_err = k, err
wx.index = wx.index - pd.Timedelta(hours=best)


def wind_at(t, half=30):
    x = wx[(wx.index >= (t - pd.Timedelta(minutes=half)).floor("h")) & (wx.index < t + pd.Timedelta(minutes=half))]
    return x.Windspeed.mean() if len(x) else np.nan


# ---------------- per-visit values ----------------
rows = []
for _, v in vis.iterrows():
    pls = list(Wn[(Wn.kind == "placebo") & (Wn.vid == v.vid)].index)
    sv = band_db(v.name, MID)
    sp = {p: band_db(p, MID) for p in pls}
    sp = {p: s for p, s in sp.items() if s is not None}
    if sv is None or not sp:
        continue
    wv = wind_at(v.t0)
    wp = {p: wind_at(Wn.loc[p, "t0"]) for p in sp}
    med = np.median(list(sp.values()), axis=0)
    closest = min(wp, key=lambda p: abs(wp[p] - wv) if not np.isnan(wp[p]) else 99)
    rows.append(dict(vid=v.vid, rec=v.rec, bird=sv[0] - med[0], insect=sv[1] - med[1],
                     dwind=wv - np.nanmedian(list(wp.values())),
                     bird_wm=sv[0] - sp[closest][0], insect_wm=sv[1] - sp[closest][1],
                     dwind_wm=wv - wp[closest]))
D = pd.DataFrame(rows)


def test(a):
    a = np.asarray(a, dtype=float)
    a = a[~np.isnan(a)]
    if len(a) < 4:
        return f"n={len(a)} (too few to test)"
    k = int((a < 0).sum())
    return (f"median {np.median(a):+.2f} dB, quieter in {k}/{len(a)}, sign p={stats.binomtest(k, len(a)).pvalue:.3f}, "
            f"Wilcoxon p={stats.wilcoxon(a).pvalue:.3f}")


out = ["WIND CHECK: is the quieting around visits just windier weather?", "=" * 92,
       f"{len(D)} human-presence visits; values are visit minus normal days, -30 to +30 min, human-tagged clips removed (dB).",
       f"Wind on visit hours minus normal days: median {D.dwind.median():+.2f} km/h, windier in {(D.dwind > 0).sum()}/{len(D)}", ""]
low = D[D.dwind.abs() <= 1.0]
calm = D[D.dwind <= D.dwind.median()]
for band, label in [("bird", "BIRD band 2-7 kHz"), ("insect", "INSECT band 10-13 kHz")]:
    out.append(label)
    out.append(f"  all visits                              : {test(D[band])}")
    out.append(f"  low-wind visits (within 1 km/h, n={len(low):>2})    : {test(low[band])}")
    out.append(f"  calmer half of visits (n={len(calm):>2})            : {test(calm[band])}")
    out.append(f"  vs the normal day with the closest wind : {test(D[band + '_wm'])}"
               f"   (wind difference then: median {D.dwind_wm.median():+.2f} km/h)")
    ok = D[["dwind", band]].dropna()
    if len(ok) >= 6:
        rho, p = stats.spearmanr(ok.dwind, ok[band])
        A = np.c_[np.ones(len(ok)), ok.dwind.values]
        b0 = np.linalg.lstsq(A, ok[band].values, rcond=None)[0][0]
        rb = np.random.default_rng(1)
        boot = []
        for _ in range(2000):
            ix = rb.integers(0, len(ok), len(ok))
            if ok.dwind.values[ix].std() > 0:
                boot.append(np.linalg.lstsq(A[ix], ok[band].values[ix], rcond=None)[0][0])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out.append(f"  follows the wind difference?            : Spearman rho={rho:+.2f}, p={p:.3f}")
        out.append(f"  quieting expected at ZERO wind difference: {b0:+.2f} dB  (95% bootstrap interval {lo:+.2f} to {hi:+.2f})")
    out.append("")

# ---------------- time course ----------------
tc = []
for _, v in vis.iterrows():
    pls = list(Wn[(Wn.kind == "placebo") & (Wn.vid == v.vid)].index)
    for st in steps:
        sv = band_db(v.name, (st,))
        sp = [s for s in (band_db(p, (st,)) for p in pls) if s is not None]
        wv, wp = wind_at(v.t0 + pd.Timedelta(minutes=st + 2.5), 3), [wind_at(Wn.loc[p, "t0"] + pd.Timedelta(minutes=st + 2.5), 3) for p in pls]
        if sv is not None and sp:
            d = sv - np.median(sp, axis=0)
            tc.append(dict(vid=v.vid, step=st, bird=d[0], insect=d[1], dwind=wv - np.nanmedian(wp)))
T = pd.DataFrame(tc)
T.to_csv(os.path.join(OUT, "wind_check_timecourse.csv"), index=False)
out.append("TIME COURSE (visit minus normal days, dB; * = quieter in at least 75% of visits)")
out.append("  minutes      bird 2-7k        insect 10-13k     wind (km/h)")
summ = []
for st in steps:
    x = T[T.step == st]
    if len(x) < 4:
        continue
    cells = []
    for band in ["bird", "insect"]:
        k = (x[band] < 0).mean()
        cells.append(f"{x[band].median():+6.2f} {int((x[band] < 0).sum()):>2}/{len(x):<2}{'*' if k >= .75 else ' '}")
    mark = "  <- logged" if st == 0 else ""
    out.append(f"  {st:+3d} to {st + STEP:+3d}   {cells[0]}    {cells[1]}    {x.dwind.median():+5.2f}{mark}")
    summ.append(dict(step=st, bird=x.bird.median(), insect=x.insect.median(), wind=x.dwind.median()))
out += ["",
        "How to read it:",
        "  - If the low-wind visits, the closest-wind comparison and the 'zero wind difference' value are still",
        "    clearly negative, wind does not explain the quieting.",
        "  - If the quieting disappears in the low-wind visits and follows the wind difference, it is probably wind.",
        "  - Time course: when does the quiet start, and does wind change at the same time or all day?"]
txt = "\n".join(out)
print(txt)
open(os.path.join(OUT, "wind_check_results.txt"), "w").write(txt)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    S = pd.DataFrame(summ)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(S.step + STEP / 2, S.bird, lw=2.8, color="#C8641C", label="birds 2-7 kHz")
    ax.plot(S.step + STEP / 2, S.insect, lw=2.8, color="#3B6EA8", label="insects 10-13 kHz")
    ax.axvline(0, color="#1F3A2E", ls="--", lw=1.5)
    ax.axvspan(-5, 0, color="#DDDDDD", alpha=0.6, lw=0)
    ax.axhline(0, color="#999999", lw=0.8)
    ax.set_xlabel("minutes from the logged human presence (grey: people audible)")
    ax.set_ylabel("visit minus normal days (dB)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "wind_check.png"), dpi=150)
    print("Saved wind_check.png")
except Exception as e:
    print(f"(plot skipped: {e})")
print(f"\nSaved to {OUT}")
