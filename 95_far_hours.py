r"""
95_far_hours.py  —  run in Google Colab after scripts 92-94 (reads some audio; ~1-1.5 h, resumable)
---------------------------------------------------------------------------------------------------
Scripts 93-94: around human-presence visits, birds (2-7 kHz) and insects (10-13 kHz) are quieter
than on normal days, but the quiet is already there 40 min before people arrive and does not build up.
So is it the PEOPLE, or the DAY (visit days just being different, e.g. windier)?

Test: on each visit day, take hours FAR from any event (at least 3 h from any event at that
recorder and 2 h from any event at any recorder, so the team is likely not nearby), and compare them
with the same clock time on that visit's normal days (same recorder, same days as before).

  - far hours also quieter  -> it is the day, not the people
  - far hours normal, quiet only near visits -> the quiet is tied to people being in the forest

Also: the same visits' near-visit values (from script 92's features), a paired test near minus far
(within the same day, so anything about the day cancels), and wind at the far hours.

Results: far_hours_results.txt and far_hours.csv in toucanet_soundscape.
Clips are cached in far_hours_cache.npz, so if Colab disconnects, just run it again.

Colab:  from google.colab import drive; drive.mount("/content/drive")
        %run /content/95_far_hours.py
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
MID = tuple(range(-30, 30, 5))
FAR_PER_STEP = 8                       # clips per 5-min step in far windows
GAP_SAME, GAP_ANY = pd.Timedelta("3h"), pd.Timedelta("2h")
OFFSETS_H = [-3, 3, -4, 4, -5, 5, -6, 6]
DAY_START, DAY_END = 5.5, 18.5         # keep far windows in daylight (local clock hours)
vis = Wn[(Wn.kind == "visit") & (Wn.group == "presence")]
vis = vis[[((Wn.kind == "placebo") & (Wn.vid == v)).any() for v in vis.vid]]
print(f"Human-presence visits with normal days: {len(vis)}")


def bands(vecs):
    if len(vecs) < 20:
        return None
    A = 10 * np.mean(vecs, axis=0)
    return np.array([A[bmask].mean(), A[imask].mean()])


# ---------------- near-visit values (script 92 features, same as script 94) ----------------
def near_db(i):
    return bands([feat["spec"][n] for st in MID for n in plan.get((i, st), []) if n in feat["spec"] and not human_tag.get(n, False)])


# ---------------- far windows ----------------
def quiet_time(rec, t):
    same = E[E.rec == rec].t.sub(t).abs().min() if (E.rec == rec).any() else pd.Timedelta("99D")
    anyr = E.t.sub(t).abs().min()
    return same >= GAP_SAME and anyr >= GAP_ANY


rng2 = np.random.default_rng(7)
far_plan, need2, far_pairs = {}, set(), []
for _, v in vis.iterrows():
    pls = list(Wn[(Wn.kind == "placebo") & (Wn.vid == v.vid)].index)
    chosen = []
    for h in OFFSETS_H:
        if len(chosen) == 2:
            break
        f = v.t0 + pd.Timedelta(hours=h)
        hr = f.hour + f.minute / 60
        if not (DAY_START <= hr <= DAY_END) or not quiet_time(v.rec, f):
            continue
        if covered(v.rec, f - pd.Timedelta("30min"), f + pd.Timedelta("30min")) < 0.8:
            continue
        ok_pl = []
        for p in pls:
            fp = f + (Wn.loc[p, "t0"] - v.t0)
            if quiet_time(v.rec, fp) and covered(v.rec, fp - pd.Timedelta("30min"), fp + pd.Timedelta("30min")) >= 0.8:
                ok_pl.append((p, fp))
        if not ok_pl:
            continue
        chosen.append(h)
        key = (v.vid, h)
        far_pairs.append(dict(vid=v.vid, rec=v.rec, h=h, f=f, placebos=ok_pl))
        g = by_rec[v.rec]
        for tag, t in [("visit", f)] + [(f"pl{p}", fp) for p, fp in ok_pl]:
            names = []
            for st in MID:
                a = t + pd.Timedelta(minutes=st)
                x = g[(g.ts >= a) & (g.ts < a + pd.Timedelta(minutes=STEP))]
                if len(x) >= 6:
                    names += list(x.clip_name.sample(min(FAR_PER_STEP, len(x)), random_state=int(rng2.integers(1e9))))
            far_plan[(key, tag)] = names
            need2 |= set(names)
print(f"Far windows: {len(far_pairs)} (on {len(set(p['vid'] for p in far_pairs))} visit days); clips to read: {len(need2):,}")

cache2 = os.path.join(OUT, "far_hours_cache.npz")
spec2 = {}
if os.path.exists(cache2):
    z2 = np.load(cache2, allow_pickle=True)
    spec2 = dict(zip(z2["names"], z2["vals"].astype("float32")))
for n in need2:
    if n not in spec2 and n in feat["spec"]:
        spec2[n] = feat["spec"][n]
todo = sorted(n for n in need2 if n not in spec2)
print(f"  {len(need2) - len(todo):,} already measured, {len(todo):,} to read")


def save2():
    np.savez(cache2 + ".tmp.npz", names=np.array(list(spec2.keys()), dtype=object),
             vals=np.array(list(spec2.values()), dtype="float16") if spec2 else np.zeros((0, 48), "float16"))
    os.replace(cache2 + ".tmp.npz", cache2)


if todo:
    print("Finding the clips (a few minutes)...")
    want, paths = set(todo), {}
    for rec in sorted(set(p["rec"] for p in far_pairs)):
        for root, _, files in os.walk(os.path.join(AUDIO, rec.replace("_", " "))):
            for f_ in files:
                if f_ in want:
                    paths[f_] = os.path.join(root, f_)
    print(f"  found {len(paths):,} of {len(todo):,}")

    def read(name):
        try:
            y, sr = load(paths[name])
            if sr != 32000:
                y = signal.resample_poly(y, 32000, sr).astype("float32")
            y = y - y.mean()
            f, P = signal.welch(y, fs=32000, nperseg=1024)
            s = np.array([P[(f >= a) & (f < b)].mean() for a, b in zip(EDGES[:-1], EDGES[1:])])
            return name, np.log10(s + 1e-15).astype("float32")
        except Exception:
            return None

    names = [n for n in todo if n in paths]
    t_start = time.time()
    with ThreadPoolExecutor(THREADS) as ex:
        for b0 in range(0, len(names), 1000):
            for r in ex.map(read, names[b0:b0 + 1000]):
                if r:
                    spec2[r[0]] = r[1]
            save2()
            print(f"  {min(b0 + 1000, len(names)):,}/{len(names):,}  ({(time.time() - t_start) / 60:.1f} min)")


def far_db(key, tag):
    return bands([spec2[n] for n in far_plan.get((key, tag), []) if n in spec2 and not human_tag.get(n, False)])


# ---------------- weather (wind) ----------------
WX = ["Temperature", "Humidity", "Windspeed"]
wx = None
if all(c in df.columns for c in WX + ["Datetime"]):
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


def wind_at(t):
    if wx is None:
        return np.nan
    x = wx[(wx.index >= (t - pd.Timedelta("30min")).floor("h")) & (wx.index < t + pd.Timedelta("30min"))]
    return x.Windspeed.mean() if len(x) else np.nan


# ---------------- compare ----------------
rows = []
for _, v in vis.iterrows():
    pls = list(Wn[(Wn.kind == "placebo") & (Wn.vid == v.vid)].index)
    nv = near_db(v.name)
    npl = [x for x in (near_db(p) for p in pls) if x is not None]
    near = nv - np.median(npl, axis=0) if nv is not None and npl else np.array([np.nan, np.nan])
    near_w = wind_at(v.t0) - np.nanmedian([wind_at(Wn.loc[p, "t0"]) for p in pls])
    fars, fws = [], []
    for fp in [p for p in far_pairs if p["vid"] == v.vid]:
        key = (v.vid, fp["h"])
        fv = far_db(key, "visit")
        fpl = [x for x in (far_db(key, f"pl{p}") for p, _ in fp["placebos"]) if x is not None]
        if fv is not None and fpl:
            fars.append(fv - np.median(fpl, axis=0))
            fws.append(wind_at(fp["f"]) - np.nanmedian([wind_at(t) for _, t in fp["placebos"]]))
    far = np.mean(fars, axis=0) if fars else np.array([np.nan, np.nan])
    rows.append(dict(vid=v.vid, rec=v.rec, n_far=len(fars), near_bird=near[0], near_insect=near[1],
                     far_bird=far[0], far_insect=far[1], near_wind=near_w, far_wind=np.nanmean(fws) if fws else np.nan))
R = pd.DataFrame(rows)
R.to_csv(os.path.join(OUT, "far_hours.csv"), index=False)


def test(a, word="quieter"):
    a = np.asarray(a, dtype=float)
    a = a[~np.isnan(a)]
    if len(a) < 4:
        return f"n={len(a)} (too few to test)"
    k = int((a < 0).sum())
    return (f"median {np.median(a):+.2f} dB, {word} in {k}/{len(a)}, sign p={stats.binomtest(k, len(a)).pvalue:.3f}, "
            f"Wilcoxon p={stats.wilcoxon(a).pvalue:.3f}")


both = R.dropna(subset=["far_bird", "near_bird"])
out = ["FAR HOURS: is the quiet about the PEOPLE or about the DAY?", "=" * 92,
       f"Visit days with at least one far window: {int((R.n_far > 0).sum())} of {len(R)}  "
       f"(far = at least 3 h from any event at that recorder and 2 h from any event anywhere; daylight only)",
       "All values: visit day minus the same clock time on that visit's normal days, -30 to +30 min, human-tagged clips removed (dB).", ""]
for band, label in [("bird", "BIRD band 2-7 kHz"), ("insect", "INSECT band 10-13 kHz")]:
    out.append(label)
    out.append(f"  near the visit            : {test(R[f'near_{band}'])}")
    out.append(f"  far from visits, same day : {test(R[f'far_{band}'])}")
    out.append(f"  near minus far (same day) : {test(both[f'near_{band}'] - both[f'far_{band}'], 'more quiet near the visit')}")
    out.append("")
out.append("WIND, visit day minus normal days (km/h)")
out.append(f"  near the visit            : median {R.near_wind.median():+.2f}, windier in {(R.near_wind > 0).sum()}/{R.near_wind.notna().sum()}")
out.append(f"  far from visits, same day : median {R.far_wind.median():+.2f}, windier in {(R.far_wind > 0).sum()}/{R.far_wind.notna().sum()}")
out += ["",
        "How to read it:",
        "  - Far hours quieter too, and 'near minus far' about zero -> the quiet is about the DAY (weather, season),",
        "    not the people.",
        "  - Far hours normal, and 'near minus far' clearly negative -> the quiet is tied to people being in the",
        "    forest; that comparison is within the same day, so the day's weather cancels out.",
        "  - Wind: if the far hours are windier too, the whole day was windy; if not, wind was only up near visits."]
txt = "\n".join(out)
print(txt)
open(os.path.join(OUT, "far_hours_results.txt"), "w").write(txt)
print(f"\nSaved to {OUT}")
