r"""
96_route_approach.py  —  run in Google Colab after scripts 92-95 (reads some audio; ~30-60 min, resumable)
--------------------------------------------------------------------------------------------------------
Script 95: birds (2-7 kHz) and insects (10-13 kHz) are quieter when people are in the area, and not at
other hours of the same day. The quiet starts ~40 min before the logged moment. Is that EARLY WARNING
(animals reacting before people get close), or simply the team already being nearby?

The team walks a route between recorders and logs its presence at each one. So for each human-presence
visit at a recorder, the previous log that day at ANOTHER recorder tells us when the team was one stop
back. Three windows at THIS recorder, each compared with the same clock time on that visit's normal days:

  BEFORE   : 40 to 10 min before the team was logged at the previous stop  (team not yet near)
  APPROACH : from the previous stop's log to 5 min before this log         (team one stop away)
  HERE     : 5 min before to 10 min after this log                         (team here)

  - quiet in APPROACH but not BEFORE -> the forest here reacts while people are still one stop away
                                        (the closest this data gets to early warning)
  - quiet only HERE                  -> a local reaction to people at the recorder
  - quiet already in BEFORE          -> the team was around even earlier (or something else about that hour)

Also a time course aligned on the previous stop's log. Only visits whose previous stop was logged
10-120 min earlier are used. Results: route_approach2_results.txt and route_approach2.csv.
Clips cached in route_cache.npz, so if Colab disconnects, just run it again.

Colab:  from google.colab import drive; drive.mount("/content/drive")
        %run /content/96_route_approach.py
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



PER_STEP2 = 8
GAP_MIN, GAP_MAX = pd.Timedelta("10min"), pd.Timedelta("120min")


def quiet_on(rec, t):
    """No event at this recorder within 2 h (for normal-day comparison windows)."""
    return not (E[E.rec == rec].t.sub(t).abs() < pd.Timedelta("2h")).any()


# ---------------- previous stop for each visit ----------------
legs = []
for _, v in vis.iterrows():
    prior = E[(E.rec != v.rec) & (E.t < v.t0) & (E.t >= v.t0 - GAP_MAX) & (E.t.dt.date == v.t0.date())]
    if prior.empty:
        continue
    p = prior.sort_values("t").iloc[-1]
    if v.t0 - p.t < GAP_MIN:
        continue
    legs.append(dict(vid=v.vid, idx=v.name, rec=v.rec, t0=v.t0, prev_rec=p.rec, tp=p.t, prev_type=p.type,
                     gap_min=(v.t0 - p.t).total_seconds() / 60))
L = pd.DataFrame(legs)
print(f"Visits with a previous stop logged 10-120 min earlier: {len(L)} of {len(vis)}")
if len(L):
    print("  legs: " + "; ".join(f"{a.prev_rec[-1]}->{a.rec[-1]} ({a.gap_min:.0f} min)" for a in L.itertuples()))
if len(L) < 4:
    sys.exit("Too few visits with a logged previous stop for this test.")


def windows(tp, t0):
    return {"before": (tp - pd.Timedelta("40min"), tp - pd.Timedelta("10min")),
            "approach": (tp, t0 - pd.Timedelta("5min")),
            "here": (t0 - pd.Timedelta("5min"), t0 + pd.Timedelta("10min"))}


TC = list(range(-40, 45, 5))          # time course: 5-min steps relative to the previous stop's log
rng3 = np.random.default_rng(11)
rplan, need3 = {}, set()


def sample(rec, a, b, key):
    g = by_rec[rec]
    names, t = [], a
    while t < b:
        x = g[(g.ts >= t) & (g.ts < min(t + pd.Timedelta("5min"), b))]
        if len(x) >= 4:
            names += list(x.clip_name.sample(min(PER_STEP2, len(x)), random_state=int(rng3.integers(1e9))))
        t += pd.Timedelta("5min")
    rplan[key] = names
    need3.update(names)


for a in L.itertuples():
    pls = [p for p in Wn[(Wn.kind == "placebo") & (Wn.vid == a.vid)].index]
    offs = [Wn.loc[p, "t0"] - a.t0 for p in pls]
    offs = [o for o in offs if quiet_on(a.rec, a.tp + o) and quiet_on(a.rec, a.t0 + o)]
    L.loc[L.vid == a.vid, "n_normal"] = len(offs)
    for name, (s, e) in windows(a.tp, a.t0).items():
        sample(a.rec, s, e, (a.vid, name, "visit"))
        for k, o in enumerate(offs):
            sample(a.rec, s + o, e + o, (a.vid, name, k))
    for st in TC:
        s = a.tp + pd.Timedelta(minutes=st)
        sample(a.rec, s, s + pd.Timedelta("5min"), (a.vid, ("tc", st), "visit"))
        for k, o in enumerate(offs):
            sample(a.rec, s + o, s + o + pd.Timedelta("5min"), (a.vid, ("tc", st), k))
print(f"Clips to measure: {len(need3):,}")

cache3 = os.path.join(OUT, "route_cache.npz")
spec3 = {}
if os.path.exists(cache3):
    z3 = np.load(cache3, allow_pickle=True)
    spec3 = dict(zip(z3["names"], z3["vals"].astype("float32")))
far_cache = os.path.join(OUT, "far_hours_cache.npz")
reuse = dict(feat["spec"])
if os.path.exists(far_cache):
    zf = np.load(far_cache, allow_pickle=True)
    reuse.update(zip(zf["names"], zf["vals"].astype("float32")))
for n in need3:
    if n not in spec3 and n in reuse:
        spec3[n] = reuse[n]
todo = sorted(n for n in need3 if n not in spec3)
print(f"  {len(need3) - len(todo):,} already measured, {len(todo):,} to read")


def save3():
    np.savez(cache3 + ".tmp.npz", names=np.array(list(spec3.keys()), dtype=object),
             vals=np.array(list(spec3.values()), dtype="float16") if spec3 else np.zeros((0, 48), "float16"))
    os.replace(cache3 + ".tmp.npz", cache3)


if todo:
    print("Finding the clips (a few minutes)...")
    want, paths = set(todo), {}
    for rec in sorted(L.rec.unique()):
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
                    spec3[r[0]] = r[1]
            save3()
            print(f"  {min(b0 + 1000, len(names)):,}/{len(names):,}  ({(time.time() - t_start) / 60:.1f} min)")


def db(key, minimum=12):
    v = [spec3[n] for n in rplan.get(key, []) if n in spec3 and not human_tag.get(n, False)]
    if len(v) < minimum:
        return None
    A = 10 * np.mean(v, axis=0)
    return np.array([A[bmask].mean(), A[imask].mean()])


def diff(vid, name, n_norm, minimum=12):
    dv = db((vid, name, "visit"), minimum)
    dn = [x for x in (db((vid, name, k), minimum) for k in range(n_norm)) if x is not None]
    return dv - np.median(dn, axis=0) if dv is not None and dn else np.array([np.nan, np.nan])


rows = []
for a in L.itertuples():
    nn = int(getattr(a, "n_normal", 0) or 0)
    r = dict(vid=a.vid, rec=a.rec, prev_rec=a.prev_rec, gap_min=a.gap_min, n_normal=nn)
    for w in ["before", "approach", "here"]:
        d = diff(a.vid, w, nn, minimum=6 if w == "approach" else 12)
        r[f"{w}_bird"], r[f"{w}_insect"] = d
    for st in TC:
        d = diff(a.vid, ("tc", st), nn, minimum=6)
        r[f"tc{st}_bird"], r[f"tc{st}_insect"] = d
    rows.append(r)
R = pd.DataFrame(rows)
R.to_csv(os.path.join(OUT, "route_approach2.csv"), index=False)


def test(a, word="quieter"):
    a = np.asarray(a, dtype=float)
    a = a[~np.isnan(a)]
    if len(a) < 4:
        return f"n={len(a)} (too few to test)"
    k = int((a < 0).sum())
    return (f"median {np.median(a):+.2f} dB, {word} in {k}/{len(a)}, sign p={stats.binomtest(k, len(a)).pvalue:.3f}, "
            f"Wilcoxon p={stats.wilcoxon(a).pvalue:.3f}")


out = ["ROUTE APPROACH: does the forest here go quiet while people are still one stop away?", "=" * 92,
       f"{len(R)} human-presence visits with the previous stop logged 10-120 min earlier "
       f"(median gap {R.gap_min.median():.0f} min); with normal days: {(R.n_normal > 0).sum()}",
       "All values: this recorder, visit day minus the same clock time on its normal days, human-tagged clips removed (dB).", ""]
for band, label in [("bird", "BIRD band 2-7 kHz"), ("insect", "INSECT band 10-13 kHz")]:
    out.append(label)
    out.append(f"  BEFORE   (40-10 min before the previous stop): {test(R[f'before_{band}'])}")
    out.append(f"  APPROACH (team one stop away)              : {test(R[f'approach_{band}'])}")
    out.append(f"  HERE     (team at this recorder)           : {test(R[f'here_{band}'])}")
    out.append(f"  approach minus before (same visit)          : {test(R[f'approach_{band}'] - R[f'before_{band}'], 'quieter on approach')}")
    out.append("")
out.append("TIME COURSE at this recorder, aligned on the PREVIOUS stop's log (0); median gap to this stop's log "
           f"is {R.gap_min.median():.0f} min. * = quieter in at least 75% of visits")
out.append("  minutes      bird 2-7k        insect 10-13k")
for st in TC:
    cells = []
    for band in ["bird", "insect"]:
        x = R[f"tc{st}_{band}"].dropna()
        if len(x) < 4:
            cells.append("      n/a       ")
            continue
        k = (x < 0).mean()
        cells.append(f"{x.median():+6.2f} {int((x < 0).sum()):>2}/{len(x):<2}{'*' if k >= .75 else ' '}")
    mark = "  <- team logged at previous stop" if st == 0 else ""
    out.append(f"  {st:+3d} to {st + 5:+3d}   {cells[0]}    {cells[1]}{mark}")
out += ["",
        "How to read it:",
        "  - APPROACH quieter, BEFORE normal, and 'approach minus before' negative -> the forest at this recorder",
        "    goes quiet while people are still one stop away (early warning, as far as this data can show).",
        "  - Only HERE quieter -> a local reaction once people reach the recorder.",
        "  - BEFORE already quieter -> people were around even earlier; this data cannot time the start.",
        "  - The previous stop's log can include played sounds (gunshot, vehicle, chainsaw); see prev_type in the csv."]
txt = "\n".join(out)
print(txt)
open(os.path.join(OUT, "route_approach2_results.txt"), "w").write(txt)
print(f"\nSaved to {OUT}")
