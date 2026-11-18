r"""
99_same_moment_all_recorders.py  —  run in Google Colab after scripts 92-98 (reads audio; ~1.5 h, resumable)
------------------------------------------------------------------------------------------------------------
The problem so far: whole days (especially evenings) can be quieter everywhere, and comparisons with
normal days pick that up. Script 98 showed unvisited recorders were often quiet too.

This analysis compares recorders ONLY AT THE SAME MOMENT, using all six recorders on every day with
team activity (human presence or played sounds), every 10-min slot in daylight (06:00-18:30):

  1. Each recorder-slot: bird (2-7 kHz) and insect (10-13 kHz) level minus that recorder's normal level at
     that clock time (median of up to 5 days with no team activity at all; if too few, days with no event at
     that recorder). Human-tagged clips removed.
  2. Each recorder-slot is labelled by its time to the nearest team event AT THAT RECORDER that day:
       approach : team arrives in 20-60 min      near : within 20 min of the team (before or after)
       after    : team left 20-60 min ago        mid  : 60-120 min away
       far      : more than 2 h away or no event there that day  (reference)
  3. Model with one intercept per moment (day x slot): only differences BETWEEN recorders at the same
     moment count, so weather, light, the evening chorus and season cancel completely.
  4. Permutation test: the team's schedule is moved to other recorders at random (same times, same days),
     2000 times. p = how often a fake schedule shows as much quiet as the real one. One honest number.

Results: same_moment_results.txt and same_moment_cells.csv in toucanet_soundscape.
Clips cached in same_moment_cache.npz, so if Colab disconnects, just run it again.

Colab:  from google.colab import drive; drive.mount("/content/drive")
        %run /content/99_same_moment_all_recorders.py
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



SLOT = 10                                   # minutes
DAY_START, DAY_END = 6 * 60, 18 * 60 + 30   # daylight, minutes after midnight
PER_CELL, PER_CELL_NORM, N_NORM_DAYS, N_PERM = 4, 3, 5, 2000
RECS = sorted(by_rec)
rng5 = np.random.default_rng(31)

E["day"] = E.t.dt.normalize()
act_days = sorted(E.day.unique())
all_days = sorted(df.ts.dt.normalize().unique())
quiet_days = [d for d in all_days if d not in set(act_days)]
print(f"Days with team activity: {len(act_days)}; days with no activity at all: {len(quiet_days)}")

# ---------------- sample clips ----------------
cell_clips, norm_clips, need5 = {}, {}, set()
for r in RECS:
    g = by_rec[r][["clip_name", "ts"]].copy()
    g["day"] = g.ts.dt.normalize()
    g["slot"] = (g.ts.dt.hour * 60 + g.ts.dt.minute) // SLOT * SLOT
    g = g[(g.slot >= DAY_START) & (g.slot < DAY_END)]
    g = g.sample(frac=1, random_state=int(rng5.integers(1e9)))
    cnt = g.groupby(["day", "slot"]).clip_name.transform("size")
    g = g[cnt >= 6]
    a = g[g.day.isin(act_days)].groupby(["day", "slot"]).head(PER_CELL)
    for (d, s), x in a.groupby(["day", "slot"]):
        cell_clips[(r, d, s)] = list(x.clip_name)
    good = []
    for pool in [quiet_days, [d for d in all_days if d not in set(E[E.rec == r].day)]]:
        qd = g[g.day.isin(pool)]
        cover = qd.groupby("day").slot.nunique()
        good = list(cover[cover >= 0.6 * (DAY_END - DAY_START) / SLOT].index)
        if len(good) >= 2:
            break
    qd = g[g.day.isin(good)]
    pick = [good[k] for k in rng5.permutation(len(good))[:N_NORM_DAYS]]
    n = qd[qd.day.isin(pick)].groupby(["day", "slot"]).head(PER_CELL_NORM)
    for (d, s), x in n.groupby(["day", "slot"]):
        norm_clips[(r, d, s)] = list(x.clip_name)
for v in list(cell_clips.values()) + list(norm_clips.values()):
    need5.update(v)
print(f"Recorder-slots on activity days: {len(cell_clips):,}; normal-day slots: {len(norm_clips):,}; clips: {len(need5):,}")

cache5 = os.path.join(OUT, "same_moment_cache.npz")
spec5 = {}
if os.path.exists(cache5):
    z5 = np.load(cache5, allow_pickle=True)
    spec5 = dict(zip(z5["names"], z5["vals"].astype("float32")))
reuse = dict(feat["spec"])
for fname in ["far_hours_cache.npz", "route_cache.npz", "same_evening_cache.npz"]:
    fp_ = os.path.join(OUT, fname)
    if os.path.exists(fp_):
        zz = np.load(fp_, allow_pickle=True)
        reuse.update(zip(zz["names"], zz["vals"].astype("float32")))
for n in need5:
    if n not in spec5 and n in reuse:
        spec5[n] = reuse[n]
todo = sorted(n for n in need5 if n not in spec5)
print(f"  {len(need5) - len(todo):,} already measured, {len(todo):,} to read")


def save5():
    np.savez(cache5 + ".tmp.npz", names=np.array(list(spec5.keys()), dtype=object),
             vals=np.array(list(spec5.values()), dtype="float16") if spec5 else np.zeros((0, 48), "float16"))
    os.replace(cache5 + ".tmp.npz", cache5)


if todo:
    print("Finding the clips (a few minutes)...")
    want, paths = set(todo), {}
    for rec in RECS:
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
                    spec5[r[0]] = r[1]
            save5()
            print(f"  {min(b0 + 1000, len(names)):,}/{len(names):,}  ({(time.time() - t_start) / 60:.1f} min)")


def cell_db(names, minimum):
    v = [spec5[n] for n in names if n in spec5 and not human_tag.get(n, False)]
    if len(v) < minimum:
        return None
    A = 10 * np.mean(v, axis=0)
    return np.array([A[bmask].mean(), A[imask].mean()])


# ---------------- normal level per recorder x clock slot ----------------
norm = {}
for (r, d, s), names in norm_clips.items():
    x = cell_db(names, 2)
    if x is not None:
        norm.setdefault((r, s), []).append(x)
norm = {k: np.median(v, axis=0) for k, v in norm.items() if len(v) >= 2}

rows = []
for (r, d, s), names in cell_clips.items():
    x = cell_db(names, 2)
    if x is None or (r, s) not in norm:
        continue
    y = x - norm[(r, s)]
    rows.append(dict(rec=r, day=d, slot=s, y_bird=y[0], y_insect=y[1]))
C = pd.DataFrame(rows)
if C.empty:
    sys.exit("No usable recorder-slots.")
C["moment"] = C.day.astype(str) + "_" + C.slot.astype(str)
C = C[C.groupby("moment").rec.transform("nunique") >= 2].reset_index(drop=True)
print(f"Usable recorder-slots: {len(C):,} in {C.moment.nunique():,} moments")
if len(C) < 50:
    sys.exit("Too few moments with two or more recorders measured.")

# signed minutes from each slot centre to the nearest event, for every (day, slot, SOURCE recorder)
days = sorted(C.day.unique())
slots = sorted(C.slot.unique())
di = {d: k for k, d in enumerate(days)}
si = {s: k for k, s in enumerate(slots)}
ri = {r: k for k, r in enumerate(RECS)}
DT = np.full((len(days), len(slots), len(RECS)), np.nan)
for d in days:
    Ed = E[E.day == d]
    for r in RECS:
        ts_ = Ed[Ed.rec == r].t
        if ts_.empty:
            continue
        evm = (ts_.dt.hour * 60 + ts_.dt.minute + ts_.dt.second / 60).values
        for s in slots:
            dt = (s + SLOT / 2) - evm                      # positive = after the event
            DT[di[d], si[s], ri[r]] = dt[np.argmin(np.abs(dt))]
C_d = C.day.map(di).values
C_s = C.slot.map(si).values
C_r = C.rec.map(ri).values
moment_codes = pd.factorize(C.moment)[0]

BINS = ["approach", "near", "after", "mid"]


def label(dt):
    lab = np.full(dt.shape, "far", dtype=object)
    a = np.abs(dt)
    lab[(dt >= -60) & (dt < -20)] = "approach"
    lab[a < 20] = "near"
    lab[(dt >= 20) & (dt < 60)] = "after"
    lab[(a >= 60) & (a < 120)] = "mid"
    lab[np.isnan(dt)] = "far"
    return lab


def demean(M):
    M = np.asarray(M, dtype=float)
    out = M.copy()
    sums = np.zeros((moment_codes.max() + 1,) + M.shape[1:])
    np.add.at(sums, moment_codes, M)
    cnt = np.bincount(moment_codes).astype(float)
    return out - (sums / cnt.reshape((-1,) + (1,) * (M.ndim - 1)))[moment_codes]


Yb, Yi = demean(C.y_bird.values), demean(C.y_insect.values)


def fit(labels):
    X = np.column_stack([(labels == b).astype(float) for b in BINS])
    Xd = demean(X)
    keep = Xd.std(axis=0) > 0
    beta = np.full(len(BINS), np.nan)
    if keep.any():
        coef_b = np.linalg.lstsq(Xd[:, keep], Yb, rcond=None)[0]
        coef_i = np.linalg.lstsq(Xd[:, keep], Yi, rcond=None)[0]
        bb, bi = beta.copy(), beta.copy()
        bb[keep], bi[keep] = coef_b, coef_i
        return bb, bi
    return beta, beta


real_lab = label(DT[C_d, C_s, C_r])
C["bin"] = real_lab
C["dt_min"] = DT[C_d, C_s, C_r]
C.to_csv(os.path.join(OUT, "same_moment_cells.csv"), index=False)
real_b, real_i = fit(real_lab)

print(f"Permutation test ({N_PERM} fake schedules)...")
perm_b, perm_i = [], []
for _ in range(N_PERM):
    P = np.array([rng5.permutation(len(RECS)) for _ in days])     # day x recorder -> source recorder
    lab = label(DT[C_d, C_s, P[C_d, C_r]])
    b, i_ = fit(lab)
    perm_b.append(b)
    perm_i.append(i_)
perm_b, perm_i = np.array(perm_b), np.array(perm_i)


def pvals(real, perm, k):
    pr = perm[:, k][~np.isnan(perm[:, k])]
    if np.isnan(real[k]) or len(pr) == 0:
        return np.nan, np.nan
    one = (1 + (pr <= real[k]).sum()) / (1 + len(pr))
    two = (1 + (np.abs(pr) >= abs(real[k])).sum()) / (1 + len(pr))
    return one, two


out = ["SAME MOMENT, ALL RECORDERS: are recorders near the team quieter than the others at that moment?", "=" * 92,
       f"{len(days)} days with team activity, {len(C):,} recorder-slots (10 min, 06:00-18:30) in {C.moment.nunique():,} moments.",
       "Level = recorder-slot minus that recorder's normal level at that clock time; compared only between recorders",
       "at the same moment. Reference = recorders more than 2 h from any team event that day.", "",
       "  slots per group: " + ", ".join(f"{b} {int((real_lab == b).sum())}" for b in BINS + ["far"]), ""]
desc = {"approach": "APPROACH (team arrives in 20-60 min)", "near": "NEAR (within 20 min of the team) ",
        "after": "AFTER (team left 20-60 min ago)   ", "mid": "MID (60-120 min from the team)    "}
for band, real, perm in [("BIRD 2-7 kHz", real_b, perm_b), ("INSECT 10-13 kHz", real_i, perm_i)]:
    out.append(band + "   (dB vs far recorders at the same moment; p from fake schedules)")
    for k, b in enumerate(BINS):
        one, two = pvals(real, perm, k)
        out.append(f"  {desc[b]}: {real[k]:+6.2f} dB   p(quieter)={one:.3f}   p(two-sided)={two:.3f}")
    out.append("")

for part, mask in [("AFTERNOON moments (12:00 or later)", C.slot.values >= 12 * 60), ("MORNING moments (before 12:00)", C.slot.values < 12 * 60)]:
    sub = mask
    if sub.sum() < 50:
        continue
    yb, yi = demean(C.y_bird.values)[sub], demean(C.y_insect.values)[sub]
    X = demean(np.column_stack([(real_lab == b).astype(float) for b in BINS]))[sub]
    keep = X.std(axis=0) > 0
    cb = np.full(len(BINS), np.nan); ci = cb.copy()
    cb[keep] = np.linalg.lstsq(X[:, keep], yb, rcond=None)[0]
    ci[keep] = np.linalg.lstsq(X[:, keep], yi, rcond=None)[0]
    out.append(f"{part}: bird " + ", ".join(f"{b} {cb[k]:+.2f}" for k, b in enumerate(BINS)) +
               " | insect " + ", ".join(f"{b} {ci[k]:+.2f}" for k, b in enumerate(BINS)))
out.append("")

# time course (descriptive): median within-moment difference vs far recorders, by 20-min bins of signed time
out.append("TIME COURSE (descriptive): recorder minus the far recorders at the same moment, median dB")
out.append("  minutes from team event at this recorder    n slots   bird     insect")
dtv = C.dt_min.values
far_mask = real_lab == "far"
mom = moment_codes
far_mean_b = pd.Series(C.y_bird.values[far_mask]).groupby(mom[far_mask]).mean()
far_mean_i = pd.Series(C.y_insect.values[far_mask]).groupby(mom[far_mask]).mean()
for lo in range(-120, 120, 20):
    sel = (dtv >= lo) & (dtv < lo + 20) & np.isin(mom, far_mean_b.index)
    if sel.sum() < 5:
        continue
    db_ = C.y_bird.values[sel] - far_mean_b.loc[mom[sel]].values
    di_ = C.y_insect.values[sel] - far_mean_i.loc[mom[sel]].values
    out.append(f"  {lo:+4d} to {lo + 20:+4d}                             {sel.sum():>5}   {np.median(db_):+6.2f}   {np.median(di_):+6.2f}")
out += ["",
        "How to read it:",
        "  - NEAR clearly negative with p(quieter) below 0.05 -> recorders where the team is are quieter than the other",
        "    recorders at the very same moment; weather, light and the evening chorus cannot cause that.",
        "  - APPROACH negative and low p -> the forest goes quiet before the team arrives (early warning).",
        "  - p comes from moving the team's real schedule to random other recorders 2000 times, so it already",
        "    accounts for chance; it is the one number to quote.",
        "  - Played-sound events count as team presence too (the team was there to play them)."]
txt = "\n".join(out)
print(txt)
open(os.path.join(OUT, "same_moment_results.txt"), "w").write(txt)
print(f"\nSaved to {OUT}")
