r"""
89_event_anchored.py  —  run in Google Colab (reads a sample of audio; ~1.5-2 h, resumable)
-------------------------------------------------------------------------------------------
New information (Daan, Nov 2026): each disturbance was logged in the field as ONE moment
(DeepData Capture app). The labelled windows in the dataset were added afterwards:
  human presence -> -600 s to +600 s around the logged moment
  played sound   ->    0 s to +300 s after it
So the only real anchor is the logged moment (Sim Relative Time = 0). Earlier scripts used
the start of the labelled window as "arrival", which for human presence is ~10 min early.

This script lines everything up on the logged moment (time 0):
  - events: every (recorder, logged moment, type) recovered from Sim Type / Sim Relative Time
  - groups of events at a recorder less than 40 min apart are one visit, anchored on its
    FIRST event, so the period before the anchor is free of other events at that recorder
  - visits are split by the type of that first event: HUMAN PRESENCE vs PLAYED SOUND
  - normal days: same recorder, same clock time, other days with no event within 2 h
  - time course from 40 min before to 40 min after, 5-min steps, 8 pitch groups,
    each step compared with the visit's own baseline (40-30 min before) and with normal days
  - BirdNET (no audio): Great Tinamou and any-bird share over the same time course

Results: event_anchored_results.txt, event_anchored.csv, event_anchored.png in toucanet_soundscape.
Measured clips are cached in event_anchored_cache.csv, so if Colab disconnects, run it again.

Colab:  upload this file, then
   from google.colab import drive; drive.mount("/content/drive")
   %run /content/89_event_anchored.py
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
PER_STEP, MAX_PLACEBO, THREADS, SEED = 12, 3, 32, 0
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
cols = {"clip_name", "Recorder", "Timestamp", "Sim Type", "Sim Relative Time", "species", "confidence"}
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

# ---------------- B. BirdNET time course (no audio) ----------------
def birdset(row):
    sp, cf = as_list(row["species"]), as_list(row["confidence"])
    if len(cf) == 1 and len(sp) > 1:
        cf = cf * len(sp)
    s = set()
    for a, c in zip(sp, cf):
        try:
            if float(c) >= CONF:
                s.add(str(a).split("_")[-1])
        except (TypeError, ValueError):
            pass
    return s


print("Parsing BirdNET species...")
df["birds"] = df.apply(birdset, axis=1)
df["any_bird"] = (df.birds.str.len() > 0).astype(float)
df["tinamou"] = df.birds.apply(lambda s: any("tinamou" in b.lower() for b in s)).astype(float)
by_rec = {r: g for r, g in df.groupby("Recorder")}
steps = list(range(-SPAN, SPAN, STEP))

# ---------------- A. sample clips and measure pitch groups ----------------
plan, need = {}, set()
for w in Wn.itertuples():
    g = by_rec[w.rec]
    for st in steps:
        a = w.t0 + pd.Timedelta(minutes=st)
        x = g[(g.ts >= a) & (g.ts < a + pd.Timedelta(minutes=STEP))]
        plan[(w.Index, st, "birdnet")] = x[["any_bird", "tinamou"]].mean() if len(x) >= 10 else None
        if len(x) >= 8:
            pick = list(x.clip_name.sample(min(PER_STEP, len(x)), random_state=int(rng.integers(1e9))))
            plan[(w.Index, st)] = pick
            need |= set(pick)
cache_file = os.path.join(OUT, "event_anchored_cache.csv")
gcols = [f"g{a}_{b}" for a, b in GROUPS]
cache = pd.read_csv(cache_file) if os.path.exists(cache_file) else pd.DataFrame(columns=["clip_name"] + gcols)
todo = sorted(need - set(cache.clip_name))
print(f"Clips: {len(need):,} needed, {len(need) - len(todo):,} already measured, {len(todo):,} to read")

if todo:
    print("Finding the clips (a few minutes)...")
    want, paths = set(todo), {}
    for rec in sorted(Wn.rec.unique()):
        for root, _, files in os.walk(os.path.join(AUDIO, rec.replace("_", " "))):
            for f in files:
                if f in want:
                    paths[f] = os.path.join(root, f)
    print(f"  found {len(paths):,} of {len(todo):,}")

    def groups(name):
        try:
            y, sr = load(paths[name])
            f, P = signal.welch(y, fs=sr, nperseg=2048)
            return [name] + [float(P[(f >= a * 1000) & (f < b * 1000)].mean()) for a, b in GROUPS]
        except Exception:
            return None

    t_start, rows = time.time(), []
    names = [n for n in todo if n in paths]
    with ThreadPoolExecutor(THREADS) as ex:
        for k, r in enumerate(ex.map(groups, names), 1):
            if r:
                rows.append(r)
            if k % 2000 == 0 or k == len(names):
                cache = pd.concat([cache, pd.DataFrame(rows, columns=["clip_name"] + gcols)], ignore_index=True)
                cache.to_csv(cache_file, index=False)
                rows = []
                print(f"  {k:,}/{len(names):,}  ({(time.time() - t_start) / 60:.1f} min)")
spec = {r[0]: np.array(r[1:], dtype=float) for r in cache.itertuples(index=False)}


def level(i, st):
    arr = [spec[n] for n in plan.get((i, st), []) if n in spec]
    return 10 * np.log10(np.mean(arr, axis=0) + 1e-15) if len(arr) >= 6 else None


# ---------------- compare visits with their normal days ----------------
names_g = [f"{a}-{b}k" for a, b in GROUPS]
rows, out = [], ["SOUNDSCAPE AROUND THE LOGGED MOMENT  (time 0 = disturbance logged in the field)", "=" * 92,
                 "Change since the baseline (40-30 min before), beyond normal days; median dB. "
                 "* = same direction in at least 75% of visits.", ""]
for group, label in [("presence", "HUMAN PRESENCE (first event of the visit)"),
                     ("sound", "PLAYED SOUND (vehicle, chainsaw, gunshot...)")]:
    vis = Wn[(Wn.kind == "visit") & (Wn.group == group)]
    res = {st: [] for st in steps}
    bres = {st: [] for st in steps}
    for i, v in vis.iterrows():
        pls = Wn[(Wn.kind == "placebo") & (Wn.vid == v.vid)].index
        if len(pls) == 0:
            continue
        d = {}
        for st in steps:
            lv = level(i, st)
            lp = [x for x in (level(j, st) for j in pls) if x is not None]
            if lv is not None and lp:
                d[st] = lv - np.median(lp, axis=0)
            bv = plan.get((i, st, "birdnet"))
            bp = [plan.get((j, st, "birdnet")) for j in pls]
            bp = [x for x in bp if x is not None]
            if bv is not None and bp:
                bres[st].append((bv - pd.concat(bp, axis=1).median(axis=1)).values)
        if all(b in d for b in BASE_STEPS):
            base = (d[BASE_STEPS[0]] + d[BASE_STEPS[1]]) / 2
            for st in d:
                res[st].append(d[st] - base)
    n_ok = max((len(v_) for v_ in res.values()), default=0)
    out.append(f"{label}: {n_ok} visits with normal days")
    if n_ok < 4:
        out += ["  too few visits to show", ""]
        continue
    out.append("  minutes     " + "".join(f"{g:>9}" for g in names_g) + "    tinamou  any bird")
    for st in steps:
        arr = np.array(res[st])
        if len(arr) < 4:
            continue
        med = np.median(arr, axis=0)
        fu = (arr > 0).mean(axis=0)
        cells = "".join(f"{m:+8.2f}{'*' if (f >= .75 or f <= .25) and st not in BASE_STEPS else ' '}" for m, f in zip(med, fu))
        b = np.array(bres[st])
        bt = f"{np.median(b[:, 1]):+8.3f}  {np.median(b[:, 0]):+8.3f}" if len(b) >= 4 else ""
        mark = "  <- logged" if st == 0 else ""
        out.append(f"  {st:+3d} to {st + STEP:+3d} {cells} {bt}{mark}")
        for gname, m, f in zip(names_g, med, fu):
            rows.append(dict(group=group, step_start=st, band=gname, median_change_db=m, share_up=f, visits=len(arr)))
    out.append("")
pd.DataFrame(rows).to_csv(os.path.join(OUT, "event_anchored.csv"), index=False)
out += ["How to read it:",
        "  - HUMAN PRESENCE rows before 0: a change there, with 0-1k and 1-2k flat, is the forest reacting before",
        "    the person logged their presence (early warning). Note people were likely within earshot",
        "    in the minutes just before 0, so the earliest steps are the cleanest test.",
        "  - PLAYED SOUND rows: nothing should change before 0 except from the person carrying the sound;",
        "    a change right after 0 is the reaction to the sound itself.",
        "  - tinamou / any bird: share of clips with a BirdNET detection, change since baseline beyond normal days."]
txt = "\n".join(out)
print(txt)
open(os.path.join(OUT, "event_anchored_results.txt"), "w").write(txt)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    R = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, group in zip(axes, ["presence", "sound"]):
        for band, color in [("6-8k", "#C8641C"), ("0-1k", "#7A8C83"), ("11-14k", "#3B6EA8")]:
            x = R[(R.group == group) & (R.band == band)]
            if len(x):
                ax.plot(x.step_start + STEP / 2, x.median_change_db, lw=2.5, color=color, label=band)
        ax.axvline(0, color="#1F3A2E", ls="--", lw=1.5)
        ax.axhline(0, color="#999999", lw=0.8)
        ax.set_title("Human presence" if group == "presence" else "Played sound")
        ax.set_xlabel("minutes from the logged moment")
    axes[0].set_ylabel("change vs baseline, beyond normal days (dB)")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "event_anchored.png"), dpi=150)
    print("Saved event_anchored.png")
except Exception as e:
    print(f"(plot skipped: {e})")
print(f"\nSaved to {OUT}")
