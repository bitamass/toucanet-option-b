r"""
90_focused_test.py  —  run in Google Colab after script 89 (reuses its measurements; minutes)
-------------------------------------------------------------------------------------------
One focused test of the early-warning lead from script 89, instead of reading a table:

  For each human-presence visit (anchored on the logged moment, as in script 89):
    before-change = mean 6-8 kHz level from 25 to 5 minutes before the logged moment
                    minus its baseline (40-30 min before)
    diff          = visit before-change minus the median before-change of its own normal days
  Test: how many visits are quieter (sign test), and a Wilcoxon signed-rank test.

The same is shown for the other pitch bands for context, and for the 5 minutes just before
logging (when the people themselves become audible). The 6-8 kHz band and the -25 to -5 window
were chosen after seeing script 89, so this is a check of that lead, not an independent proof.

Colab:  upload this file (script 89 must have run first), then
   from google.colab import drive; drive.mount("/content/drive")
   %run /content/90_focused_test.py
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


def window_change(i, sts):
    base = [level(i, b) for b in BASE_STEPS]
    vals = [level(i, s_) for s_ in sts]
    vals = [x for x in vals if x is not None]
    if any(b is None for b in base) or len(vals) < max(1, len(sts) - 1):
        return None
    return np.mean(vals, axis=0) - np.mean(base, axis=0)


names_g = [f"{a}-{b} kHz" for a, b in GROUPS]
WINDOWS = [("25 to 5 min before (main test)", [-25, -20, -15, -10]),
           ("5 min just before logging", [-5]),
           ("0 to 15 min after", [0, 5, 10])]
vis = Wn[(Wn.kind == "visit") & (Wn.group == "presence")]
out = ["FOCUSED TEST: does the soundscape change before a logged human presence?", "=" * 78,
       "Per visit: change since baseline (40-30 min before), minus the same on its own normal days (dB).", ""]
rows = []
for wname, sts in WINDOWS:
    diffs = []
    for i, v in vis.iterrows():
        pls = Wn[(Wn.kind == "placebo") & (Wn.vid == v.vid)].index
        cv = window_change(i, sts)
        cp = [c for c in (window_change(j, sts) for j in pls) if c is not None]
        if cv is not None and cp:
            d = cv - np.median(cp, axis=0)
            diffs.append(d)
            rows.append(dict(window=wname, rec=v.rec, t0=v.t0, **{g: x for g, x in zip(names_g, d)}))
    diffs = np.array(diffs)
    out.append(f"{wname}   ({len(diffs)} visits)")
    if len(diffs) < 5:
        out += ["  too few visits", ""]
        continue
    for k, g in enumerate(names_g):
        x = diffs[:, k]
        dn, up = int((x < 0).sum()), int((x > 0).sum())
        p_sign = stats.binomtest(dn, dn + up, 0.5).pvalue if dn + up else np.nan
        p_w = stats.wilcoxon(x).pvalue if np.any(x != 0) else np.nan
        main = "  <- main test" if (g.startswith("6-8") and sts == WINDOWS[0][1]) else ""
        out.append(f"  {g:<10} median {np.median(x):+6.2f} dB   quieter in {dn}/{dn + up}   "
                   f"sign p={p_sign:.3f}   Wilcoxon p={p_w:.3f}{main}")
    out.append("")
pd.DataFrame(rows).to_csv(os.path.join(OUT, "focused_test_visits.csv"), index=False)
out += ["How to read it:",
        "  - Main test: 6-8 kHz, 25 to 5 min before. p < 0.05 with most visits quieter supports the early-warning lead",
        "    (still post hoc, since this band and window came from script 89).",
        "  - 0-1 kHz in the last 5 min before logging should rise: that is the people becoming audible (a sanity check)."]
txt = "\n".join(out)
print(txt)
open(os.path.join(OUT, "focused_test_results.txt"), "w").write(txt)
print(f"\nSaved to {OUT}")
