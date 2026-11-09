r"""
91_positive_control.py  —  run in Google Colab (reads ~10,000 clips, ~1 h, resumable)
------------------------------------------------------------------------------------
Positive control: can our measures see an animal reaction at all?
A gunshot (and probably a chainsaw or vehicle) should make animals react: alarm calls,
flushing, or going quiet. So we line up on the logged moment each sound was played
and look minute by minute from 5 minutes before to 10 minutes after.

Baseline = the 5 minutes just before the sound (people are already there then, so this
isolates the reaction to the sound itself, not to the people).
For each sound event and each minute:
  - level in 8 pitch groups minus its baseline (dB)
  - "quiet share": share of clips quieter than the 25th percentile of that event's baseline
    clips (a measure of sudden silence)
  - BirdNET: share of clips with any bird
Across events, per sound type: median change, how many events went the same way, sign test.

Note: right after a chainsaw or vehicle the sound itself fills the recording (and can mask
birds); gunshots are brief, so they are the cleanest test of the animals' reaction.

Colab:  upload this file, then
   from google.colab import drive; drive.mount("/content/drive")
   %run /content/91_positive_control.py
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


# ---------------- sound events ----------------
S_ = E[E.kind == "sound"].copy()
S_["stype"] = S_.type.str.lower().map(lambda t: "gunshot" if "gun" in t else ("chainsaw" if "chain" in t else ("vehicle" if "vehic" in t else t)))
print("Parsing BirdNET species...")
def any_bird(v):
    for c in as_list(v):
        try:
            if float(c) >= CONF:
                return 1.0
        except (TypeError, ValueError):
            pass
    return 0.0


df["any_bird"] = df["confidence"].apply(any_bird)
by_rec = {r: g for r, g in df.groupby("Recorder")}
MINUTES = list(range(-5, 10))
BASE_MIN = [-5, -4, -3]   # baseline: minutes -5 to -2, before the sound (logged times can lag the sound by ~1 min)
plan, need = {}, set()
for k, e in S_.reset_index(drop=True).iterrows():
    g = by_rec.get(e.rec)
    if g is None:
        continue
    for m in MINUTES:
        a = e.t + pd.Timedelta(minutes=m)
        x = g[(g.ts >= a) & (g.ts < a + pd.Timedelta(minutes=1))]
        if len(x) >= 10:
            plan[(k, m)] = list(x.clip_name)
            plan[(k, m, "bird")] = x.any_bird.mean()
            need |= set(x.clip_name)
S_ = S_.reset_index(drop=True)
print(f"Sound events: {len(S_)} ({dict(S_.stype.value_counts())})   clips to measure: {len(need):,}")

cache_file = os.path.join(OUT, "positive_control_cache.csv")
gcols = [f"g{a}_{b}" for a, b in GROUPS] + ["total"]
cache = pd.read_csv(cache_file) if os.path.exists(cache_file) else pd.DataFrame(columns=["clip_name"] + gcols)
todo = sorted(need - set(cache.clip_name))
print(f"  {len(need) - len(todo):,} already measured, {len(todo):,} to read")
if todo:
    print("Finding the clips (a few minutes)...")
    want, paths = set(todo), {}
    for rec in sorted(S_.rec.unique()):
        for root, _, files in os.walk(os.path.join(AUDIO, rec.replace("_", " "))):
            for f in files:
                if f in want:
                    paths[f] = os.path.join(root, f)
    print(f"  found {len(paths):,} of {len(todo):,}")

    def groups(name):
        try:
            y, sr = load(paths[name])
            f, P = signal.welch(y, fs=sr, nperseg=2048)
            return [name] + [float(P[(f >= a * 1000) & (f < b * 1000)].mean()) for a, b in GROUPS] + [float(P.mean())]
        except Exception:
            return None

    t_start, rows = time.time(), []
    names = [n for n in todo if n in paths]
    with ThreadPoolExecutor(THREADS) as ex:
        for k, r in enumerate(ex.map(groups, names), 1):
            if r:
                rows.append(r)
            if k % 2000 == 0 or k == len(names):
                new = pd.DataFrame(rows, columns=["clip_name"] + gcols)
                cache = new if len(cache) == 0 else pd.concat([cache, new], ignore_index=True)
                cache.to_csv(cache_file, index=False)
                rows = []
                print(f"  {k:,}/{len(names):,}  ({(time.time() - t_start) / 60:.1f} min)")
spec = {r[0]: np.array(r[1:], dtype=float) for r in cache.itertuples(index=False)}

# ---------------- reaction per event ----------------
names_g = [f"{a}-{b}k" for a, b in GROUPS]
res = {}          # (stype, minute) -> list of arrays [8 groups dB change, quiet share change, bird change]
for k, e in S_.iterrows():
    base_clips = [n for m in BASE_MIN for n in plan.get((k, m), []) if n in spec]
    if len(base_clips) < 30:
        continue
    B = np.array([spec[n] for n in base_clips])
    base_db = np.median(10 * np.log10(B[:, :8] + 1e-15), axis=0)   # median clip: one loud clip cannot dominate
    thr = np.percentile(B[:, 8], 25)
    base_q = (B[:, 8] < thr).mean()
    base_bird = np.nanmean([plan.get((k, m, "bird"), np.nan) for m in BASE_MIN])
    for m in MINUTES:
        cl = [n for n in plan.get((k, m), []) if n in spec]
        if len(cl) < 10:
            continue
        X = np.array([spec[n] for n in cl])
        d = list(np.median(10 * np.log10(X[:, :8] + 1e-15), axis=0) - base_db)
        d.append((X[:, 8] < thr).mean() - base_q)
        d.append(plan.get((k, m, "bird"), np.nan) - base_bird)
        res.setdefault((e.stype, m), []).append(d)

out = ["POSITIVE CONTROL: do animals visibly react to a played sound?", "=" * 96,
       "Minute by minute around the logged sound (0 = played). Change vs minutes -5 to -2 (median clip).",
       "Columns: pitch groups (median dB), quiet = share of very quiet clips, bird = BirdNET any-bird share.",
       "* = same direction in at least 75% of events.", ""]
rows = []
for stype in ["gunshot", "chainsaw", "vehicle"]:
    nev = max((len(res.get((stype, m), [])) for m in MINUTES), default=0)
    out.append(f"{stype.upper()}  ({nev} events)")
    if nev < 4:
        out += ["  too few events", ""]
        continue
    out.append("  minute  " + "".join(f"{g:>8}" for g in names_g) + "    quiet     bird")
    for m in MINUTES:
        arr = np.array(res.get((stype, m), []), dtype=float)
        if len(arr) < 4:
            continue
        med = np.nanmedian(arr, axis=0)
        fu = np.nanmean(arr > 0, axis=0)
        cells = "".join(f"{x:+7.2f}{'*' if (f >= .75 or f <= .25) and m >= 0 else ' '}" for x, f in zip(med[:8], fu[:8]))
        q = f"{med[8]:+8.3f}{'*' if (fu[8] >= .75 or fu[8] <= .25) and m >= 0 else ' '}"
        b = f"{med[9]:+8.3f}{'*' if (fu[9] >= .75 or fu[9] <= .25) and m >= 0 else ' '}"
        out.append(f"  {m:+3d}    {cells} {q} {b}{'  <- sound' if m == 0 else ''}")
        rows.append(dict(stype=stype, minute=m, events=len(arr), **{g: x for g, x in zip(names_g + ['quiet', 'bird'], med)}))
    # one summary test per type: minutes +2..+6 (after the sound itself), any-bird and quiet share
    for col, idx in [("bird", 9), ("quiet", 8), ("6-8k", 4)]:
        per = []
        for e_i in range(nev):
            vals = [res[(stype, m)][e_i][idx] for m in range(2, 7) if (stype, m) in res and e_i < len(res[(stype, m)])]
            if vals:
                per.append(np.nanmean(vals))
        per = pd.Series(per, dtype=float).dropna()
        if len(per) >= 4:
            up, dn = int((per > 0).sum()), int((per < 0).sum())
            p = stats.binomtest(up, up + dn, 0.5).pvalue if up + dn else np.nan
            out.append(f"  minutes +2 to +6, {col:<5}: median {per.median():+.3f}   up in {up}/{up + dn}   sign p={p:.3f}")
    out.append("")
pd.DataFrame(rows).to_csv(os.path.join(OUT, "positive_control.csv"), index=False)
out += ["How to read it:",
        "  - GUNSHOT rows are the cleanest: a clear change after minute 0 in 'bird' or 'quiet' (with stars)",
        "    means our measures CAN see an animal reaction; then we know its shape and size.",
        "  - Minute 0-1 includes the sound itself; read from minute +1 or +2 onwards.",
        "  - For chainsaw and vehicle, the sound may keep playing and mask birds, so changes there are harder to read.",
        "  - No change even after gunshots -> these measures cannot capture reactions; we need finer ones."]
txt = "\n".join(out)
print(txt)
open(os.path.join(OUT, "positive_control_results.txt"), "w").write(txt)
print(f"\nSaved to {OUT}")
