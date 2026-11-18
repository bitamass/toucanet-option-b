r"""
98_same_evening_control.py  —  run in Google Colab after scripts 92-97 (reads some audio; ~1 h, resumable)
--------------------------------------------------------------------------------------------------------
Script 97: the "quieter near people" result is driven by late-afternoon visits (16:40-17:24), when the
evening chorus builds. On those visits the chorus near the recorder was 5-14 dB weaker than on normal
afternoons. Is that the PEOPLE, or was it just a quiet EVENING (cloud, wind, light)?

Test: at the same time on the same day, look at the OTHER recorders the team did not visit
(no event there within 90 min). Weather and light are the same for all recorders.

For each human-presence visit, -30 to +30 min around the logged moment, each recorder vs its own
normal days at the same clock time (same normal days as before):
  VISITED  : the recorder the team was at
  CONTROL  : other recorders with no event within 90 min (60 if too few), up to 3 per visit
  VISITED minus CONTROL : same moment, same weather; negative = quieter only where people were

  - only the visited recorder is quieter -> the people
  - all recorders quieter that evening   -> the evening, not the people
Everything is also split into MORNING and AFTERNOON visits.

Results: same_evening_results.txt and same_evening.csv in toucanet_soundscape.
Clips cached in same_evening_cache.npz, so if Colab disconnects, just run it again.

Colab:  from google.colab import drive; drive.mount("/content/drive")
        %run /content/98_same_evening_control.py
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



MAX_CTRL, PER_STEP4 = 3, 6
ALL_RECS = sorted(by_rec)
CTRL_GAPS = [pd.Timedelta("90min"), pd.Timedelta("60min")]   # unvisited = no event there within this; fall back if too few


def quiet_at(rec, t, gap=pd.Timedelta("2h")):
    return not (E[E.rec == rec].t.sub(t).abs() < gap).any()


def n_with_ctrl(gap):
    n = 0
    for _, v in vis.iterrows():
        if any(r != v.rec and quiet_at(r, v.t0, gap) for r in ALL_RECS):
            n += 1
    return n


CTRL_GAP = CTRL_GAPS[0]
for gp in CTRL_GAPS:
    CTRL_GAP = gp
    k = n_with_ctrl(gp)
    print(f"  visits with an unvisited recorder (no event within {int(gp.total_seconds() / 60)} min): {k}")
    if k >= 8:
        break


def visited_db(i):
    return bands([feat["spec"][n] for st in MID for n in plan.get((i, st), []) if n in feat["spec"] and not human_tag.get(n, False)])


rng4 = np.random.default_rng(23)
cplan, need4, ctrl = {}, set(), {}
for _, v in vis.iterrows():
    offs = [Wn.loc[p, "t0"] - v.t0 for p in Wn[(Wn.kind == "placebo") & (Wn.vid == v.vid)].index]
    cands = []
    for r in ALL_RECS:
        if r == v.rec or not quiet_at(r, v.t0, CTRL_GAP):
            continue
        if covered(r, v.t0 - pd.Timedelta("30min"), v.t0 + pd.Timedelta("30min")) < 0.8:
            continue
        o_ok = [o for o in offs if quiet_at(r, v.t0 + o) and
                covered(r, v.t0 + o - pd.Timedelta("30min"), v.t0 + o + pd.Timedelta("30min")) >= 0.8]
        if o_ok:
            cands.append((r, o_ok))
    cands = [cands[k] for k in rng4.permutation(len(cands))[:MAX_CTRL]] if cands else []
    ctrl[v.vid] = cands
    for r, o_ok in cands:
        g = by_rec[r]
        for tag, t in [("day", v.t0)] + [(f"n{k}", v.t0 + o) for k, o in enumerate(o_ok)]:
            names = []
            for st in MID:
                a = t + pd.Timedelta(minutes=st)
                x = g[(g.ts >= a) & (g.ts < a + pd.Timedelta(minutes=STEP))]
                if len(x) >= 4:
                    names += list(x.clip_name.sample(min(PER_STEP4, len(x)), random_state=int(rng4.integers(1e9))))
            cplan[(v.vid, r, tag)] = names
            need4 |= set(names)
print(f"Control recorders: {sum(len(c) for c in ctrl.values())} across {sum(1 for c in ctrl.values() if c)} visits; clips: {len(need4):,}")

cache4 = os.path.join(OUT, "same_evening_cache.npz")
spec4 = {}
if os.path.exists(cache4):
    z4 = np.load(cache4, allow_pickle=True)
    spec4 = dict(zip(z4["names"], z4["vals"].astype("float32")))
reuse = dict(feat["spec"])
for fname in ["far_hours_cache.npz", "route_cache.npz"]:
    fp_ = os.path.join(OUT, fname)
    if os.path.exists(fp_):
        zz = np.load(fp_, allow_pickle=True)
        reuse.update(zip(zz["names"], zz["vals"].astype("float32")))
for n in need4:
    if n not in spec4 and n in reuse:
        spec4[n] = reuse[n]
todo = sorted(n for n in need4 if n not in spec4)
print(f"  {len(need4) - len(todo):,} already measured, {len(todo):,} to read")


def save4():
    np.savez(cache4 + ".tmp.npz", names=np.array(list(spec4.keys()), dtype=object),
             vals=np.array(list(spec4.values()), dtype="float16") if spec4 else np.zeros((0, 48), "float16"))
    os.replace(cache4 + ".tmp.npz", cache4)


if todo:
    print("Finding the clips (a few minutes)...")
    want, paths = set(todo), {}
    recs = sorted({r for c in ctrl.values() for r, _ in c})
    for rec in recs:
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
                    spec4[r[0]] = r[1]
            save4()
            print(f"  {min(b0 + 1000, len(names)):,}/{len(names):,}  ({(time.time() - t_start) / 60:.1f} min)")


def ctrl_db(key):
    return bands([spec4[n] for n in cplan.get(key, []) if n in spec4 and not human_tag.get(n, False)])


rows = []
for _, v in vis.iterrows():
    pls = list(Wn[(Wn.kind == "placebo") & (Wn.vid == v.vid)].index)
    dv = visited_db(v.name)
    dn = [x for x in (visited_db(p) for p in pls) if x is not None]
    visited = dv - np.median(dn, axis=0) if dv is not None and dn else np.array([np.nan, np.nan])
    cdiffs = []
    for r, o_ok in ctrl.get(v.vid, []):
        cd = ctrl_db((v.vid, r, "day"))
        cn = [x for x in (ctrl_db((v.vid, r, f"n{k}")) for k in range(len(o_ok))) if x is not None]
        if cd is not None and cn:
            cdiffs.append(cd - np.median(cn, axis=0))
    control = np.median(cdiffs, axis=0) if cdiffs else np.array([np.nan, np.nan])
    rows.append(dict(vid=v.vid, rec=v.rec, time=str(v.t0)[:16], part="afternoon" if v.t0.hour >= 12 else "morning",
                     n_ctrl=len(cdiffs), visited_bird=visited[0], visited_insect=visited[1],
                     control_bird=control[0], control_insect=control[1]))
R = pd.DataFrame(rows)
R["vmc_bird"] = R.visited_bird - R.control_bird
R["vmc_insect"] = R.visited_insect - R.control_insect
R.to_csv(os.path.join(OUT, "same_evening.csv"), index=False)


def test(a, word="quieter"):
    a = np.asarray(a, dtype=float)
    a = a[~np.isnan(a)]
    if len(a) < 4:
        return f"n={len(a)} (too few to test)" + (f"; values {', '.join(f'{x:+.1f}' for x in a)}" if len(a) else "")
    k = int((a < 0).sum())
    return (f"median {np.median(a):+.2f} dB, {word} in {k}/{len(a)}, sign p={stats.binomtest(k, len(a)).pvalue:.3f}, "
            f"Wilcoxon p={stats.wilcoxon(a).pvalue:.3f}")


out = ["SAME MOMENT, OTHER RECORDERS: is the quiet about the PEOPLE or about the EVENING?", "=" * 92,
       f"{(R.n_ctrl > 0).sum()} of {len(R)} visits have at least one unvisited recorder at the same time "
       f"(no event there within {int(CTRL_GAP.total_seconds() / 60)} min); up to {MAX_CTRL} per visit.",
       "All values: -30 to +30 min around the logged moment, each recorder minus its own normal days at that clock time,",
       "human-tagged clips removed (dB).", ""]
for part, sub in [("ALL VISITS", R), ("AFTERNOON visits (12:00 or later)", R[R.part == "afternoon"]),
                  ("MORNING visits (before 12:00)", R[R.part == "morning"])]:
    out.append(f"{part}  (n = {len(sub)})")
    for band, label in [("bird", "bird 2-7 kHz   "), ("insect", "insect 10-13 kHz")]:
        out.append(f"  {label} visited recorder        : {test(sub[f'visited_{band}'])}")
        out.append(f"  {label} unvisited, same time    : {test(sub[f'control_{band}'])}")
        out.append(f"  {label} visited minus unvisited : {test(sub[f'vmc_{band}'], 'quieter where people were')}")
    out.append("")
out.append("PER VISIT (dB)")
out.append("  time              recorder       part        bird: visited  unvisited  diff    insect: visited  unvisited  diff")
for r in R.sort_values("time").itertuples():
    out.append(f"  {r.time:<17} {r.rec:<14} {r.part:<10}  {r.visited_bird:+8.2f} {r.control_bird:+9.2f} {r.vmc_bird:+7.2f}"
               f"        {r.visited_insect:+8.2f} {r.control_insect:+9.2f} {r.vmc_insect:+7.2f}")
out += ["",
        "How to read it:",
        "  - 'visited minus unvisited' clearly negative (especially in the afternoon) -> only the recorder where people",
        "    were went quiet; the evening's weather and light were the same everywhere, so it is the people.",
        "  - unvisited recorders just as quiet -> it was a quiet evening, not the people.",
        "  - Unvisited recorders may still be within earshot of the team on some visits; that would make this test",
        "    conservative (it would hide part of a real effect, not create a false one)."]
txt = "\n".join(out)
print(txt)
open(os.path.join(OUT, "same_evening_results.txt"), "w").write(txt)
print(f"\nSaved to {OUT}")
