r"""
97_robustness_and_species.py  —  run in Google Colab after scripts 92-95 (no audio; a few minutes)
------------------------------------------------------------------------------------------------
Two quick follow-ups on the far-hours result (script 95), using only what is already saved:

 1. ROBUSTNESS: does "quieter near the visit than at far hours of the same day" depend on one or two
    visit days? Per-visit table, plus the result recomputed leaving out each visit day in turn.
    (reads far_hours.csv from script 95)

 2. WHICH BIRDS: BirdNET detections (confidence >= 0.25, human tags removed), all clips, per species:
    how often the species is detected near the visit (-30 to +30 min) vs the far hours of the same day,
    each relative to the same clock time on that visit's normal days.
    effect = (near - its normal days) - (far - its normal days); negative = the species goes quiet near people.
    Same visits, normal days and far hours as script 95.

Results: robustness_species_results.txt, species_near_far.csv in toucanet_soundscape.

Colab:  from google.colab import drive; drive.mount("/content/drive")
        %run /content/97_robustness_and_species.py
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


out = ["ROBUSTNESS AND SPECIES (follow-up to script 95)", "=" * 92, ""]


def wil(a):
    a = np.asarray(a, dtype=float)
    a = a[~np.isnan(a)]
    return stats.wilcoxon(a).pvalue if len(a) >= 4 and np.any(a != 0) else np.nan


# ---------------- 1. robustness ----------------
fh = os.path.join(OUT, "far_hours.csv")
if os.path.exists(fh):
    F = pd.read_csv(fh)
    F = F.dropna(subset=["far_bird", "near_bird"]).copy()
    vdate = dict(zip(vis.vid, vis.t0))
    F["date"] = [str(vdate.get(v, ""))[:16] for v in F.vid]
    F["bird_nf"] = F.near_bird - F.far_bird
    F["insect_nf"] = F.near_insect - F.far_insect
    out.append("1. ROBUSTNESS: near the visit minus far hours, same day (dB; negative = quieter near the visit)")
    out.append("   visit day          recorder        bird near  bird far   bird near-far   insect near-far")
    for r in F.sort_values("bird_nf").itertuples():
        out.append(f"   {r.date:<18} {r.rec:<14}  {r.near_bird:+7.2f}  {r.far_bird:+7.2f}    {r.bird_nf:+8.2f}        {r.insect_nf:+8.2f}")
    for band in ["bird", "insect"]:
        a = F[f"{band}_nf"].values
        loo = [(F.iloc[i].date, np.median(np.delete(a, i)), (np.delete(a, i) < 0).sum(), wil(np.delete(a, i))) for i in range(len(a))]
        worst = max(loo, key=lambda x: x[3] if not np.isnan(x[3]) else -1)
        out.append(f"   {band}: all {len(a)} days: median {np.median(a):+.2f}, quieter near in {(a < 0).sum()}/{len(a)}, Wilcoxon p={wil(a):.3f}")
        out.append(f"   {band}: leaving out one day at a time: median {min(x[1] for x in loo):+.2f} to {max(x[1] for x in loo):+.2f}, "
                   f"Wilcoxon p {min(x[3] for x in loo):.3f} to {max(x[3] for x in loo):.3f} "
                   f"(weakest without {worst[0]}: {worst[2]}/{len(a) - 1} quieter, p={worst[3]:.3f})")
    out.append("   (a result that stays below p = 0.05 whichever single day is dropped does not hinge on one day)")
    out.append("")
else:
    out += ["1. ROBUSTNESS: far_hours.csv not found (run script 95 first).", ""]

# ---------------- 2. species ----------------
print("Parsing BirdNET detections per clip...")


def detected(sp, cf):
    s, c = as_list(sp), as_list(cf)
    res = set()
    for name, conf in zip(s, c):
        try:
            if float(conf) >= CONF and not any(h in str(name).lower() for h in HUMAN_TAGS):
                res.add(str(name).split("_")[-1].strip())
        except (TypeError, ValueError):
            pass
    return res


df["det"] = [detected(a, b) for a, b in zip(df["species"], df["confidence"])]
by_rec = {r: g for r, g in df.groupby("Recorder")}


def rates(rec, t, half=pd.Timedelta("30min")):
    g = by_rec[rec]
    x = g[(g.ts >= t - half) & (g.ts < t + half)]
    if len(x) < 0.5 * 2 * half.total_seconds() / 3:
        return None
    cnt = {}
    for s in x.det:
        for k in s:
            cnt[k] = cnt.get(k, 0) + 1
    r = {k: v / len(x) for k, v in cnt.items()}
    r["__any_bird__"] = float(np.mean([len(s) > 0 for s in x.det]))
    r["__n__"] = len(x)
    return r


def rel(rec, t, offsets):
    """rates at t minus the median over normal-day times."""
    rv = rates(rec, t)
    rp = [x for x in (rates(rec, t + o) for o in offsets) if x is not None]
    if rv is None or not rp:
        return None
    keys = set(rv) | set().union(*[set(x) for x in rp])
    return {k: rv.get(k, 0.0) - np.median([x.get(k, 0.0) for x in rp]) for k in keys if k != "__n__"}


per_visit, totals = [], {}
for _, v in vis.iterrows():
    pairs = [p for p in far_pairs if p["vid"] == v.vid]
    if not pairs:
        continue
    offs = [Wn.loc[p, "t0"] - v.t0 for p in Wn[(Wn.kind == "placebo") & (Wn.vid == v.vid)].index]
    near = rel(v.rec, v.t0, offs)
    fars = [rel(v.rec, p["f"], [fp - p["f"] for _, fp in p["placebos"]]) for p in pairs]
    fars = [x for x in fars if x is not None]
    if near is None or not fars:
        continue
    keys = set(near) | set().union(*[set(x) for x in fars])
    eff = {k: near.get(k, 0.0) - np.mean([x.get(k, 0.0) for x in fars]) for k in keys}
    per_visit.append(eff)
    for t in [v.t0] + [p["f"] for p in pairs]:
        r = rates(v.rec, t)
        if r:
            for k, val in r.items():
                if not k.startswith("__"):
                    totals[k] = totals.get(k, 0) + val * r["__n__"]
nv = len(per_visit)
out.append(f"2. WHICH BIRDS GO QUIET NEAR PEOPLE? ({nv} visit days; share of 3-s clips with the species detected)")
out.append("   effect = (near - normal days) - (far hours - normal days), in percentage points of clips")
rows = []
for k in sorted(set().union(*[set(e) for e in per_visit]) if per_visit else []):
    a = np.array([e.get(k, 0.0) for e in per_visit]) * 100
    det_n = totals.get(k, 0) if k != "__any_bird__" else 1e9
    if det_n < 30 and k != "__any_bird__":
        continue
    nz = a[a != 0]
    p = stats.wilcoxon(nz).pvalue if len(nz) >= 4 else np.nan
    rows.append(dict(species=k, clips_detected=int(det_n) if k != "__any_bird__" else np.nan, median_effect_pp=np.median(a),
                     mean_effect_pp=a.mean(), quieter_near=int((a < 0).sum()), louder_near=int((a > 0).sum()), wilcoxon_p=p))
S = pd.DataFrame(rows)
if len(S):
    sp = S[S.species != "__any_bird__"].copy()
    ok = sp.wilcoxon_p.notna()
    q = np.full(len(sp), np.nan)
    if ok.sum():
        pv = sp.loc[ok, "wilcoxon_p"].values
        order = np.argsort(pv)
        m = len(pv)
        adj = np.minimum.accumulate((pv[order] * m / np.arange(1, m + 1))[::-1])[::-1]
        qq = np.empty(m)
        qq[order] = np.minimum(adj, 1)
        q[np.where(ok)[0]] = qq
    sp["fdr_q"] = q
    S = pd.concat([S[S.species == "__any_bird__"], sp])
    S.to_csv(os.path.join(OUT, "species_near_far.csv"), index=False)
    ab = S[S.species == "__any_bird__"]
    if len(ab):
        r = ab.iloc[0]
        out.append(f"   ANY BIRD: median {r.median_effect_pp:+.2f} pp, fewer near the visit in {r.quieter_near}/{nv}, Wilcoxon p={r.wilcoxon_p:.3f}")
    out.append("")
    out.append("   species (detected in >= 30 clips)          median   mean    fewer/more near   Wilcoxon p   FDR q")
    for r in sp.sort_values(["median_effect_pp", "mean_effect_pp"]).itertuples():
        out.append(f"   {r.species[:40]:<40} {r.median_effect_pp:+6.2f} {r.mean_effect_pp:+6.2f}      {r.quieter_near:>2}/{r.louder_near:<2}"
                   f"          {r.wilcoxon_p:6.3f}     {r.fdr_q:5.2f}")
out += ["",
        "How to read it:",
        "  1: if the near-minus-far result stays clearly negative and p stays low whichever day is left out,",
        "     it is not driven by one unusual day; the per-day table shows which days drive it.",
        "  2: species at the top (negative effect, most visits 'fewer near') are the ones that go quiet near people,",
        "     beyond the normal daily rhythm and the day's conditions. FDR q < 0.1 survives testing many species;",
        "     with 13 visit days, expect few to pass, so treat the list as leads for listening."]
txt = "\n".join(out)
print(txt)
open(os.path.join(OUT, "robustness_species_results.txt"), "w").write(txt)
print(f"\nSaved to {OUT}")
