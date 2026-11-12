r"""
93_model_followup.py  —  run in Google Colab AFTER script 92 (no audio; a few minutes)
--------------------------------------------------------------------------------------
Script 92 found that a simple spectrum model, using only sound above 2 kHz and no human-tagged
clips, tells visit periods from normal days on visits it never saw (AUC ~0.69, higher in 10-12 of 15
visits). But the score is raised over the WHOLE hour around the visit, not just when people are there.
That fits three explanations, which this script separates:

 1. SHAPE  - Is the score higher in the middle of the visit (-10 to +10 min) than at its edges
             (-40 to -30 and +30 to +40 min)? This compares a visit with ITSELF, so anything that is
             the same all morning on a visit day (weather of the day, season, date) cancels out.
             A reaction to people should peak in the middle; a "visit-day" difference stays flat.
 2. WEATHER - Were visit hours warmer, drier or windier than the normal days, and does the model
             score follow those differences across visits? (hourly weather, shifted as in script 87)
 3. WHAT   - Which pitch bands differ between visit periods and normal days (above 2 kHz)?

Also applies a Holm correction to the eight tests reported by script 92.
Reuses script 92's features (model_features_cache.npz) and the same visits, normal days and clips.

Colab:  from google.colab import drive; drive.mount("/content/drive")
        %run /content/93_model_followup.py
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

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

PRESENT = (0, 5)
vis = Wn[(Wn.kind == "visit") & (Wn.group == "presence")]
vis = vis[[((Wn.kind == "placebo") & (Wn.vid == v)).any() for v in vis.vid]]
print(f"Human-presence visits with normal days: {len(vis)}")


def vector(key, n):
    if key == "spec_full":
        return feat["spec"].get(n)
    if key == "spec_high":
        s = feat["spec"].get(n)
        return None if s is None or human_tag.get(n, False) else s[HIGH_BANDS]
    if key == "perch_high":
        return None if human_tag.get(n, False) else feat.get("phigh", {}).get(n)


def clips_of(i, st, key):
    return [v for v in (vector(key, n) for n in plan.get((i, st), [])) if v is not None]


def lovo(key):
    """Leave-one-visit-out scores: {vid: {step: visit minus normal days}}, as in script 92."""
    per = {}
    for _, v in vis.iterrows():
        i, vid = v.name, v.vid
        Xp, Xn = [], []
        for j, tv in vis[vis.vid != vid].iterrows():
            for st in PRESENT:
                Xp += clips_of(j, st, key)
            for p in Wn[(Wn.kind == "placebo") & (Wn.vid == tv.vid)].index:
                for st in steps:
                    Xn += clips_of(p, st, key)
        if len(Xp) < 20 or len(Xn) < 20:
            continue
        X = np.vstack(Xp + Xn).astype("float64")
        y = np.r_[np.ones(len(Xp)), np.zeros(len(Xn))]
        pca = PCA(n_components=min(20, X.shape[1]), random_state=0).fit(X)
        sc = StandardScaler().fit(pca.transform(X))
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(pca.transform(X)), y)
        score = lambda A: clf.decision_function(sc.transform(pca.transform(np.vstack(A).astype("float64"))))
        pls = Wn[(Wn.kind == "placebo") & (Wn.vid == vid)].index
        d = {}
        for st in steps:
            cv_ = clips_of(i, st, key)
            sp = [score(c).mean() for c in (clips_of(p, st, key) for p in pls) if len(c) >= 4]
            if len(cv_) >= 4 and sp:
                d[st] = score(cv_).mean() - np.median(sp)
        per[vid] = d
    return per


def test(a):
    a = np.asarray(a, dtype=float)
    a = a[~np.isnan(a)]
    if len(a) < 4:
        return f"n={len(a)} (too few)"
    k = int((a > 0).sum())
    return (f"median {np.median(a):+.2f}, higher in {k}/{len(a)}, sign p={stats.binomtest(k, len(a)).pvalue:.3f}, "
            f"Wilcoxon p={stats.wilcoxon(a).pvalue:.3f}")


def avg(d, sts):
    x = [d[s] for s in sts if s in d]
    return np.mean(x) if len(x) >= max(1, len(sts) - 1) else np.nan


CENTRE, EDGE_B, EDGE_A, MID = (-10, -5, 0, 5), (-40, -35), (30, 35), tuple(range(-30, 30, 5))
out = ["FOLLOW-UP ON THE MODEL TIME COURSE (script 92)", "=" * 92, ""]

# ---------------- 0. Holm correction for script 92 ----------------
p92 = os.path.join(OUT, "model_timecourse_results.txt")
if os.path.exists(p92):
    import re
    found, model = [], None
    for line in open(p92):
        if re.match(r"^(Perch|Spectrum)", line):
            model = line.split("(")[0].strip()
        m = re.search(r"^\s+(people present|EARLY).*Wilcoxon p=([0-9.]+)", line)
        if m and model:
            found.append((f"{model} | {m.group(1)}", float(m.group(2))))
    if found:
        order = sorted(range(len(found)), key=lambda k: found[k][1])
        adj, run = [0.0] * len(found), 0.0
        for rank, k in enumerate(order):
            run = max(run, min(1.0, found[k][1] * (len(found) - rank)))
            adj[k] = run
        out.append(f"0. HOLM CORRECTION over the {len(found)} Wilcoxon tests in script 92")
        for (name, p), pa in zip(found, adj):
            out.append(f"   {name:<62} p={p:.3f}  corrected p={pa:.3f}")
        out.append("")

# ---------------- 1. shape ----------------
out.append("1. SHAPE: middle of the visit (-10 to +10 min) minus its edges (-40 to -30 and +30 to +40 min)")
out.append("   (a visit compared with itself; positive = score peaks around the visit)")
scores = {}
for key, label in [("spec_high", "Spectrum, above 2 kHz"), ("spec_full", "Spectrum, full sound"), ("perch_high", "Perch, above 2 kHz")]:
    if key == "perch_high" and "phigh" not in feat:
        continue
    print(f"Running leave-one-visit-out: {label}...")
    per = lovo(key)
    scores[key] = per
    shape = [avg(d, CENTRE) - np.nanmean([avg(d, EDGE_B), avg(d, EDGE_A)]) for d in per.values()]
    before_edge = [avg(d, CENTRE) - avg(d, EDGE_B) for d in per.values()]
    out.append(f"   {label:<24} middle - both edges : {test(shape)}")
    out.append(f"   {'':<24} middle - early edge : {test(before_edge)}")
    out.append(f"   {'':<24} whole hour vs normal: {test([avg(d, MID) for d in per.values()])}")
out.append("")

# ---------------- 2. weather ----------------
WX = ["Temperature", "Humidity", "Windspeed"]
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

    def wmean(t):
        x = wx[(wx.index >= (t - pd.Timedelta("30min")).floor("h")) & (wx.index < t + pd.Timedelta("30min"))]
        return x.mean() if len(x) else None

    rows = []
    for _, v in vis.iterrows():
        wv = wmean(v.t0)
        wp = [wmean(Wn.loc[p, "t0"]) for p in Wn[(Wn.kind == "placebo") & (Wn.vid == v.vid)].index]
        wp = [x for x in wp if x is not None]
        if wv is None or not wp:
            continue
        r = {c: wv[c] - pd.concat(wp, axis=1).median(axis=1)[c] for c in WX}
        r["vid"] = v.vid
        rows.append(r)
    D = pd.DataFrame(rows)
    out.append(f"2. WEATHER around the logged moment (hourly, shifted {best:+d} h): visit minus its normal days")
    for c in WX:
        out.append(f"   {c:<12} {test(D[c])}")
    if "spec_high" in scores and len(D) >= 6:
        D["score"] = [avg(scores["spec_high"].get(v, {}), MID) for v in D.vid]
        D["shape"] = [avg(scores["spec_high"].get(v, {}), CENTRE) - np.nanmean([avg(scores["spec_high"].get(v, {}), EDGE_B),
                      avg(scores["spec_high"].get(v, {}), EDGE_A)]) for v in D.vid]
        out.append("   Does the spectrum (above 2 kHz) score follow the weather difference? (Spearman, across visits)")
        for c in WX:
            ok = D[[c, "score"]].dropna()
            if len(ok) >= 6:
                rho, p = stats.spearmanr(ok[c], ok["score"])
                out.append(f"     whole-hour score vs {c:<12} rho={rho:+.2f}  p={p:.3f}  (n={len(ok)})")
        try:
            ok = D[WX + ["score"]].dropna()
            A = np.c_[np.ones(len(ok)), (ok[WX] - ok[WX].mean()).values / ok[WX].std().replace(0, 1).values]
            coef, *_ = np.linalg.lstsq(A, ok["score"].values, rcond=None)
            res = ok["score"].values - A[:, 1:] @ coef[1:]
            out.append(f"   Score with the weather differences regressed out (rough, n={len(ok)}): {test(res)}")
        except Exception as e:
            out.append(f"   (regression skipped: {e})")
    out.append("")
else:
    out += ["2. WEATHER: no weather columns in the dataset; skipped.", ""]

# ---------------- 3. which bands ----------------
out.append("3. WHICH PITCH BANDS DIFFER? visit minus normal days, -30 to +30 min, above 2 kHz, no human-tagged clips (dB)")
band_rows = []
for _, v in vis.iterrows():
    def mean_spec(i):
        A = [feat["spec"][n] for st in MID for n in plan.get((i, st), []) if n in feat["spec"] and not human_tag.get(n, False)]
        return 10 * np.mean(A, axis=0) if len(A) >= 20 else None   # mean log10 power x10 = dB
    sv = mean_spec(v.name)
    sp = [s for s in (mean_spec(p) for p in Wn[(Wn.kind == "placebo") & (Wn.vid == v.vid)].index) if s is not None]
    if sv is not None and sp:
        band_rows.append(sv - np.median(sp, axis=0))
B = np.array(band_rows)
if len(B) >= 4:
    centres = (EDGES[:-1] + EDGES[1:]) / 2000
    out.append("   band (kHz)   median dB   louder in")
    for b in np.where(HIGH_BANDS)[0][::2]:
        col = B[:, b:b + 2].mean(axis=1)
        k = int((col > 0).sum())
        star = "*" if k >= 0.75 * len(col) or k <= 0.25 * len(col) else " "
        out.append(f"   {EDGES[b] / 1000:5.2f}-{EDGES[min(b + 2, 48)] / 1000:5.2f}    {np.median(col):+6.2f}     {k:>2}/{len(col)}{star}")
out.append("")
pd.DataFrame([dict(features=k, vid=v, step=s, score=x) for k, per in scores.items() for v, d in per.items() for s, x in d.items()]
             ).to_csv(os.path.join(OUT, "model_followup_scores.csv"), index=False)
out += ["How to read it:",
        "  0: corrected p below 0.05 = survives testing several models and windows at once.",
        "  1: a positive, significant 'middle - both edges' means the score peaks around the visit (a reaction, or",
        "     people around longer than logged). Flat (near 0) means visit periods just sound different all hour.",
        "  2: weather differences, or a strong rho, mean weather may explain the model's score.",
        "  3: * = same direction in at least 75% of visits; tells you what the model is hearing (e.g. insects at 12-13 kHz)."]
txt = "\n".join(out)
print(txt)
open(os.path.join(OUT, "model_followup_results.txt"), "w").write(txt)
print(f"\nSaved to {OUT}")
