r"""
92_model_timecourse.py  —  run in Google Colab, ideally with a GPU runtime (~1.5-2.5 h, resumable)
------------------------------------------------------------------------------------------------
Two questions, answered with a model rather than loudness bands:

 A. WHEN does a model start to "notice" a human-presence visit?
    A classifier learns "people present" (0 to +10 min after the logged moment) versus normal days
    (same recorder, same clock time, no event within 2 h). It is then run, minute by minute, over a
    visit it has NEVER seen (leave-one-visit-out), from 40 min before to 40 min after the logged moment.
    People become audible only in the last ~5 min before logging (script 90), so a score that rises
    from -25 to -5 min means the model picks up something in the ENVIRONMENT before the people.

 B. Does it still work WITHOUT the people's own sounds?
    The same thing is repeated using only the sound above 2 kHz (high-pass filtered audio), and
    dropping clips where BirdNET tagged human voice, engine, gun, dog, siren or power tools.
    If the high-frequency model still separates visits from normal days, the signal is the forest's
    response, not the people.

Features (both full and above-2-kHz versions):
  perch    : Perch v2 embedding of the whole 3-s clip (same model as the summer work)
  spectrum : 48-band log spectrum, 0-16 kHz (always computed; used alone if Perch cannot load)
Classifier as in the summer: PCA(20) -> StandardScaler -> balanced logistic regression,
everything fitted on the training visits only.

Uses the same visits, normal days and sampled clips as script 89 (12 clips per 5-min step).
Results: model_timecourse_results.txt, model_timecourse.csv, model_timecourse.png in toucanet_soundscape.
Features are cached in model_features_cache.npz, so if Colab disconnects, just run it again.

Colab:  Runtime > Change runtime type > T4 GPU (optional, faster), upload this file, then
   from google.colab import drive; drive.mount("/content/drive")
   %run /content/92_model_timecourse.py
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

# ---------------- features ----------------
EDGES = np.linspace(0, 16000, 49)                 # 48 bands of 1/3 kHz
HIGH_BANDS = EDGES[:-1] >= 2000
SOS = signal.butter(8, 2000, btype="highpass", fs=32000, output="sos")


def read(name):
    try:
        y, sr = load(paths[name])
        if sr != 32000:
            y = signal.resample_poly(y, 32000, sr).astype("float32")
        y = y - y.mean()
        f, P = signal.welch(y, fs=32000, nperseg=1024)
        spec_ = np.array([P[(f >= a) & (f < b)].mean() for a, b in zip(EDGES[:-1], EDGES[1:])])
        return name, y.astype("float32"), np.log10(spec_ + 1e-15).astype("float32")
    except Exception:
        return None


perch = None
def perch_embed(arrays):
    res = perch.encode_arrays(iter([(a, 32000) for a in arrays]), n_producers=1, batch_size=32)
    if hasattr(res, "embeddings"):
        E = np.asarray(res.embeddings, dtype="float32")
        if E.ndim == 3:
            E = np.nanmean(E, axis=1)
    else:
        E = np.array([np.asarray(r["embedding"] if isinstance(r, dict) else getattr(r, "embedding", r),
                                 dtype="float32").reshape(-1) for r in res])
    if len(E) != len(arrays):
        raise ValueError(f"Perch returned {len(E)} embeddings for {len(arrays)} clips")
    return E.reshape(len(arrays), -1)


def load_perch():
    global perch
    try:
        try:
            from birdnet import load_perch_v2
        except ImportError:
            import subprocess
            print("Installing birdnet (for Perch v2)...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "birdnet"], check=False)
            from birdnet import load_perch_v2
        dev = "CPU"
        try:
            import tensorflow as tf
            dev = "GPU" if tf.config.list_physical_devices("GPU") else "CPU"
        except Exception:
            pass
        perch = load_perch_v2(device=dev)
        test = perch_embed([np.zeros(96000, dtype="float32")] * 2)
        print(f"Perch v2 loaded on {dev}; embedding size {test.shape[1]}")
        return True
    except Exception as e:
        perch = None
        print(f"Perch v2 could not be used ({str(e)[:150]}). Continuing with the spectrum features only.")
        return False


cache_file = os.path.join(OUT, "model_features_cache.npz")
feat = {"spec": {}, "pfull": {}, "phigh": {}}
if os.path.exists(cache_file):
    z = np.load(cache_file, allow_pickle=True)
    for key in feat:
        if f"{key}_names" in z:
            feat[key] = dict(zip(z[f"{key}_names"], z[f"{key}_vals"].astype("float32")))


def save_cache():
    arrs = {}
    for key, d in feat.items():
        arrs[f"{key}_names"] = np.array(list(d.keys()), dtype=object)
        arrs[f"{key}_vals"] = np.array(list(d.values()), dtype="float16") if d else np.zeros((0, 1), "float16")
    np.savez(cache_file + ".tmp.npz", **arrs)
    os.replace(cache_file + ".tmp.npz", cache_file)


use_perch = load_perch()
todo = sorted(n for n in need if n not in feat["spec"] or (use_perch and n not in feat["pfull"]))
print(f"  {len(need) - len(todo):,} already done, {len(todo):,} to read")
if todo:
    print("Finding the clips (a few minutes)...")
    want, paths = set(todo), {}
    for rec in sorted(Wn.rec.unique()):
        for root, _, files in os.walk(os.path.join(AUDIO, rec.replace("_", " "))):
            for f in files:
                if f in want:
                    paths[f] = os.path.join(root, f)
    print(f"  found {len(paths):,} of {len(todo):,}")
    names = [n for n in todo if n in paths]
    t_start, BATCH = time.time(), 1000
    with ThreadPoolExecutor(THREADS) as ex:
        for b0 in range(0, len(names), BATCH):
            got = [r for r in ex.map(read, names[b0:b0 + BATCH]) if r]
            for n, _, s in got:
                feat["spec"][n] = s
            if use_perch and got:
                try:
                    ys = [y[:96000] if len(y) >= 96000 else np.pad(y, (0, 96000 - len(y))) for _, y, _ in got]
                    full = perch_embed(ys)
                    high = perch_embed([signal.sosfiltfilt(SOS, y).astype("float32") for y in ys])
                    for (n, _, _), a, b in zip(got, full, high):
                        feat["pfull"][n], feat["phigh"][n] = a, b
                except Exception as e:
                    print(f"  Perch failed on a batch ({str(e)[:120]}); continuing with the spectrum only.")
                    use_perch = False
            save_cache()
            k = min(b0 + BATCH, len(names))
            print(f"  {k:,}/{len(names):,}  ({(time.time() - t_start) / 60:.1f} min)")

# ---------------- leave-one-visit-out model ----------------
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

PRESENT = (0, 5)                         # "people present": 0 to +10 min after the logged moment
EARLY = (-25, -20, -15, -10)             # 25 to 5 min before: before people are audible
vis = Wn[(Wn.kind == "visit") & (Wn.group == "presence")]
vis = vis[[(Wn.kind == "placebo").__and__(Wn.vid == v).any() for v in vis.vid]]
print(f"\nHuman-presence visits with normal days: {len(vis)}")


def vector(key, n):
    if key == "spec_full":
        return feat["spec"].get(n)
    if key == "spec_high":
        s = feat["spec"].get(n)
        return None if s is None or human_tag.get(n, False) else s[HIGH_BANDS]
    if key == "perch_full":
        return feat["pfull"].get(n)
    if key == "perch_high":
        return None if human_tag.get(n, False) else feat["phigh"].get(n)


def clips_of(i, st, key):
    out = []
    for n in plan.get((i, st), []):
        v = vector(key, n)
        if v is not None:
            out.append(v)
    return out


sets = [("perch_full", "Perch, full sound"), ("perch_high", "Perch, above 2 kHz, no human-tagged clips"),
        ("spec_full", "Spectrum, full sound"), ("spec_high", "Spectrum, above 2 kHz, no human-tagged clips")]
sets = [s for s in sets if not s[0].startswith("perch") or len(feat["pfull"]) > 0]
out = ["MODEL TIME COURSE: when does a model notice a human-presence visit it has never seen?", "=" * 92,
       "Trained on: 0 to +10 min after the logged moment (people present) vs normal days; leave one visit out.",
       "Score = visit's mean model score minus its normal days' at the same clock time (log-odds).",
       "* = higher than normal days in at least 75% of visits. People become audible only in the last ~5 min before 0.", ""]
rows = []
for key, label in sets:
    per_visit, pooled_y, pooled_s = {}, [], []
    for _, v in vis.iterrows():
        i, vid = v.name, v.vid
        tr_vis = vis[vis.vid != vid]
        Xp, Xn = [], []
        for j, tv in tr_vis.iterrows():
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
                sv = score(cv_)
                d[st] = sv.mean() - np.median(sp)
                if st in PRESENT:
                    pooled_s += list(sv); pooled_y += [1] * len(sv)
                    for p in pls:
                        c = clips_of(p, st, key)
                        if len(c) >= 4:
                            s_ = score(c); pooled_s += list(s_); pooled_y += [0] * len(s_)
        per_visit[vid] = d
    n = len(per_visit)
    out.append(f"{label}  ({n} visits)")
    if n < 4:
        out += ["  too few visits", ""]
        continue
    auc = roc_auc_score(pooled_y, pooled_s) if len(set(pooled_y)) == 2 else float("nan")
    out.append("  minutes      score   higher in")
    for st in steps:
        a = np.array([d[st] for d in per_visit.values() if st in d])
        if len(a) < 4:
            continue
        fu = (a > 0).mean()
        tag = "  <- logged" if st == 0 else ("  (trained here)" if st in PRESENT else "")
        out.append(f"  {st:+3d} to {st + STEP:+3d}   {np.median(a):+6.2f}   {(a > 0).sum():>2}/{len(a):<2}{'*' if fu >= .75 else ' '}{tag}")
        rows.append(dict(features=key, step_start=st, median_score=np.median(a), share_higher=fu, visits=len(a)))
    early = np.array([np.mean([d[s] for s in EARLY if s in d]) for d in per_visit.values() if sum(s in d for s in EARLY) >= 3])
    pres = np.array([np.mean([d[s] for s in PRESENT if s in d]) for d in per_visit.values() if any(s in d for s in PRESENT)])
    for nm, a in [("people present, 0 to +10 min", pres), ("EARLY, 25 to 5 min before   ", early)]:
        if len(a) >= 4:
            k = int((a > 0).sum())
            out.append(f"  {nm}: median {np.median(a):+.2f}, higher in {k}/{len(a)}, "
                       f"sign p={stats.binomtest(k, len(a)).pvalue:.3f}, Wilcoxon p={stats.wilcoxon(a).pvalue:.3f}")
    out.append(f"  clip-level AUC, people present vs normal days (held-out visits): {auc:.2f}  (0.5 = chance)")
    out.append("")
pd.DataFrame(rows).to_csv(os.path.join(OUT, "model_timecourse.csv"), index=False)
out += ["How to read it:",
        "  - 'people present' line and AUC: can the model recognise a visit at all, on visits it never saw?",
        "    For the 'full sound' models this is expected (it can hear people). For the 'above 2 kHz' models a",
        "    clearly positive result means the forest itself sounds different while people are there (question B).",
        "  - EARLY line: a positive, significant score 25 to 5 min before the logged moment means the model",
        "    notices the visit before the people are audible (question A, early warning).",
        "  - Each visit is compared with its own normal days, so daily rhythm and recorder differences cancel."]
txt = "\n".join(out)
print(txt)
open(os.path.join(OUT, "model_timecourse_results.txt"), "w").write(txt)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    R = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = {"perch_full": "#1F3A2E", "perch_high": "#C8641C", "spec_full": "#7A8C83", "spec_high": "#3B6EA8"}
    for key, label in sets:
        x = R[R.features == key]
        if len(x):
            ax.plot(x.step_start + STEP / 2, x.median_score, lw=2.5, color=colors[key], label=label)
    ax.axvline(0, color="#1F3A2E", ls="--", lw=1.5)
    ax.axvspan(-5, 0, color="#DDDDDD", alpha=0.5, lw=0)
    ax.axhline(0, color="#999999", lw=0.8)
    ax.set_xlabel("minutes from the logged moment (grey: people audible)")
    ax.set_ylabel("model score, visit minus normal days")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "model_timecourse.png"), dpi=150)
    print("Saved model_timecourse.png")
except Exception as e:
    print(f"(plot skipped: {e})")
print(f"\nSaved to {OUT}")
