"""
75_species_and_power.py
-----------------------
Two pre-specified follow-ups to script 74 (bioacoustics_dataset labels, all clips,
daytime sessions only, same before/during windows and placebo rules).

A. Species hypotheses, fixed BEFORE looking at results:
   expected QUIETER during people (shy ground birds):
     Great Tinamou, Streak-chested Antpitta, Chestnut-backed Antbird
   expected MORE calling (alarm / mobbing callers):
     Black-cheeked Ant-Tanager, Black-hooded Antshrike, Rufous Piha, Scarlet Macaw
   For each: share of clips with that species, during - before, sessions vs placebo.
   Holm correction across the 7 species. Also: do the changes go in the
   predicted direction more often than chance?

B. Smallest detectable effect: using the spread of (during - before) in placebo
   windows, how big a change in "share of clips with birds" could these
   sessions detect with 80% power (two-sided, alpha 0.05)?

Usage: python 75_species_and_power.py "C:\\...\\bioacoustics_dataset.csv"
"""
import ast
import sys
import numpy as np
import pandas as pd
from scipy import stats

PATH, CONF = sys.argv[1], 0.25
QUIETER = ["Great Tinamou", "Streak-chested Antpitta", "Chestnut-backed Antbird"]
LOUDER = ["Black-cheeked Ant-Tanager", "Black-hooded Antshrike", "Rufous Piha", "Scarlet Macaw"]
TARGETS = QUIETER + LOUDER

want = {"clip_name", "Recorder", "Timestamp", "species", "confidence", "Sim Type", "Sim Relative Time"}
df = pd.read_csv(PATH, usecols=lambda c: c in want, low_memory=False)
df["clip_name"] = df["clip_name"].astype(str)
df = df.drop_duplicates("clip_name")
df["ts"] = pd.to_datetime(df["Timestamp"].astype(str), format="%Y%m%d_%H%M%S", errors="coerce")
df = df.dropna(subset=["ts"]).sort_values(["Recorder", "ts"]).reset_index(drop=True)


def as_list(v):
    if isinstance(v, str) and v.strip().startswith("["):
        try:
            return list(ast.literal_eval(v))
        except (ValueError, SyntaxError):
            return []
    if v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() in ("", "nan", "None"):
        return []
    return [v]


def birds(row):
    sp, cf = as_list(row["species"]), as_list(row["confidence"])
    if len(cf) == 1 and len(sp) > 1:
        cf = cf * len(sp)
    out = set()
    for s, c in zip(sp, cf):
        try:
            if float(c) >= CONF and str(s).strip():
                out.add(str(s))
        except (TypeError, ValueError):
            pass
    return out


print("Parsing BirdNET detections...")
bs = df.apply(birds, axis=1)
df["bird"] = bs.str.len() > 0
for t in TARGETS:
    df[t] = bs.apply(lambda s, t=t: any(t.lower() in x.lower() for x in s))
print("  detections of target species (whole dataset):")
for t in TARGETS:
    print(f"    {t:<28} {int(df[t].sum()):>6}")

sim = df[df["Sim Type"].astype(str).str.strip() != "[]"].copy()
sim["rel"] = pd.to_numeric(sim["Sim Relative Time"].astype(str).str.strip("[]").str.split(",").str[0],
                           errors="coerce").fillna(0)
sim["event"] = sim.ts - pd.to_timedelta(sim.rel, unit="s")
sess = []
for r, g in sim.groupby("Recorder"):
    e = g.sort_values("event")
    grp = (e.event.diff() > pd.Timedelta("20min")).cumsum()
    for _, x in e.groupby(grp):
        sess.append(dict(rec=r, start=x.ts.min(), end=x.ts.max() + pd.Timedelta("3s"), centre=x.event.median()))
sess = pd.DataFrame(sess)
by_rec = {r: g for r, g in df.groupby("Recorder")}
cols = ["bird"] + TARGETS


def win(g, a, b):
    w = g[(g.ts >= a) & (g.ts < b)]
    cov = len(w) / ((b - a).total_seconds() / 3)
    return cov, (w[cols].mean() if len(w) else pd.Series(np.nan, index=cols))


def delta(g, s, e):
    L = e - s
    m0, m1 = s - L, e + L
    if not (m0.date() == m1.date() and m0.hour * 60 + m0.minute >= 360 and m1.hour * 60 + m1.minute <= 1050):
        return None
    cb, b = win(g, s - L, s)
    cd, d = win(g, s, e)
    if min(cb, cd) < 0.8:
        return None
    return d - b


real, plac = [], []
days = sorted(df.ts.dt.normalize().unique())
for x in sess.itertuples():
    g = by_rec[x.rec]
    dv = delta(g, x.start, x.end)
    if dv is None:
        continue
    real.append(dv)
    for d in days:
        off = d - x.start.normalize()
        if off == pd.Timedelta(0):
            continue
        s2, e2 = x.start + off, x.end + off
        if ((sess.rec == x.rec) & ((sess.centre - (s2 + (e2 - s2) / 2)).abs() < pd.Timedelta("2h"))).any():
            continue
        dp = delta(g, s2, e2)
        if dp is not None:
            plac.append(dp)
R, P = pd.DataFrame(real), pd.DataFrame(plac)
out = ["SPECIES TEST AND DETECTABLE EFFECT (daytime, bioacoustics labels)", "=" * 66,
       f"Daytime sessions: {len(R)}   placebo windows: {len(P)}", "",
       "A. Target species: change in share of clips with the species (during - before)"]
ps, agree, tested = [], 0, 0
for t in TARGETS:
    dr, dp = R[t].dropna(), P[t].dropna()
    if (dr != 0).sum() + (dp != 0).sum() < 3:
        out.append(f"  {t:<28} too few detections to test")
        continue
    p = stats.mannwhitneyu(dr, dp).pvalue
    ps.append((t, p))
    expect = -1 if t in QUIETER else 1
    diff = dr.median() - dp.median()
    if diff != 0:
        tested += 1
        agree += int(np.sign(diff) == expect)
    out.append(f"  {t:<28} expected {'quieter' if expect < 0 else 'louder ':<7}  sessions {dr.median():+.4f}"
               f"  placebo {dp.median():+.4f}  mean diff {dr.mean() - dp.mean():+.4f}  p={p:.3f}")
if ps:
    ps_sorted = sorted(ps, key=lambda x: x[1])
    m = len(ps_sorted)
    holm = [min(1, (m - i) * p) for i, (_, p) in enumerate(ps_sorted)]
    holm = np.maximum.accumulate(holm)
    out.append("  Holm-corrected: " + ", ".join(f"{t.split()[-1]} {h:.2f}" for (t, _), h in zip(ps_sorted, holm)))
if tested:
    out.append(f"  Direction matched prediction for {agree}/{tested} species "
               f"(chance p={stats.binomtest(agree, tested, 0.5, alternative='greater').pvalue:.2f})")

sd = P["bird"].std(ddof=1)
n, m = len(R), len(P)
mde = (stats.norm.ppf(0.975) + stats.norm.ppf(0.8)) * sd * np.sqrt(1 / n + 1 / m) if n and m else np.nan
base = df["bird"].mean()
out += ["", "B. Smallest detectable change in share of clips with birds",
        f"  spread of placebo changes (SD): {sd:.3f}",
        f"  with {n} sessions and {m} placebo windows, 80% power to detect a change of about "
        f"{mde:.3f} ({mde * 100:.1f} percentage points)",
        f"  for scale: birds are heard in {base:.1%} of all clips, so that is about "
        f"{mde / base:.0%} of the typical level"]
txt = "\n".join(out)
print(txt)
open("species_power_results.txt", "w").write(txt)
