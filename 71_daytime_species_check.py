"""
71_daytime_species_check.py
---------------------------
Pre-specified follow-up to script 70: daytime simulation sessions only
(06:00-17:30 local, whole before/during/after span inside it).

For each session: bird-detected clips and number of species heard in the
30 min BEFORE, the 30-min simulation window (DURING) and 30 min AFTER.
Placebo: the same clock windows at every recorder on days with no
simulation within 2 h. A real reaction should change more in real
sessions than in placebo windows. Rarefied richness (species in 15 random
clips) checks whether a richness drop is just fewer detections overall.

Usage: python 71_daytime_species_check.py path/to/cleaned_df.csv
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats

df = pd.read_csv(sys.argv[1], low_memory=False).drop_duplicates("clip_name")
df = df[df["Human Activity"].astype(str).str.lower() != "speech"]
df["ts"] = pd.to_datetime(df["Timestamp"], format="%Y%m%d_%H%M%S")  # local time

sim = df[df["Sim Type"] != "[]"].copy()
sim["rel"] = sim["Sim Relative Time"].str.strip("[]").str.split(",").str[0].astype(float)
sim["event"] = sim.ts - pd.to_timedelta(sim.rel, unit="s")
cent = []
for r, g in sim.groupby("Recorder"):
    e = g.sort_values("event")
    grp = (e.event.diff() > pd.Timedelta("20min")).cumsum()
    for _, x in e.groupby(grp):
        cent.append((r, x.event.median()))
cent = pd.DataFrame(cent, columns=["r", "c"])

H, F = pd.Timedelta("15min"), pd.Timedelta("30min")
mins = lambda t: t.hour * 60 + t.minute
day = cent[[mins(c - H - F) >= 360 and mins(c + H + F) <= 1050 for c in cent.c]]
rng = np.random.default_rng(0)


def rarefied(sp, k=15):
    if len(sp) < k:
        return np.nan
    return np.mean([len(set(rng.choice(sp, k, replace=False))) for _ in range(50)])


def windows(g, c):
    w = lambda a, b: g[(g.ts >= a) & (g.ts < b)].species.values
    B, D, A = w(c - H - F, c - H), w(c - H, c + H), w(c + H, c + H + F)
    return dict(nb=len(B), nd=len(D), na=len(A),
                rb=len(set(B)), rd=len(set(D)), ra=len(set(A)),
                qb=rarefied(B), qd=rarefied(D), qa=rarefied(A))


def active(g, c):
    return (((g.ts < c - H - F) & (g.ts > c - pd.Timedelta("3h"))).any()
            and ((g.ts > c + H + F) & (g.ts < c + pd.Timedelta("3h"))).any())


real = pd.DataFrame([windows(df[df.Recorder == r], c) for r, c in day.itertuples(index=False)
                     if active(df[df.Recorder == r], c)])  # recorder must be running before AND after
plac, days = [], sorted(df.ts.dt.normalize().unique())
for r, c in day.itertuples(index=False):
    tod = c - c.normalize()
    for rec in df.Recorder.unique():
        g = df[df.Recorder == rec]
        for d in days:
            cc = d + tod
            if (abs(cent[cent.r == rec].c - cc) < pd.Timedelta("2h")).any() or not active(g, cc):
                continue
            plac.append(windows(g, cc))
plac = pd.DataFrame(plac)

print(f"Daytime sessions: {len(real)}   placebo windows: {len(plac)}\n")
for x, y, lab in [("rd", "rb", "Species heard, during - before"),
                  ("ra", "rb", "Species heard, after - before"),
                  ("nd", "nb", "Bird clips, during - before"),
                  ("na", "nb", "Bird clips, after - before"),
                  ("qd", "qb", "Rarefied species, during - before"),
                  ("qa", "qb", "Rarefied species, after - before")]:
    dr, dp = (real[x] - real[y]).dropna(), (plac[x] - plac[y]).dropna()
    p = stats.mannwhitneyu(dr, dp).pvalue
    print(f"{lab:34s} sessions median {dr.median():+.2f} (lower in {int((dr < 0).sum())}/{len(dr)})"
          f"  placebo median {dp.median():+.2f}  p={p:.3f}")
