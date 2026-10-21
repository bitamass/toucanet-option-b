"""
70_forest_response.py
---------------------
Does the forest (birds) change when people are present?

No model is trained. For each simulation session at each recorder, bird
activity is compared:
  - BEFORE (30 min before), DURING (the 30-min simulation window),
    AFTER (30 min after), all on the same day at the same recorder;
  - DURING vs the SAME clock window on OTHER days at the same recorder
    (no simulation nearby), which controls for location and time of day.

Data: cleaned_df.csv, one row per 3-second clip that BirdNET kept (a bird
was detected). Recording is continuous 3-second clips, so
"bird clips per minute" = how much of the time a bird was detected (max 20).

Notes
  - Clip-name times are LOCAL (owls peak 18:00-05:00 on that clock). The
    file's 'Time Of Day' and 'Timestamp Local' columns are shifted 6 hours
    and are not used.
  - Clips flagged as human speech by the existing detector are removed.
  - Simulation windows are rebuilt from 'Sim Relative Time' (seconds from
    the logged event; labels cover +/-15 min around it).

Usage:  python 70_forest_response.py path/to/cleaned_df.csv [out_dir]
"""
import sys, os, ast
import numpy as np
import pandas as pd
from scipy import stats

CSV = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "."
HALF = pd.Timedelta("15min")      # simulation window = event +/- 15 min
FLANK = pd.Timedelta("30min")     # before / after windows
MIN_CONTROL_DAYS = 1


def shannon(counts):
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum()) if len(p) else 0.0


def window_stats(g, t0, t1):
    w = g[(g.ts >= t0) & (g.ts < t1)]
    minutes = (t1 - t0).total_seconds() / 60
    if len(w) == 0:
        return dict(rate=0.0, richness=0, shannon=0.0, owl_frac=np.nan, n=0)
    sp = w.species.value_counts()
    return dict(rate=len(w) / minutes, richness=int(sp.size), shannon=shannon(sp),
                owl_frac=float(w.species.str.contains("Owl").mean()), n=len(w))


def recorder_active(g, t0, t1, pad=pd.Timedelta("2h")):
    """Recorder considered running if it logged bird clips within 2 h on both sides."""
    return ((g.ts < t0) & (g.ts >= t0 - pad)).any() and ((g.ts >= t1) & (g.ts < t1 + pad)).any()


df = pd.read_csv(CSV, low_memory=False).drop_duplicates("clip_name")
df = df[df["Human Activity"].astype(str).str.lower() != "speech"]
df["ts"] = pd.to_datetime(df["Timestamp"], format="%Y%m%d_%H%M%S")
df["rec"] = df["Recorder"].str.replace("Audio_Moth_", "AM")
df = df.sort_values(["rec", "ts"])

# ---- rebuild simulation sessions from event times -------------------------
sim = df[df["Sim Type"] != "[]"].copy()
sim["rel"] = sim["Sim Relative Time"].str.strip("[]").str.split(",").str[0].astype(float)
sim["event"] = sim.ts - pd.to_timedelta(sim.rel, unit="s")
sessions = []
for rec, g in sim.groupby("rec"):
    ev = g.sort_values("event")
    grp = (ev.event.diff() > pd.Timedelta("20min")).cumsum()
    for _, e in ev.groupby(grp):
        center = e.event.median().round("s")
        types = sorted({t for s in e["Sim Type"] for t in ast.literal_eval(s)})
        sessions.append(dict(rec=rec, center=center, types=", ".join(types)))
sess = pd.DataFrame(sessions)

# ---- compare windows ------------------------------------------------------
rows = []
for s in sess.itertuples():
    g = df[df.rec == s.rec]
    d0, d1 = s.center - HALF, s.center + HALF
    if not recorder_active(g, d0 - FLANK, d1 + FLANK):
        continue
    dur = window_stats(g, d0, d1)
    bef = window_stats(g, d0 - FLANK, d0)
    aft = window_stats(g, d1, d1 + FLANK)
    # same clock window on other days, no simulation within 1 h
    ctrl = []
    for day in sorted(g.ts.dt.normalize().unique()):
        if day == s.center.normalize():
            continue
        c0 = day + (d0 - d0.normalize()); c1 = c0 + 2 * HALF
        near = sess[(sess.rec == s.rec) & (abs(sess.center - (c0 + HALF)) < pd.Timedelta("1h"))]
        if len(near) or not recorder_active(g, c0, c1):
            continue
        ctrl.append(window_stats(g, c0, c1))
    row = dict(rec=s.rec, start=d0, types=s.types, ctrl_days=len(ctrl))
    for k in ["rate", "richness", "shannon", "owl_frac"]:
        row[f"during_{k}"] = dur[k]; row[f"before_{k}"] = bef[k]; row[f"after_{k}"] = aft[k]
        cv = [c[k] for c in ctrl if not (isinstance(c[k], float) and np.isnan(c[k]))]
        row[f"ctrl_{k}"] = float(np.mean(cv)) if cv else np.nan
        row[f"ctrl_sd_{k}"] = float(np.std(cv)) if len(cv) > 1 else np.nan
    rows.append(row)

res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, "forest_response_by_session.csv"), index=False)

# ---- summary --------------------------------------------------------------
lines = ["FOREST RESPONSE TO SIMULATIONS", "=" * 66,
         f"Sessions analysed: {len(res)} (of {len(sess)}), recorders: {', '.join(sorted(res.rec.unique()))}",
         "Metric 'rate' = bird-detected clips per minute (max 20).", ""]


def paired(a, b, label):
    m = (~a.isna()) & (~b.isna())
    a, b = a[m], b[m]
    if len(a) < 3:
        return f"  {label:<34} n={len(a)}  too few"
    diff = a - b
    try:
        p = stats.wilcoxon(a, b).pvalue
    except ValueError:
        p = np.nan
    pct = 100 * diff.median() / max(b.median(), 1e-9)
    lower = int((diff < 0).sum())
    return (f"  {label:<34} n={len(a):>2}  median change {diff.median():+.2f} ({pct:+.0f}%)  "
            f"lower in {lower}/{len(a)}  p={p:.3f}")


for k, name in [("rate", "Bird activity (clips/min)"), ("richness", "Species heard"),
                ("shannon", "Diversity (Shannon)")]:
    lines.append(name)
    lines.append(paired(res[f"during_{k}"], res[f"before_{k}"], "during vs before (same day)"))
    lines.append(paired(res[f"after_{k}"], res[f"before_{k}"], "after vs before (same day)"))
    lines.append(paired(res[f"during_{k}"], res[f"ctrl_{k}"], "during vs same time, other days"))
    lines.append(paired(res[f"before_{k}"], res[f"ctrl_{k}"], "before vs same time, other days*"))
    lines.append("")
lines.append("* sanity check: 'before' should look like other days if days are comparable.")
lines.append("")
lines.append("Per recorder (median during/before ratio of bird activity):")
for rec, g in res.groupby("rec"):
    r = (g.during_rate / g.before_rate.replace(0, np.nan)).median()
    lines.append(f"  {rec}: {len(g)} sessions, during/before = {r:.2f}")
txt = "\n".join(lines)
print(txt)
open(os.path.join(OUT, "forest_response_summary.txt"), "w").write(txt)
