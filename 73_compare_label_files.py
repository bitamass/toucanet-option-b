"""
73_compare_label_files.py
-------------------------
Compare simulation labels in cleaned_df (1).csv and bioacoustics_dataset.csv.

Prints:
  - row and clip counts in each file (is bioacoustics unfiltered?)
  - how many clips are labelled as simulation in each, and how many disagree
  - the simulation sessions each file implies (recorder, centre time, clips)
  - whether disagreements look like a constant time shift (e.g. 1 hour)
Saves label_disagreements.csv.

Usage:
  python 73_compare_label_files.py "cleaned_df (1).csv" "bioacoustics_dataset.csv"
"""
import sys
import pandas as pd

A, B = sys.argv[1], sys.argv[2]
cols = ["clip_name", "Recorder", "Timestamp", "Sim Type", "Sim Relative Time"]


def load(path):
    d = pd.read_csv(path, usecols=lambda c: c in cols, low_memory=False)
    d["clip_name"] = d["clip_name"].astype(str).str.replace(".wav", "", regex=False) + ".wav"
    d = d.drop_duplicates("clip_name")
    d["sim"] = d["Sim Type"].astype(str).str.strip() != "[]"
    d["ts"] = pd.to_datetime(d["Timestamp"].astype(str), format="%Y%m%d_%H%M%S", errors="coerce")
    return d


a, b = load(A), load(B)
print(f"cleaned_df:   {len(a):>8,} clips, {a.sim.sum():>6,} labelled simulation")
print(f"bioacoustics: {len(b):>8,} clips, {b.sim.sum():>6,} labelled simulation")
both = a.merge(b, on="clip_name", suffixes=("_c", "_b"))
print(f"clips in both files: {len(both):,}")
dis = both[both.sim_c != both.sim_b]
print(f"label disagreements: {len(dis):,}  "
      f"(sim only in bioacoustics: {(dis.sim_b & ~dis.sim_c).sum():,}, "
      f"sim only in cleaned_df: {(dis.sim_c & ~dis.sim_b).sum():,})")
dis[["clip_name", "Recorder_c", "Sim Type_c", "Sim Type_b", "Sim Relative Time_b"]].to_csv(
    "label_disagreements.csv", index=False)


def sessions(d, name):
    s = d[d.sim].copy()
    s["rel"] = s["Sim Relative Time"].astype(str).str.strip("[]").str.split(",").str[0]
    s["rel"] = pd.to_numeric(s["rel"], errors="coerce").fillna(0)
    s["event"] = s.ts - pd.to_timedelta(s.rel, unit="s")
    out = []
    for r, g in s.groupby("Recorder"):
        e = g.sort_values("event")
        grp = (e.event.diff() > pd.Timedelta("20min")).cumsum()
        for _, x in e.groupby(grp):
            out.append(dict(file=name, rec=r, centre=x.event.median().round("min"), clips=len(x)))
    return pd.DataFrame(out)


sa, sb = sessions(a, "cleaned_df"), sessions(b, "bioacoustics")
print(f"\nsessions: cleaned_df {len(sa)}, bioacoustics {len(sb)}")
print("\nSessions side by side (nearest match at the same recorder):")
for r in sorted(set(sa.rec) | set(sb.rec)):
    for x in sb[sb.rec == r].itertuples():
        m = sa[sa.rec == r]
        if len(m):
            j = (m.centre - x.centre).abs().idxmin()
            shift = (m.loc[j, "centre"] - x.centre).total_seconds() / 60
            print(f"  {r}  bioacoustics {x.centre}  ({x.clips:>5} clips)   "
                  f"cleaned_df {m.loc[j, 'centre']}  shift {shift:+.0f} min")
        else:
            print(f"  {r}  bioacoustics {x.centre}  ({x.clips:>5} clips)   cleaned_df: none")
print("\nIf most shifts are the same (for example +60 min), the two files use different "
      "time references for the simulation log.")
