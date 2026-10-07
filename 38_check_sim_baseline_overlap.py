"""
38_check_sim_baseline_overlap.py
================================================================
Answers one question: are simulation and baseline clips recorded in
the SAME recorder-days, or in separate ones?

If they share recorder-days -> a recorder-day LOSSO is meaningful.
If they never share -> sim vs baseline is confounded with day/session
by design, and no grouping can separate the disturbance from the
recording context (a data-collection finding to report to Sammy).

Reads am*_feature_matrix.csv from this script's folder. No edits needed.
================================================================
"""

import os, re, glob
import pandas as pd

DIR = os.path.dirname(os.path.abspath(__file__))
paths = [p for p in sorted(glob.glob(os.path.join(DIR, "am*_feature_matrix.csv")))
         if re.search(r'am\d+_feature_matrix\.csv$', os.path.basename(p), re.I)]

def parse(name):
    m = re.search(r'(?:Audio[_ ]?Moth[_ ]?|AM)(\d+).*?(\d{8})_(\d{6})', str(name), re.I)
    if m:
        return f"AM{int(m.group(1))}", m.group(2), m.group(3)
    return "AM?", "????????", "??????"

rows = []
for p in paths:
    df = pd.read_csv(p, usecols=["clip_name", "clip_label"])
    pr = df["clip_name"].apply(parse)
    df["recorder"] = [a for a, _, _ in pr]
    df["day"]      = [b for _, b, _ in pr]
    df["rec_day"]  = df["recorder"] + "|" + df["day"]
    rows.append(df)

d = pd.concat(rows, ignore_index=True)
d["label"] = d["clip_label"].astype(int)

print("=" * 60)
print("RECORDER-DAY  x  LABEL")
print("=" * 60)
g = d.groupby("rec_day")["label"].agg(n="count", pos="sum")
g["neg"] = g["n"] - g["pos"]
both = g[(g.pos > 0) & (g.neg > 0)]
pos_only = g[(g.pos > 0) & (g.neg == 0)]
neg_only = g[(g.pos == 0) & (g.neg > 0)]

print(f"recorder-days total ................ {len(g)}")
print(f"  with BOTH sim & baseline ......... {len(both)}   <-- usable LOSSO folds")
print(f"  sim only ......................... {len(pos_only)}")
print(f"  baseline only .................... {len(neg_only)}\n")

print("mixed recorder-days (both classes present):")
if len(both):
    print(both.sort_values("pos", ascending=False).to_string())
else:
    print("  NONE -- sims and baselines never share a recorder-day.")
print()

# also check the coarser recorder-level and day-level views
print("=" * 60)
print("Do sim and baseline at least share DAYS (any recorder)?")
print("=" * 60)
d["date"] = d["day"]
by_day = d.groupby("date")["label"].agg(pos="sum", n="count")
by_day["neg"] = by_day["n"] - by_day["pos"]
mixed_days = by_day[(by_day.pos > 0) & (by_day.neg > 0)]
print(by_day.to_string())
print(f"\ndays with both sim & baseline: {len(mixed_days)} of {len(by_day)}")

print("\n" + "=" * 60)
print("By RECORDER: does each recorder have both classes?")
print("=" * 60)
rr = d.groupby("recorder")["label"].agg(pos="sum", n="count")
rr["neg"] = rr["n"] - rr["pos"]
print(rr.to_string())