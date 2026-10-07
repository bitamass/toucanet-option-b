"""find_bouts.py -- inspect recording timestamps to reveal bout structure."""
import os, re, glob
import numpy as np
import pandas as pd

DIR = os.path.dirname(os.path.abspath(__file__))

def parse(name):
    m = re.search(r'(?:Audio[_ ]?Moth[_ ]?|AM)(\d+).*?(\d{8})_(\d{6})', str(name), re.I)
    if m:
        rec = f"AM{int(m.group(1))}"
        ts = pd.to_datetime(m.group(2) + m.group(3), format="%Y%m%d%H%M%S")
        return rec, ts
    return "AM?", pd.NaT

paths = [p for p in sorted(glob.glob(os.path.join(DIR, "am*_feature_matrix.csv")))
         if re.search(r'am\d+_feature_matrix\.csv$', os.path.basename(p), re.I)]

rows = []
for p in paths:
    df = pd.read_csv(p, usecols=["clip_name", "clip_label"])
    df["rows_per_wav"] = df.groupby("clip_name")["clip_name"].transform("count")
    pr = df["clip_name"].apply(parse)
    df["recorder"] = [a for a, _ in pr]
    df["ts"] = [b for _, b in pr]
    rows.append(df)
d = pd.concat(rows, ignore_index=True)
d["label"] = d["clip_label"].astype(int)

print("=" * 60)
print("SEGMENTS vs RECORDINGS")
print("=" * 60)
print(f"unique .wav files ...... {d['clip_name'].nunique()}")
print(f"total rows (clips) ..... {len(d)}")
print(f"rows per .wav: min={d['rows_per_wav'].min()}, "
      f"median={int(d['rows_per_wav'].median())}, max={d['rows_per_wav'].max()}")
print("  (>1 => each .wav is split into multiple segment-rows)\n")

recs = (d.sort_values("ts")
          .groupby(["recorder", "clip_name"])
          .agg(ts=("ts", "first"), pos=("label", "sum"), n=("label", "count"))
          .reset_index())

print("=" * 60)
print("GAPS BETWEEN CONSECUTIVE RECORDINGS (per recorder)")
print("=" * 60)
for rec, g in recs.sort_values(["recorder", "ts"]).groupby("recorder"):
    g = g.sort_values("ts")
    gaps = g["ts"].diff().dt.total_seconds().div(60)
    print(f"\n{rec}: {len(g)} recordings, {g['ts'].dt.date.nunique()} day(s)")
    big = gaps[gaps > 60]
    print(f"  gaps >60 min: {len(big)}  -> ~{len(big)+ g['ts'].dt.date.nunique()} bouts (rough)")
    if len(gaps.dropna()):
        q = gaps.dropna().quantile([.5, .75, .9, .99]).round(1)
        print(f"  gap minutes  median={q[.5]}  p75={q[.75]}  p90={q[.9]}  p99={q[.99]}")

print("\n" + "=" * 60)
print("Do recordings mix classes, or is each .wav pure?")
print("=" * 60)
recs["kind"] = np.where(recs.pos == recs.n, "all-sim",
                np.where(recs.pos == 0, "all-baseline", "mixed"))
print(recs["kind"].value_counts().to_string())
