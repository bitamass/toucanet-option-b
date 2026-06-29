"""
24_cluster_count_comparison.py
-------------------------------
Shows how many distinct acoustic clusters exist in baseline vs simulation
after time-controlled analysis on AudioMoth 4.

Key fix: runs UMAP + HDBSCAN on ALL segments combined, then counts
which clusters appear in baseline vs simulation. This avoids the
sample-size bias of running them separately.

Inputs:
  am4_time_controlled_emb.npy
  am4_time_controlled_labels.npy

Outputs:
  am4_cluster_count_comparison.png
"""

import numpy as np
import matplotlib.pyplot as plt
import hdbscan
import umap

BASE_DIR = r"C:\Users\BitaMassoudi\KasmirWorldProjects\toucanet-option-b"
EMB_FILE = BASE_DIR + r"\am4_time_controlled_emb.npy"
LAB_FILE = BASE_DIR + r"\am4_time_controlled_labels.npy"
OUT_PLOT = BASE_DIR + r"\am4_cluster_count_comparison.png"

# ── 1. load ───────────────────────────────────────────────────────────────────
print("Loading embeddings...")
embeddings = np.load(EMB_FILE)
labels     = np.load(LAB_FILE)

print(f"  Total segments    : {len(embeddings)}")
print(f"  Baseline segments : {(labels == 0).sum()}")
print(f"  Simulation segs   : {(labels == 1).sum()}")

# ── 2. UMAP + HDBSCAN on ALL segments combined ────────────────────────────────
# Critical: run on everything together so both groups share the same
# cluster space. Splitting them separately gives misleading results
# because HDBSCAN is sensitive to sample size.
print("\nRunning UMAP on all segments combined (this may take a minute)...")
X_2d = umap.UMAP(n_components=2, random_state=42).fit_transform(embeddings)

print("Running HDBSCAN on all segments combined...")
cluster_ids = hdbscan.HDBSCAN(min_cluster_size=5).fit_predict(X_2d)

total_clusters = len(set(cluster_ids)) - (1 if -1 in cluster_ids else 0)
print(f"  Total clusters found across all segments: {total_clusters}")

# ── 3. count distinct clusters in each group ──────────────────────────────────
base_clusters = set(cluster_ids[labels == 0]) - {-1}
sim_clusters  = set(cluster_ids[labels == 1]) - {-1}

n_base = len(base_clusters)
n_sim  = len(sim_clusters)

# clusters that appear in baseline but NOT in simulation (went silent)
disappeared = base_clusters - sim_clusters
# clusters that appear in simulation but NOT in baseline (new sounds)
appeared    = sim_clusters - base_clusters
# clusters present in both
shared      = base_clusters & sim_clusters

# ── 4. print results ──────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("CLUSTER COUNT — AudioMoth 4 (Time-Controlled)")
print("=" * 55)
print(f"  Baseline   : {n_base} distinct acoustic clusters")
print(f"  Simulation : {n_sim}  distinct acoustic clusters")
drop_n   = n_base - n_sim
drop_pct = round(drop_n / n_base * 100, 1)
print(f"  Drop       : {drop_n} fewer clusters ({drop_pct}% reduction)")
print("-" * 55)
print(f"  Clusters present in both          : {len(shared)}")
print(f"  Clusters that went silent (base only): {len(disappeared)}")
print(f"  New clusters during simulation    : {len(appeared)}")
print("=" * 55)
print("\nInterpretation:")
print(f"  On a normal night at 7-8pm, {n_base} distinct acoustic")
print(f"  patterns are present. During a disturbance at the same")
print(f"  hours, only {n_sim} patterns remain. {len(disappeared)} acoustic")
print(f"  groups went silent — species that stopped calling.")
print(f"  {len(appeared)} new patterns appeared — likely alarm calls.")
print(f"  This simplification is NOT explained by time of day")
print(f"  because we are comparing the same hours.")
print(f"  This is what motivated the move to entropy as a")
print(f"  species-agnostic measure of soundscape complexity.")

# ── 5. bar chart ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
fig.patch.set_facecolor("#2C5F2D")
ax.set_facecolor("#2C5F2D")

bars = ax.bar(
    ["Baseline\n(normal 7-8pm)", "Simulation\n(disturbed 7-8pm)"],
    [n_base, n_sim],
    color=["#4A90D9", "#E05C5C"],
    width=0.5,
    edgecolor="none"
)

# Value labels on top of bars
for bar, val in zip(bars, [n_base, n_sim]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        str(val),
        ha="center", va="bottom",
        color="white", fontsize=20, fontweight="bold"
    )

ax.set_ylabel("Number of distinct acoustic clusters",
              color="white", fontsize=11)
ax.set_title(
    "Acoustic Diversity: Baseline vs Simulation\nAudioMoth 4 — Time-Controlled (same hours only)",
    color="white", fontsize=12, pad=12
)
ax.tick_params(colors="white", labelsize=11)
for spine in ax.spines.values():
    spine.set_edgecolor("#97BC62")
ax.set_ylim(0, n_base * 1.3)

# Annotation showing the drop
mid_y = (n_base + n_sim) / 2
ax.annotate(
    f"−{drop_n} clusters\n({drop_pct}% fewer)",
    xy=(1, n_sim + 0.5),
    xytext=(1.32, mid_y),
    color="#97BC62", fontsize=10, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color="#97BC62", lw=1.5)
)

plt.tight_layout()
plt.savefig(OUT_PLOT, dpi=150, bbox_inches="tight", facecolor="#2C5F2D")
plt.show()
print(f"\nPlot saved -> {OUT_PLOT}")