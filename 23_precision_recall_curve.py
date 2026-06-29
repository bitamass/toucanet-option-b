# 23_precision_recall_curve.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve
from scipy.stats import entropy as scipy_entropy
import librosa, os

# Load your saved feature matrix directly — no need to re-extract
feat_df    = pd.read_csv(r"C:\Users\BitaMassoudi\KasmirWorldProjects\toucanet-option-b\am4_feature_matrix.csv")
embeddings = np.load(r"C:\Users\BitaMassoudi\KasmirWorldProjects\toucanet-option-b\am4_time_controlled_emb.npy")
labels     = np.load(r"C:\Users\BitaMassoudi\KasmirWorldProjects\toucanet-option-b\am4_time_controlled_labels.npy")

meta_cols    = {"clip_name", "clip_label", "clip_hour", "n_segments"}
feature_cols = [c for c in feat_df.columns if c not in meta_cols]
X = feat_df[feature_cols].values.astype(np.float32)
y = labels.astype(int)

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)
cv       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf      = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)

all_probs = np.zeros(len(y), dtype=np.float32)
for tr, va in cv.split(X_scaled, y):
    clf.fit(X_scaled[tr], y[tr])
    all_probs[va] = clf.predict_proba(X_scaled[va])[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y, all_probs)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(recalls, precisions, color="#97BC62", linewidth=2.5, label="AM4 classifier")
ax.axhline(0.70, color="white", linestyle="--", linewidth=1.2, alpha=0.6, label="Target precision 0.70")
ax.axvline(0.70, color="white", linestyle=":",  linewidth=1.2, alpha=0.6, label="Target recall 0.70")

# Mark the chosen operating point (t=0.71)
idx = np.argmin(np.abs(thresholds - 0.71))
ax.scatter(recalls[idx], precisions[idx], color="white", s=80, zorder=5)
ax.annotate("t=0.71\nP=0.704 R=0.851",
            xy=(recalls[idx], precisions[idx]),
            xytext=(recalls[idx] - 0.18, precisions[idx] - 0.08),
            color="white", fontsize=9,
            arrowprops=dict(arrowstyle="->", color="white", lw=1))

ax.set_facecolor("#2C5F2D")
fig.patch.set_facecolor("#2C5F2D")
ax.tick_params(colors="white")
ax.xaxis.label.set_color("white")
ax.yaxis.label.set_color("white")
ax.title.set_color("white")
for spine in ax.spines.values():
    spine.set_edgecolor("#97BC62")

ax.set_xlabel("Recall", fontsize=12)
ax.set_ylabel("Precision", fontsize=12)
ax.set_title("Precision-Recall Curve — AM4 Classifier", fontsize=13)
ax.legend(facecolor="#1F451F", labelcolor="white", framealpha=0.8)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig(r"C:\Users\BitaMassoudi\KasmirWorldProjects\toucanet-option-b\am4_precision_recall_curve.png",
            dpi=150, bbox_inches="tight", facecolor="#2C5F2D")
plt.show()
print("Saved am4_precision_recall_curve.png")