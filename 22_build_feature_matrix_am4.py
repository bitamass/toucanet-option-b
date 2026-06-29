"""
22_build_feature_matrix_am4.py
--------------------------------
Builds a per-segment feature matrix for AM4 using:
  1. Spectral features per wav (MFCCs, centroid, entropy, etc.)
  2. Hour-normalized features (z-score relative to same-hour baseline mean/std)
  3. Perch embedding PCA (top 20 components)

Then trains a logistic regression with:
  - Threshold tuning
  - Clip-level voting

Key insight from v1 results: centroid_mean and rms_mean are top features but
are likely confounded by time-of-day acoustic level differences. Hour-normalized
features remove that absolute-level bias so the model learns disturbance-specific
deviations from the normal soundscape at that hour.

Inputs:
  am4_time_controlled_emb.npy
  am4_time_controlled_labels.npy
  cleaned_df (1).csv
  filtered_clips_4/

Outputs:
  am4_feature_matrix.csv
  am4_classifier_results.txt
"""

import os
import numpy as np
import pandas as pd
import librosa
from scipy.stats import entropy as scipy_entropy
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (classification_report,
                              precision_recall_fscore_support,
                              precision_recall_curve)

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = r"C:\Users\BitaMassoudi\KasmirWorldProjects\toucanet-option-b"
CLIPS_DIR  = os.path.join(BASE_DIR, "filtered_clips_4")
META_CSV   = os.path.join(BASE_DIR, "cleaned_df (1).csv")
EMB_FILE   = os.path.join(BASE_DIR, "am4_time_controlled_emb.npy")
LAB_FILE   = os.path.join(BASE_DIR, "am4_time_controlled_labels.npy")
OUT_CSV    = os.path.join(BASE_DIR, "am4_feature_matrix.csv")
OUT_REPORT = os.path.join(BASE_DIR, "am4_classifier_results.txt")

SR          = None
ONSET_DELTA = 0.1
ONSET_WAIT  = 15
WINDOW      = 0.2
RANDOM_SEED = 42
VOTE_FRAC   = 0.5

# ── 1. load saved embeddings + labels ─────────────────────────────────────────
print("Loading embeddings and labels...")
embeddings = np.load(EMB_FILE)
labels     = np.load(LAB_FILE)
print(f"  {len(embeddings)} segments  |  {int(labels.sum())} sim  |  {int((labels==0).sum())} baseline")

# ── 2. rebuild clip list in exactly the same order as script 20 ───────────────
print("Rebuilding clip list (matching script 20 order)...")
df  = pd.read_csv(META_CSV)
am4 = df[df['Recorder'] == 'Audio_Moth_4'].copy()
am4['is_sim'] = am4['Sim Type'] != '[]'
am4['hour']   = am4['clip_name'].str[22:24].astype(int)

sim_clips = am4[am4['is_sim']][['clip_name', 'hour']].copy()
sim_hours = sim_clips['hour'].unique()

baseline_parts = []
for hour in sorted(sim_hours):
    hour_sim_count = len(am4[(am4['is_sim']) & (am4['hour'] == hour)])
    hour_baseline  = am4[(~am4['is_sim']) & (am4['hour'] == hour)]
    n_sample       = min(hour_sim_count * 5, len(hour_baseline))
    if n_sample > 0:
        sampled = hour_baseline.sample(n=n_sample, random_state=42)
        baseline_parts.append(sampled[['clip_name', 'hour']])

baseline_df   = pd.concat(baseline_parts)
clips_ordered = (
    [(c, 1, h) for c, h in zip(sim_clips['clip_name'], sim_clips['hour'])] +
    [(c, 0, h) for c, h in zip(baseline_df['clip_name'], baseline_df['hour'])]
)
print(f"  {len(clips_ordered)} clips  ({len(sim_clips)} sim, {len(baseline_df)} baseline)")

# ── 3. extract spectral features per wav ──────────────────────────────────────
print("\nExtracting spectral features from wav files...")

SPECTRAL_KEYS = (
    [f"mfcc_mean_{i}" for i in range(13)] +
    [f"mfcc_std_{i}"  for i in range(13)] +
    ["centroid_mean", "centroid_std",
     "rolloff_mean",  "rolloff_std",
     "bandwidth_mean","bandwidth_std",
     "zcr_mean",      "zcr_std",
     "rms_mean",      "rms_std",
     "silence_fraction",
     "spec_entropy_mean", "spec_entropy_std",
     "temporal_entropy",  "onset_count"]
)

rows    = []
missing = 0

for clip_idx, (clip_name, clip_label, clip_hour) in enumerate(clips_ordered):
    if (clip_idx + 1) % 200 == 0:
        print(f"  {clip_idx+1}/{len(clips_ordered)} clips processed...")

    wav_path = os.path.join(CLIPS_DIR, clip_name)
    if not os.path.exists(wav_path):
        missing += 1
        continue

    try:
        y, sr = librosa.load(wav_path, sr=SR, mono=True)
    except Exception:
        missing += 1
        continue

    onsets       = librosa.onset.onset_detect(
        y=y, sr=sr, units='time', delta=ONSET_DELTA, wait=ONSET_WAIT)
    valid_onsets = [o for o in onsets if int((o + WINDOW) * sr) < len(y)]
    n_segments   = len(valid_onsets)
    if n_segments == 0:
        continue

    mfcc      = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    centroid  = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rolloff   = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    zcr       = librosa.feature.zero_crossing_rate(y)[0]
    rms       = librosa.feature.rms(y=y)[0]
    S         = np.abs(librosa.stft(y)) ** 2
    S_norm    = S / (S.sum(axis=0, keepdims=True) + 1e-10)
    frame_ent = -np.sum(S_norm * np.log(S_norm + 1e-10), axis=0)
    rms_norm  = rms / (rms.sum() + 1e-10)

    feat = {
        **{f"mfcc_mean_{i}": mfcc.mean(axis=1)[i] for i in range(13)},
        **{f"mfcc_std_{i}":  mfcc.std(axis=1)[i]  for i in range(13)},
        "centroid_mean":     centroid.mean(),
        "centroid_std":      centroid.std(),
        "rolloff_mean":      rolloff.mean(),
        "rolloff_std":       rolloff.std(),
        "bandwidth_mean":    bandwidth.mean(),
        "bandwidth_std":     bandwidth.std(),
        "zcr_mean":          zcr.mean(),
        "zcr_std":           zcr.std(),
        "rms_mean":          rms.mean(),
        "rms_std":           rms.std(),
        "silence_fraction":  (rms < 0.01).mean(),
        "spec_entropy_mean": frame_ent.mean(),
        "spec_entropy_std":  frame_ent.std(),
        "temporal_entropy":  scipy_entropy(rms_norm + 1e-10),
        "onset_count":       n_segments,
        "clip_name":         clip_name,
        "clip_label":        clip_label,
        "clip_hour":         clip_hour,
        "n_segments":        n_segments,
    }

    for _ in range(n_segments):
        rows.append(feat)

if missing:
    print(f"  WARNING: {missing} wav files not found / unreadable")

feat_df = pd.DataFrame(rows)
print(f"  Built {len(feat_df)} feature rows from {len(clips_ordered) - missing} clips")

# ── 4. alignment check ────────────────────────────────────────────────────────
print(f"\nEmbedding rows : {len(embeddings)}")
print(f"Feature rows   : {len(feat_df)}")

if len(feat_df) != len(embeddings):
    print("WARNING: counts differ -- trimming to shorter length.")
    n          = min(len(feat_df), len(embeddings))
    feat_df    = feat_df.iloc[:n].reset_index(drop=True)
    embeddings = embeddings[:n]
    labels     = labels[:n]
else:
    print("Row counts match -- alignment confirmed.")

label_match = (feat_df['clip_label'].values == labels).mean()
print(f"Label agreement: {label_match*100:.1f}%")
if label_match < 0.95:
    print("WARNING: labels don't match well -- check clip ordering.")

# ── 5. hour-normalized features ───────────────────────────────────────────────
# For each spectral feature, compute the mean and std across BASELINE clips
# at the same hour. Then z-score every clip relative to that baseline
# distribution. This removes absolute acoustic level as a confound and forces
# the model to learn deviations from the normal soundscape at that hour.
print("\nComputing hour-normalized features...")

norm_rows = []

for hour in feat_df['clip_hour'].unique():
    # baseline clips at this hour
    base_mask = (feat_df['clip_hour'] == hour) & (feat_df['clip_label'] == 0)
    base_rows = feat_df.loc[base_mask, SPECTRAL_KEYS]

    hour_mean = base_rows.mean()
    hour_std  = base_rows.std().replace(0, 1e-6)   # avoid divide-by-zero

    # all clips at this hour (baseline + sim)
    all_mask  = feat_df['clip_hour'] == hour
    all_rows  = feat_df.loc[all_mask, SPECTRAL_KEYS]

    z_scored  = (all_rows - hour_mean) / hour_std
    z_scored.columns = [f"z_{c}" for c in z_scored.columns]
    norm_rows.append(z_scored)

norm_df = pd.concat(norm_rows).sort_index()
feat_df = pd.concat([feat_df.reset_index(drop=True),
                     norm_df.reset_index(drop=True)], axis=1)

print(f"  Added {len(norm_df.columns)} hour-normalized features")

# ── 6. add Perch PCA features ─────────────────────────────────────────────────
print("Adding Perch embedding PCA features (top 20 components)...")
pca    = PCA(n_components=20, random_state=RANDOM_SEED)
emb_pc = pca.fit_transform(embeddings)
emb_df = pd.DataFrame(emb_pc, columns=[f"emb_pc{i}" for i in range(20)])
feat_df = pd.concat([feat_df.reset_index(drop=True), emb_df], axis=1)

# ── 7. save feature matrix ────────────────────────────────────────────────────
feat_df.to_csv(OUT_CSV, index=False)
print(f"Saved --> {OUT_CSV}  shape={feat_df.shape}")

# ── 8. prepare X, y ───────────────────────────────────────────────────────────
meta_cols    = {"clip_name", "clip_label", "clip_hour", "n_segments"}
feature_cols = [c for c in feat_df.columns if c not in meta_cols]
X = feat_df[feature_cols].values.astype(np.float32)
y = labels.astype(int)

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
clf = LogisticRegression(max_iter=1000, class_weight="balanced",
                          random_state=RANDOM_SEED)

# ── 9. CV ─────────────────────────────────────────────────────────────────────
print("\nRunning 5-fold stratified CV...")
all_preds = np.zeros_like(y)
all_probs = np.zeros(len(y), dtype=np.float32)

for fold, (tr, va) in enumerate(cv.split(X_scaled, y)):
    clf.fit(X_scaled[tr], y[tr])
    all_preds[va] = clf.predict(X_scaled[va])
    all_probs[va] = clf.predict_proba(X_scaled[va])[:, 1]
    p, r, f, _ = precision_recall_fscore_support(
        y[va], all_preds[va], average="binary", zero_division=0)
    print(f"  Fold {fold+1}: precision={p:.3f}  recall={r:.3f}  F1={f:.3f}")

# ── 10. baseline results (t=0.50) ─────────────────────────────────────────────
report_base = classification_report(y, all_preds,
                                     target_names=["baseline", "simulation"],
                                     zero_division=0)
p0, r0, f0, _ = precision_recall_fscore_support(
    y, all_preds, average="binary", zero_division=0)

print("\nSegment-level results (threshold=0.50)")
print(report_base)
print(f"Precision {p0:.3f}  Recall {r0:.3f}  F1 {f0:.3f}")

# ── 11. threshold tuning ──────────────────────────────────────────────────────
print("\nThreshold tuning")
precisions, recalls, thresholds = precision_recall_curve(y, all_probs)

valid = np.where((precisions[:-1] >= 0.70) & (recalls[:-1] >= 0.70))[0]

if len(valid) > 0:
    best_idx    = valid[np.argmax(precisions[valid] + recalls[valid])]
    best_thresh = float(thresholds[best_idx])
    preds_tuned = (all_probs >= best_thresh).astype(int)
    p1, r1, f1, _ = precision_recall_fscore_support(
        y, preds_tuned, average="binary", zero_division=0)
    print(f"Best threshold: {best_thresh:.3f}")
    print(f"Precision {p1:.3f}  Recall {r1:.3f}  F1 {f1:.3f}")
    beat_thresh = (p1 >= 0.70 and r1 >= 0.70)
    print("BEAT baseline" if beat_thresh else "Not yet beating baseline")
else:
    # Find the threshold that maximises precision+recall even if neither hits 0.70
    best_idx    = np.argmax(precisions[:-1] + recalls[:-1])
    best_thresh = float(thresholds[best_idx])
    preds_tuned = (all_probs >= best_thresh).astype(int)
    p1, r1, f1, _ = precision_recall_fscore_support(
        y, preds_tuned, average="binary", zero_division=0)
    beat_thresh = False
    print(f"No threshold achieves both >= 0.70")
    print(f"Best available (t={best_thresh:.3f}): precision={p1:.3f}  recall={r1:.3f}  F1={f1:.3f}")

# ── 12. clip-level voting ─────────────────────────────────────────────────────
print(f"\nClip-level voting (vote_frac={VOTE_FRAC})")
clip_results = []
for clip_name in feat_df['clip_name'].unique():
    mask       = feat_df['clip_name'].values == clip_name
    true_label = feat_df['clip_label'].values[mask][0]
    seg_probs  = all_probs[mask]
    frac_above = (seg_probs >= best_thresh).mean()
    pred_label = int(frac_above >= VOTE_FRAC)
    clip_results.append({
        'clip_name':  clip_name,
        'true':       true_label,
        'pred':       pred_label,
        'n_segments': int(mask.sum()),
        'frac_sim':   round(float(frac_above), 3),
    })

clip_df    = pd.DataFrame(clip_results)
p2, r2, f2, _ = precision_recall_fscore_support(
    clip_df['true'], clip_df['pred'], average="binary", zero_division=0)
report_clip = classification_report(clip_df['true'], clip_df['pred'],
                                     target_names=["baseline", "simulation"],
                                     zero_division=0)
beat_clip = (p2 >= 0.70 and r2 >= 0.70)
print(report_clip)
print(f"Precision {p2:.3f}  Recall {r2:.3f}  F1 {f2:.3f}  ({len(clip_df)} clips)")
print("BEAT baseline" if beat_clip else "Not yet beating baseline")

# ── 13. summary ───────────────────────────────────────────────────────────────
print("\nSummary")
print(f"{'Method':<38} {'Precision':>10} {'Recall':>8} {'F1':>6}")
print("-" * 65)
print(f"{'Segment default (t=0.50)':<38} {p0:>10.3f} {r0:>8.3f} {f0:>6.3f}")
print(f"{'Segment tuned  (t={:.2f})'.format(best_thresh):<38} {p1:>10.3f} {r1:>8.3f} {f1:>6.3f}")
print(f"{'Clip voting    (frac={})'.format(VOTE_FRAC):<38} {p2:>10.3f} {r2:>8.3f} {f2:>6.3f}")
print(f"{'Baseline to beat':<38} {'0.700':>10} {'0.700':>8} {'-':>6}")

# ── 14. top features ──────────────────────────────────────────────────────────
clf.fit(X_scaled, y)
importance = np.abs(clf.coef_[0])
top_idx    = np.argsort(importance)[::-1][:20]
print("\nTop 20 features:")
for rank, i in enumerate(top_idx, 1):
    print(f"  {rank:2d}. {feature_cols[i]:<35s}  coef={clf.coef_[0][i]:+.4f}")

# ── 15. save report (utf-8 to handle special chars) ──────────────────────────
with open(OUT_REPORT, "w", encoding="utf-8") as fh:
    fh.write("AM4 Feature Matrix -- Classifier Results\n")
    fh.write("=" * 50 + "\n\n")
    fh.write(f"n_segments:  {len(feat_df)}\n")
    fh.write(f"n_clips:     {len(clip_df)}\n")
    fh.write(f"n_features:  {len(feature_cols)}\n")
    fh.write(f"sim segs:    {int(y.sum())}\n")
    fh.write(f"base segs:   {int((y==0).sum())}\n\n")
    fh.write("Segment level (t=0.50)\n")
    fh.write(report_base)
    fh.write(f"\nPrecision {p0:.3f}  Recall {r0:.3f}  F1 {f0:.3f}\n\n")
    fh.write(f"Threshold tuned (t={best_thresh:.3f})\n")
    fh.write(f"Precision {p1:.3f}  Recall {r1:.3f}  F1 {f1:.3f}\n")
    fh.write("BEAT baseline\n\n" if beat_thresh else "Not yet beating baseline\n\n")
    fh.write(f"Clip voting (frac={VOTE_FRAC})\n")
    fh.write(report_clip)
    fh.write(f"\nPrecision {p2:.3f}  Recall {r2:.3f}  F1 {f2:.3f}\n")
    fh.write("BEAT baseline\n" if beat_clip else "Not yet beating baseline\n")
    fh.write("\nTop 20 features:\n")
    for rank, i in enumerate(top_idx, 1):
        fh.write(f"  {rank:2d}. {feature_cols[i]:<35s}  coef={clf.coef_[0][i]:+.4f}\n")

print(f"\nReport saved --> {OUT_REPORT}")