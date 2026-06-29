"""
28_audiomoth5_analysis.py
--------------------------
AudioMoth 5 disturbance analysis — time-controlled, 2:1 baseline ratio.

Applies everything learned from AM4 and AM2:
  - Time-control from the start (same hours only)
  - 2:1 baseline:simulation ratio (learned from AM2 balance experiment)
  - Same 106-feature pipeline (spectral + z-scores + Perch PCA)
  - Same logistic regression with threshold tuning + clip-level voting

AM5 specifics:
  - 306 simulation clips: Human trail, Chainsaw, Gunshot combos
  - Recorder column in metadata: 'Audio_Moth_5'
  - Wav files in: filtered_clips_5/

Outputs:
  am5_time_controlled_emb.npy
  am5_time_controlled_labels.npy
  am5_feature_matrix.csv
  am5_classifier_results.txt
  am5_disturbance_plot.png
  am5_precision_recall_curve.png
"""

import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
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
CLIPS_DIR  = os.path.join(BASE_DIR, "filtered_clips_5")
META_CSV   = os.path.join(BASE_DIR, "cleaned_df (1).csv")
SAVE_EMB   = os.path.join(BASE_DIR, "am5_time_controlled_emb.npy")
SAVE_LAB   = os.path.join(BASE_DIR, "am5_time_controlled_labels.npy")
OUT_CSV    = os.path.join(BASE_DIR, "am5_feature_matrix.csv")
OUT_REPORT = os.path.join(BASE_DIR, "am5_classifier_results.txt")
OUT_PLOT   = os.path.join(BASE_DIR, "am5_disturbance_plot.png")
OUT_CURVE  = os.path.join(BASE_DIR, "am5_precision_recall_curve.png")

SR          = None
ONSET_DELTA = 0.1
ONSET_WAIT  = 15
WINDOW      = 0.2
RANDOM_SEED = 42
VOTE_FRAC   = 0.5
BASELINE_RATIO = 2   # learned from AM2 balance experiment

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

if __name__ == '__main__':

    # ── 1. load metadata ──────────────────────────────────────────────────────
    print("Loading metadata...")
    df  = pd.read_csv(META_CSV)
    am5 = df[df['Recorder'] == 'Audio_Moth_5'].copy()
    am5['is_sim'] = am5['Sim Type'] != '[]'
    am5['hour']   = am5['clip_name'].str[22:24].astype(int)

    # Only keep clips that exist in folder
    am5['exists'] = am5['clip_name'].apply(
        lambda x: os.path.exists(os.path.join(CLIPS_DIR, x)))
    am5 = am5[am5['exists']].copy()

    print(f"  AM5 clips available in folder: {len(am5)}")
    print(f"  Simulation clips:              {am5['is_sim'].sum()}")
    print(f"  Baseline clips:                {(~am5['is_sim']).sum()}")

    print("\nSimulation types in AM5:")
    print(am5[am5['is_sim']]['Sim Type'].value_counts().to_string())

    # ── 2. time-controlled clip selection — 2:1 ratio ─────────────────────────
    print(f"\nApplying time-control (2:1 baseline ratio)...")

    sim_clips = am5[am5['is_sim']][['clip_name', 'hour', 'Sim Type']].copy()
    sim_hours = sim_clips['hour'].unique()

    print(f"  Simulation hours: {sorted(sim_hours)}")
    print(f"  Total simulation clips: {len(sim_clips)}")

    print("\nSimulation breakdown by hour:")
    for hour in sorted(sim_hours):
        hour_sims = sim_clips[sim_clips['hour'] == hour]
        print(f"  Hour {hour}: {len(hour_sims)} clips — "
              f"{hour_sims['Sim Type'].value_counts().to_dict()}")

    baseline_parts = []
    for hour in sorted(sim_hours):
        hour_sim_count = len(am5[(am5['is_sim']) & (am5['hour'] == hour)])
        hour_baseline  = am5[(~am5['is_sim']) & (am5['hour'] == hour)]
        n_sample       = min(BASELINE_RATIO * hour_sim_count, len(hour_baseline))
        if n_sample > 0:
            sampled = hour_baseline.sample(n=n_sample, random_state=RANDOM_SEED)
            baseline_parts.append(sampled[['clip_name', 'hour']])
            print(f"  Hour {hour}: {hour_sim_count} sim, "
                  f"{n_sample} baseline (of {len(hour_baseline)} available)")
        else:
            print(f"  Hour {hour}: {hour_sim_count} sim, "
                  f"NO baseline available!")

    if not baseline_parts:
        raise ValueError("No baseline clips found at simulation hours.")

    baseline_df = pd.concat(baseline_parts)
    print(f"\nTotal baseline clips: {len(baseline_df)}")
    print(f"Total simulation clips: {len(sim_clips)}")
    print(f"Actual ratio: {len(baseline_df)/len(sim_clips):.2f}:1")

    # Sims first (label=1), then baseline (label=0)
    clips_ordered = (
        [(c, 1, h) for c, h in zip(sim_clips['clip_name'], sim_clips['hour'])] +
        [(c, 0, h) for c, h in zip(baseline_df['clip_name'], baseline_df['hour'])]
    )
    print(f"Total clips to process: {len(clips_ordered)}")

    # ── 3. embed with Perch v2 ────────────────────────────────────────────────
    if os.path.exists(SAVE_EMB):
        print("\nLoading saved AM5 embeddings...")
        embeddings = np.load(SAVE_EMB)
        labels     = np.load(SAVE_LAB)
        print(f"  Loaded {len(embeddings)} segments | "
              f"{int(labels.sum())} sim | {int((labels==0).sum())} baseline")
    else:
        print("\nLoading Perch v2 model...")
        from birdnet import load_perch_v2
        model = load_perch_v2(device="CPU")
        print("Model loaded. Collecting segments...")

        all_segments = []
        all_labels   = []

        for i, (clip_name, label, hour) in enumerate(clips_ordered):
            wav_path = os.path.join(CLIPS_DIR, clip_name)
            if not os.path.exists(wav_path):
                continue
            if (i + 1) % 100 == 0:
                print(f"  Collecting {i+1}/{len(clips_ordered)}")
            try:
                y, sr = librosa.load(wav_path, sr=None, mono=True)
                onsets = librosa.onset.onset_detect(
                    y=y, sr=sr, units='time',
                    delta=ONSET_DELTA, wait=ONSET_WAIT)
                for onset in onsets:
                    start = int(onset * sr)
                    end   = int((onset + WINDOW) * sr)
                    if end < len(y):
                        all_segments.append(
                            (y[start:end].astype(np.float32), sr))
                        all_labels.append(label)
            except Exception as e:
                print(f"  WARNING: could not load {clip_name}: {e}")
                continue

        print(f"Collected {len(all_segments)} segments. Embedding...")
        results      = model.encode_arrays(
            iter(all_segments), n_producers=1, batch_size=32)
        results_list = list(results)
        embeddings   = np.array(
            [r['embedding'].flatten() for r in results_list])
        labels       = np.array(all_labels)

        np.save(SAVE_EMB, embeddings)
        np.save(SAVE_LAB, labels)
        print(f"Saved {len(embeddings)} embeddings.")

    # ── 4. UMAP + HDBSCAN ─────────────────────────────────────────────────────
    print("\nRunning UMAP + HDBSCAN...")
    import umap
    import hdbscan

    X    = embeddings.reshape(len(embeddings), 1536)
    X_2d = umap.UMAP(n_components=2, random_state=RANDOM_SEED).fit_transform(X)

    cluster_ids = hdbscan.HDBSCAN(min_cluster_size=5).fit_predict(X_2d)
    n_clusters  = len(set(cluster_ids)) - (1 if -1 in cluster_ids else 0)
    print(f"  Candidate clusters: {n_clusters}")

    base_clusters = set(cluster_ids[labels == 0]) - {-1}
    sim_clusters  = set(cluster_ids[labels == 1]) - {-1}
    print(f"  Clusters in baseline:    {len(base_clusters)}")
    print(f"  Clusters in simulation:  {len(sim_clusters)}")
    print(f"  Drop: {len(base_clusters)-len(sim_clusters)} fewer during disturbance "
          f"({(len(base_clusters)-len(sim_clusters))/max(len(base_clusters),1)*100:.1f}%)")

    # Top cluster shifts
    print("\nTop cluster frequency shifts (AM5):")
    total_b = (labels == 0).sum()
    total_s = labels.sum()
    freq_rows = []
    for c in sorted(set(cluster_ids)):
        if c == -1:
            continue
        b = ((cluster_ids == c) & (labels == 0)).sum()
        s = ((cluster_ids == c) & (labels == 1)).sum()
        freq_rows.append({
            'cluster':      c,
            'baseline_%':   round(100 * b / total_b, 2),
            'simulation_%': round(100 * s / total_s, 2),
            'difference':   round(100 * s / total_s - 100 * b / total_b, 2)
        })
    freq_df = pd.DataFrame(freq_rows).sort_values('difference', ascending=False)
    print(freq_df.head(10).to_string(index=False))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#2C5F2D")
    for ax in axes:
        ax.set_facecolor("#2C5F2D")
    axes[0].scatter(X_2d[labels==0, 0], X_2d[labels==0, 1],
                    c='steelblue', alpha=0.4, s=15,
                    label='Baseline (same hour)')
    axes[0].scatter(X_2d[labels==1, 0], X_2d[labels==1, 1],
                    c='red', alpha=0.6, s=15, label='Simulation')
    axes[0].set_title("AudioMoth 5 — Time-Controlled\nBaseline vs Simulation",
                      color='white')
    axes[0].legend(facecolor='#1F451F', labelcolor='white')
    axes[0].tick_params(colors='white')
    axes[0].set_xlabel("UMAP 1", color='white')
    axes[0].set_ylabel("UMAP 2", color='white')
    sc = axes[1].scatter(X_2d[:, 0], X_2d[:, 1],
                          c=cluster_ids, cmap='tab10', alpha=0.5, s=15)
    axes[1].set_title(f"AudioMoth 5 — {n_clusters} candidate clusters",
                      color='white')
    axes[1].tick_params(colors='white')
    axes[1].set_xlabel("UMAP 1", color='white')
    axes[1].set_ylabel("UMAP 2", color='white')
    plt.colorbar(sc, ax=axes[1])
    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=150, bbox_inches='tight', facecolor='#2C5F2D')
    plt.show()
    print(f"Plot saved -> {OUT_PLOT}")

    # ── 5. spectral feature extraction ────────────────────────────────────────
    print("\nExtracting spectral features...")
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
            "sim_type":          am5.loc[
                am5['clip_name'] == clip_name, 'Sim Type'
            ].values[0] if clip_label == 1 else 'baseline',
            "n_segments":        n_segments,
        }
        for _ in range(n_segments):
            rows.append(feat)

    if missing:
        print(f"  WARNING: {missing} wav files not found")

    feat_df = pd.DataFrame(rows)
    print(f"  Built {len(feat_df)} feature rows from "
          f"{len(clips_ordered)-missing} clips")

    # ── 6. alignment check ────────────────────────────────────────────────────
    print(f"\nEmbedding rows : {len(embeddings)}")
    print(f"Feature rows   : {len(feat_df)}")
    if len(feat_df) != len(embeddings):
        print("WARNING: trimming to shorter length.")
        n          = min(len(feat_df), len(embeddings))
        feat_df    = feat_df.iloc[:n].reset_index(drop=True)
        embeddings = embeddings[:n]
        labels     = labels[:n]
    else:
        print("Row counts match -- alignment confirmed.")

    label_match = (feat_df['clip_label'].values == labels).mean()
    print(f"Label agreement: {label_match*100:.1f}%")

    # ── 7. hour-normalized z-scores ───────────────────────────────────────────
    print("\nComputing hour-normalized z-scores...")
    norm_rows = []
    for hour in feat_df['clip_hour'].unique():
        base_mask = ((feat_df['clip_hour'] == hour) &
                     (feat_df['clip_label'] == 0))
        base_rows = feat_df.loc[base_mask, SPECTRAL_KEYS]
        if len(base_rows) == 0:
            all_mask = feat_df['clip_hour'] == hour
            zero_df  = feat_df.loc[all_mask, SPECTRAL_KEYS].copy() * 0
            zero_df.columns = [f"z_{c}" for c in zero_df.columns]
            norm_rows.append(zero_df)
            continue
        hour_mean = base_rows.mean()
        hour_std  = base_rows.std().replace(0, 1e-6)
        all_mask  = feat_df['clip_hour'] == hour
        z_scored  = (feat_df.loc[all_mask, SPECTRAL_KEYS] - hour_mean) / hour_std
        z_scored.columns = [f"z_{c}" for c in z_scored.columns]
        norm_rows.append(z_scored)

    norm_df = pd.concat(norm_rows).sort_index()
    feat_df = pd.concat([feat_df.reset_index(drop=True),
                         norm_df.reset_index(drop=True)], axis=1)
    print(f"  Added {len(norm_df.columns)} hour-normalized features")

    # ── 8. Perch PCA ──────────────────────────────────────────────────────────
    print("Adding Perch PCA features (top 20 components)...")
    pca    = PCA(n_components=20, random_state=RANDOM_SEED)
    emb_pc = pca.fit_transform(embeddings)
    emb_df = pd.DataFrame(emb_pc, columns=[f"emb_pc{i}" for i in range(20)])
    feat_df = pd.concat([feat_df.reset_index(drop=True), emb_df], axis=1)

    feat_df.to_csv(OUT_CSV, index=False)
    print(f"Saved -> {OUT_CSV}  shape={feat_df.shape}")

    # ── 9. classify ───────────────────────────────────────────────────────────
    meta_cols    = {"clip_name", "clip_label", "clip_hour",
                    "n_segments", "sim_type"}
    feature_cols = [c for c in feat_df.columns if c not in meta_cols]
    X_feat = feat_df[feature_cols].values.astype(np.float32)
    y      = labels.astype(int)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_feat)
    cv       = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    clf      = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED)

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

    # Default threshold
    report_base = classification_report(
        y, all_preds,
        target_names=["baseline", "simulation"], zero_division=0)
    p0, r0, f0, _ = precision_recall_fscore_support(
        y, all_preds, average="binary", zero_division=0)
    print("\nSegment-level results (t=0.50)")
    print(report_base)

    # Threshold tuning
    precisions, recalls, thresholds = precision_recall_curve(y, all_probs)
    valid = np.where(
        (precisions[:-1] >= 0.70) & (recalls[:-1] >= 0.70))[0]

    if len(valid) > 0:
        best_idx    = valid[np.argmax(precisions[valid] + recalls[valid])]
        best_thresh = float(thresholds[best_idx])
        preds_tuned = (all_probs >= best_thresh).astype(int)
        p1, r1, f1, _ = precision_recall_fscore_support(
            y, preds_tuned, average="binary", zero_division=0)
        beat_thresh = True
        print(f"Threshold tuned (t={best_thresh:.3f}): "
              f"P={p1:.3f}  R={r1:.3f}  F1={f1:.3f}  --> BEAT BASELINE")
    else:
        best_idx    = np.argmax(precisions[:-1] + recalls[:-1])
        best_thresh = float(thresholds[best_idx])
        preds_tuned = (all_probs >= best_thresh).astype(int)
        p1, r1, f1, _ = precision_recall_fscore_support(
            y, preds_tuned, average="binary", zero_division=0)
        beat_thresh = False
        print(f"No threshold achieves both >= 0.70")
        print(f"Best (t={best_thresh:.3f}): "
              f"P={p1:.3f}  R={r1:.3f}  F1={f1:.3f}")

    # Clip-level voting
    clip_results = []
    for clip_name in feat_df['clip_name'].unique():
        mask       = feat_df['clip_name'].values == clip_name
        true_label = feat_df['clip_label'].values[mask][0]
        sim_type   = feat_df['sim_type'].values[mask][0]
        seg_probs  = all_probs[mask]
        frac_above = (seg_probs >= best_thresh).mean()
        pred_label = int(frac_above >= VOTE_FRAC)
        clip_results.append({
            'clip_name':  clip_name,
            'true':       true_label,
            'pred':       pred_label,
            'sim_type':   sim_type,
            'n_segments': int(mask.sum()),
        })

    clip_df = pd.DataFrame(clip_results)
    p2, r2, f2, _ = precision_recall_fscore_support(
        clip_df['true'], clip_df['pred'],
        average="binary", zero_division=0)
    report_clip = classification_report(
        clip_df['true'], clip_df['pred'],
        target_names=["baseline", "simulation"], zero_division=0)
    beat_clip = (p2 >= 0.70 and r2 >= 0.70)
    print(report_clip)
    print(f"Clip voting: P={p2:.3f}  R={r2:.3f}  F1={f2:.3f}  "
          f"({'BEAT' if beat_clip else 'not beating'} baseline)")

    # Simulation type breakdown
    print("\nPerformance by simulation type:")
    sim_clip_df = clip_df[clip_df['true'] == 1]
    for st in sim_clip_df['sim_type'].unique():
        mask     = sim_clip_df['sim_type'] == st
        n_total  = mask.sum()
        n_found  = sim_clip_df.loc[mask, 'pred'].sum()
        print(f"  {st:<45s}  {n_found}/{n_total}  "
              f"({n_found/n_total:.1%})")

    # Summary
    print("\nSummary")
    print(f"{'Method':<38} {'Precision':>10} {'Recall':>8} {'F1':>6}")
    print("-" * 65)
    print(f"{'Segment default (t=0.50)':<38} {p0:>10.3f} {r0:>8.3f} {f0:>6.3f}")
    print(f"{'Segment tuned  (t={:.2f})'.format(best_thresh):<38} "
          f"{p1:>10.3f} {r1:>8.3f} {f1:>6.3f}")
    print(f"{'Clip voting    (frac={})'.format(VOTE_FRAC):<38} "
          f"{p2:>10.3f} {r2:>8.3f} {f2:>6.3f}")
    print(f"{'Baseline to beat':<38} {'0.700':>10} {'0.700':>8}")
    print("\nCross-site comparison:")
    print(f"  AM4 (91  sim, 5:1): P=0.704  R=0.851")
    print(f"  AM2 (453 sim, 2:1): P=0.715  R=0.909")
    print(f"  AM5 ({len(sim_clips)} sim, {BASELINE_RATIO}:1): "
          f"P={p1:.3f}  R={r1:.3f}")

    # Top features
    clf.fit(X_scaled, y)
    importance = np.abs(clf.coef_[0])
    top_idx    = np.argsort(importance)[::-1][:20]
    print("\nTop 20 features:")
    for rank, i in enumerate(top_idx, 1):
        print(f"  {rank:2d}. {feature_cols[i]:<35s}  "
              f"coef={clf.coef_[0][i]:+.4f}")

    # PR curve
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("#2C5F2D")
    ax.set_facecolor("#2C5F2D")
    ax.plot(recalls, precisions, color="#97BC62",
            linewidth=2.5, label="AM5 classifier")
    ax.axhline(0.70, color="white", linestyle="--",
               linewidth=1.2, alpha=0.6, label="Target precision 0.70")
    ax.axvline(0.70, color="white", linestyle=":",
               linewidth=1.2, alpha=0.6, label="Target recall 0.70")
    if beat_thresh:
        idx = np.argmin(np.abs(thresholds - best_thresh))
        ax.scatter(recalls[idx], precisions[idx],
                   color="white", s=80, zorder=5)
        ax.annotate(f"t={best_thresh:.2f}\nP={p1:.3f} R={r1:.3f}",
                    xy=(recalls[idx], precisions[idx]),
                    xytext=(recalls[idx]-0.2, precisions[idx]-0.08),
                    color="white", fontsize=9,
                    arrowprops=dict(arrowstyle="->", color="white", lw=1))
    ax.tick_params(colors="white")
    ax.set_xlabel("Recall", color="white", fontsize=12)
    ax.set_ylabel("Precision", color="white", fontsize=12)
    ax.set_title("Precision-Recall Curve -- AM5 Classifier",
                 color="white", fontsize=13)
    ax.legend(facecolor="#1F451F", labelcolor="white", framealpha=0.8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#97BC62")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(OUT_CURVE, dpi=150, bbox_inches="tight", facecolor="#2C5F2D")
    plt.show()
    print(f"Curve saved -> {OUT_CURVE}")

    # Save report
    with open(OUT_REPORT, "w", encoding="utf-8") as fh:
        fh.write("AM5 Classifier Results -- Time-Controlled, 2:1 ratio\n")
        fh.write("=" * 50 + "\n\n")
        fh.write(f"n_sim_clips:    {len(sim_clips)}\n")
        fh.write(f"n_base_clips:   {len(baseline_df)}\n")
        fh.write(f"baseline_ratio: {BASELINE_RATIO}:1\n")
        fh.write(f"n_segments:     {len(feat_df)}\n")
        fh.write(f"n_features:     {len(feature_cols)}\n")
        fh.write(f"sim_hours:      {sorted(sim_hours)}\n\n")
        fh.write("Segment level (t=0.50)\n")
        fh.write(report_base)
        fh.write(f"\nThreshold tuned (t={best_thresh:.3f})\n")
        fh.write(f"P={p1:.3f}  R={r1:.3f}  F1={f1:.3f}\n")
        fh.write("BEAT baseline\n\n" if beat_thresh
                 else "Not yet beating baseline\n\n")
        fh.write("Clip voting\n")
        fh.write(report_clip)
        fh.write(f"\nP={p2:.3f}  R={r2:.3f}  F1={f2:.3f}\n")
        fh.write("BEAT baseline\n\n" if beat_clip
                 else "Not yet beating baseline\n\n")
        fh.write("By simulation type:\n")
        for st in sim_clip_df['sim_type'].unique():
            mask    = sim_clip_df['sim_type'] == st
            n_total = mask.sum()
            n_found = sim_clip_df.loc[mask, 'pred'].sum()
            fh.write(f"  {st:<45s}  {n_found}/{n_total}  "
                     f"({n_found/n_total:.1%})\n")
        fh.write("\nTop 20 features:\n")
        for rank, i in enumerate(top_idx, 1):
            fh.write(f"  {rank:2d}. {feature_cols[i]:<35s}  "
                     f"coef={clf.coef_[0][i]:+.4f}\n")

    print(f"\nReport saved -> {OUT_REPORT}")
    print("Done.")