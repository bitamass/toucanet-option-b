"""
33_corrected_balance_experiment.py
------------------------------------
Reruns the balance experiment for AM2, AM5, and AM6 using
CORRECTED clip-level CV — no data leakage.

The original balance experiment (script 27) used leaky segment-level CV.
The optimal ratio found there (2:1) may not be optimal under honest evaluation.

Tests ratios 1:1, 2:1, 3:1, 5:1 for each recorder using corrected CV.

Uses already-saved feature matrices and embeddings — no re-embedding needed.
Should finish in about 15-20 minutes.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (precision_recall_fscore_support,
                              precision_recall_curve)

BASE_DIR = r"C:\Users\BitaMassoudi\KasmirWorldProjects\toucanet-option-b"

RECORDERS = [
    {
        'name':  'AM2',
        'feat':  'am2_feature_matrix.csv',
        'emb':   'am2_time_controlled_emb.npy',
        'lab':   'am2_time_controlled_labels.npy',
    },
    {
        'name':  'AM5',
        'feat':  'am5_feature_matrix.csv',
        'emb':   'am5_time_controlled_emb.npy',
        'lab':   'am5_time_controlled_labels.npy',
    },
    {
        'name':  'AM6',
        'feat':  'am6_feature_matrix.csv',
        'emb':   'am6_time_controlled_emb.npy',
        'lab':   'am6_time_controlled_labels.npy',
    },
]

RATIOS      = [1, 2, 3, 5]
RANDOM_SEED = 42
VOTE_FRAC   = 0.5
N_FOLDS     = 5

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


def corrected_cv_subset(feat_df, embeddings, labels):
    """
    Run corrected clip-level CV on a subset of clips.
    feat_df, embeddings, labels must already be filtered to the subset.
    Returns (p_tuned, r_tuned, best_thresh, beat, precisions, recalls, thresholds)
    """
    clip_info = feat_df.groupby('clip_name').agg(
        clip_label=('clip_label', 'first'),
        clip_hour=('clip_hour', 'first')
    ).reset_index()
    clip_names  = clip_info['clip_name'].values
    clip_labels = clip_info['clip_label'].values

    if len(np.unique(clip_labels)) < 2:
        return None

    cv = StratifiedKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    all_seg_probs = np.zeros(len(feat_df), dtype=np.float32)

    for fold, (tr_clip_idx, va_clip_idx) in enumerate(
            cv.split(clip_names, clip_labels)):

        tr_clips = set(clip_names[tr_clip_idx])
        va_clips = set(clip_names[va_clip_idx])

        tr_mask = feat_df['clip_name'].isin(tr_clips).values
        va_mask = feat_df['clip_name'].isin(va_clips).values

        feat_tr = feat_df[tr_mask].copy().reset_index(drop=True)
        feat_va = feat_df[va_mask].copy().reset_index(drop=True)
        emb_tr  = embeddings[tr_mask]
        emb_va  = embeddings[va_mask]
        y_tr    = labels[tr_mask].astype(int)

        # Z-scores from training baseline only
        z_tr_rows, z_va_rows = [], []
        for hour in feat_tr['clip_hour'].unique():
            base_mask_tr = ((feat_tr['clip_hour'] == hour) &
                            (feat_tr['clip_label'] == 0))
            base_rows = feat_tr.loc[base_mask_tr, SPECTRAL_KEYS]

            if len(base_rows) == 0:
                for feat_sub, rows_list in [(feat_tr, z_tr_rows),
                                            (feat_va, z_va_rows)]:
                    m = feat_sub['clip_hour'] == hour
                    if m.sum() > 0:
                        z = feat_sub.loc[m, SPECTRAL_KEYS].copy() * 0
                        z.columns = [f"z_{c}" for c in z.columns]
                        rows_list.append(z)
                continue

            hour_mean = base_rows.mean()
            hour_std  = base_rows.std().replace(0, 1e-6)

            tr_h = feat_tr['clip_hour'] == hour
            if tr_h.sum() > 0:
                z_tr = (feat_tr.loc[tr_h, SPECTRAL_KEYS] -
                        hour_mean) / hour_std
                z_tr.columns = [f"z_{c}" for c in z_tr.columns]
                z_tr_rows.append(z_tr)

            va_h = feat_va['clip_hour'] == hour
            if va_h.sum() > 0:
                z_va = (feat_va.loc[va_h, SPECTRAL_KEYS] -
                        hour_mean) / hour_std
                z_va.columns = [f"z_{c}" for c in z_va.columns]
                z_va_rows.append(z_va)

        if not z_tr_rows:
            continue
        z_tr_df = pd.concat(z_tr_rows).sort_index()
        z_va_df = (pd.concat(z_va_rows).sort_index()
                   if z_va_rows else
                   pd.DataFrame(0, index=feat_va.index,
                                columns=[f"z_{c}" for c in SPECTRAL_KEYS]))

        # PCA on training only
        pca       = PCA(n_components=20, random_state=RANDOM_SEED)
        emb_tr_pc = pca.fit_transform(emb_tr)
        emb_va_pc = pca.transform(emb_va)

        X_tr = np.hstack([feat_tr[SPECTRAL_KEYS].values,
                           z_tr_df.values,
                           emb_tr_pc]).astype(np.float32)
        X_va = np.hstack([feat_va[SPECTRAL_KEYS].values,
                           z_va_df.values,
                           emb_va_pc]).astype(np.float32)

        # Scaler on training only
        scaler   = StandardScaler()
        X_tr_s   = scaler.fit_transform(X_tr)
        X_va_s   = scaler.transform(X_va)

        clf = LogisticRegression(
            max_iter=1000, class_weight="balanced",
            random_state=RANDOM_SEED)
        clf.fit(X_tr_s, y_tr)

        va_probs = clf.predict_proba(X_va_s)[:, 1]
        va_indices = np.where(va_mask)[0]
        all_seg_probs[va_indices] = va_probs

    y_all = labels.astype(int)
    precisions, recalls, thresholds = precision_recall_curve(
        y_all, all_seg_probs)
    valid = np.where(
        (precisions[:-1] >= 0.70) & (recalls[:-1] >= 0.70))[0]

    if len(valid) > 0:
        best_idx    = valid[np.argmax(precisions[valid] + recalls[valid])]
        best_thresh = float(thresholds[best_idx])
        preds_tuned = (all_seg_probs >= best_thresh).astype(int)
        p1, r1, _, _ = precision_recall_fscore_support(
            y_all, preds_tuned, average="binary", zero_division=0)
        beat = True
    else:
        best_idx    = np.argmax(precisions[:-1] + recalls[:-1])
        best_thresh = float(thresholds[best_idx])
        preds_tuned = (all_seg_probs >= best_thresh).astype(int)
        p1, r1, _, _ = precision_recall_fscore_support(
            y_all, preds_tuned, average="binary", zero_division=0)
        beat = False

    return p1, r1, best_thresh, beat, precisions, recalls, thresholds


if __name__ == '__main__':

    all_results = {}
    OUT_REPORT  = os.path.join(BASE_DIR,
                               "corrected_balance_experiment_results.txt")

    for rec in RECORDERS:
        name = rec['name']
        print(f"\n{'='*60}")
        print(f"{name} — Corrected Balance Experiment")
        print(f"{'='*60}")

        feat_path = os.path.join(BASE_DIR, rec['feat'])
        emb_path  = os.path.join(BASE_DIR, rec['emb'])
        lab_path  = os.path.join(BASE_DIR, rec['lab'])

        if not os.path.exists(feat_path):
            print(f"  Feature matrix not found: {feat_path}")
            continue

        feat_df    = pd.read_csv(feat_path)
        embeddings = np.load(emb_path)
        labels     = np.load(lab_path)

        if len(feat_df) != len(embeddings):
            n = min(len(feat_df), len(embeddings))
            feat_df    = feat_df.iloc[:n].reset_index(drop=True)
            embeddings = embeddings[:n]
            labels     = labels[:n]

        # Get unique clips
        sim_clips  = feat_df[feat_df['clip_label']==1]['clip_name'].unique()
        base_clips = feat_df[feat_df['clip_label']==0]['clip_name'].unique()
        n_sim      = len(sim_clips)
        n_base_all = len(base_clips)

        print(f"  Simulation clips: {n_sim}")
        print(f"  Baseline clips available: {n_base_all}")

        rec_results = []

        for ratio in RATIOS:
            n_base_sample = min(ratio * n_sim, n_base_all)
            print(f"\n  Testing ratio {ratio}:1  "
                  f"({n_sim} sim : {n_base_sample} baseline clips)")

            # Sample baseline clips
            np.random.seed(RANDOM_SEED)
            sampled_base = np.random.choice(
                base_clips, size=n_base_sample, replace=False)

            keep = set(sim_clips) | set(sampled_base)
            seg_mask = feat_df['clip_name'].isin(keep).values

            feat_sub = feat_df[seg_mask].reset_index(drop=True)
            emb_sub  = embeddings[seg_mask]
            lab_sub  = labels[seg_mask]

            result = corrected_cv_subset(feat_sub, emb_sub, lab_sub)
            if result is None:
                print(f"    Skipped — insufficient class variety")
                continue

            p1, r1, thresh, beat, precs, recs, threshs = result
            beat_str = "YES" if beat else "no"
            print(f"    Result: P={p1:.3f}  R={r1:.3f}  "
                  f"t={thresh:.3f}  Beat={beat_str}")

            rec_results.append({
                'ratio':      ratio,
                'n_base':     n_base_sample,
                'n_sim':      n_sim,
                'p_tuned':    p1,
                'r_tuned':    r1,
                'thresh':     thresh,
                'beat':       beat,
                'precisions': precs,
                'recalls':    recs,
                'thresholds': threshs,
            })

        all_results[name] = rec_results

        # Print summary for this recorder
        print(f"\n  {name} Balance Summary (Corrected CV):")
        print(f"  {'Ratio':<8} {'Base':>6} {'Precision':>10} "
              f"{'Recall':>8} {'Beat?':>6}")
        print(f"  {'-'*45}")
        for r in rec_results:
            print(f"  {str(r['ratio'])+':1':<8} {r['n_base']:>6} "
                  f"{r['p_tuned']:>10.3f} {r['r_tuned']:>8.3f} "
                  f"{'YES' if r['beat'] else 'no':>6}")
        print(f"  {'Target':<8} {'':>6} {'0.700':>10} {'0.700':>8}")

        # Best ratio
        beating = [r for r in rec_results if r['beat']]
        if beating:
            best = max(beating, key=lambda x: x['p_tuned'] + x['r_tuned'])
            print(f"\n  Best ratio: {best['ratio']}:1  "
                  f"P={best['p_tuned']:.3f}  R={best['r_tuned']:.3f}")
        else:
            best_avail = max(rec_results,
                             key=lambda x: x['p_tuned'] + x['r_tuned'])
            print(f"\n  No ratio beats both targets.")
            print(f"  Best available: {best_avail['ratio']}:1  "
                  f"P={best_avail['p_tuned']:.3f}  "
                  f"R={best_avail['r_tuned']:.3f}")

    # ── Overall summary ───────────────────────────────────────────────────────
    print("\n\n" + "=" * 65)
    print("CORRECTED BALANCE EXPERIMENT — OVERALL SUMMARY")
    print("=" * 65)
    for name, rec_results in all_results.items():
        beating = [r for r in rec_results if r['beat']]
        print(f"\n{name}:")
        for r in rec_results:
            mark = "<<< BEST" if r['beat'] and r == max(
                rec_results, key=lambda x: x['p_tuned']+x['r_tuned']
                if x['beat'] else 0) else ""
            print(f"  {str(r['ratio'])+':1':<6}  "
                  f"P={r['p_tuned']:.3f}  R={r['r_tuned']:.3f}  "
                  f"{'BEAT' if r['beat'] else 'miss'}  {mark}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    n_recs  = len(all_results)
    fig, axes = plt.subplots(1, n_recs, figsize=(6*n_recs, 5))
    fig.patch.set_facecolor("#2C5F2D")
    if n_recs == 1:
        axes = [axes]

    colours = ["#97BC62", "#F5A623", "#E05C5C", "#4A90D9"]

    for ax, (name, rec_results) in zip(axes, all_results.items()):
        ax.set_facecolor("#2C5F2D")
        for res, col in zip(rec_results, colours):
            lbl = (f"ratio {res['ratio']}:1  "
                   f"P={res['p_tuned']:.3f} R={res['r_tuned']:.3f}")
            ax.plot(res['recalls'], res['precisions'],
                    color=col, linewidth=1.8, label=lbl)
            if res['beat']:
                idx = np.argmin(
                    np.abs(res['thresholds'] - res['thresh']))
                ax.scatter(res['recalls'][idx], res['precisions'][idx],
                           color=col, s=60, zorder=5)

        ax.axhline(0.70, color="white", linestyle="--",
                   linewidth=1.2, alpha=0.7)
        ax.axvline(0.70, color="white", linestyle=":",
                   linewidth=1.2, alpha=0.7)
        ax.tick_params(colors="white")
        ax.set_xlabel("Recall", color="white", fontsize=11)
        ax.set_ylabel("Precision", color="white", fontsize=11)
        ax.set_title(f"{name} — Corrected Balance Experiment",
                     color="white", fontsize=11)
        ax.legend(facecolor="#1F451F", labelcolor="white",
                  framealpha=0.8, fontsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#97BC62")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    plt.tight_layout()
    out_plot = os.path.join(BASE_DIR,
                            "corrected_balance_experiment_plot.png")
    plt.savefig(out_plot, dpi=150, bbox_inches="tight",
                facecolor="#2C5F2D")
    plt.show()
    print(f"\nPlot saved -> {out_plot}")

    # ── Save report ───────────────────────────────────────────────────────────
    with open(OUT_REPORT, "w", encoding="utf-8") as fh:
        fh.write("Corrected Balance Experiment Results\n")
        fh.write("=" * 50 + "\n\n")
        fh.write("Uses clip-level CV with no data leakage.\n\n")
        for name, rec_results in all_results.items():
            fh.write(f"{name}:\n")
            for r in rec_results:
                fh.write(f"  {str(r['ratio'])+':1':<6}  "
                         f"P={r['p_tuned']:.3f}  "
                         f"R={r['r_tuned']:.3f}  "
                         f"{'BEAT' if r['beat'] else 'miss'}\n")
            fh.write("\n")
    print(f"Report saved -> {OUT_REPORT}")
    print("\nDone.")