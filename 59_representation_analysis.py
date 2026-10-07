"""
59_representation_analysis.py
------------------------------
Representation diagnostic — answering Sammy's four questions.

QUESTIONS:
  1. How much site information is in the embeddings?
     Can we predict recorder from frozen Perch embeddings?

  2. Do embeddings cluster by site or disturbance?
     PCA visualisation coloured by site and by label.

  3. Which features shift most across sites?
     Variance decomposition: site vs disturbance vs residual.

  4. Does adaptive normalisation reduce site separation?
     Between-site distances before and after z-score normalisation.

These diagnostics determine whether the bottleneck is in
the REPRESENTATION itself or in the downstream classifier.

If site prediction accuracy is near 100%:
  -> Embeddings are dominated by site identity
  -> Trying more classifiers will not help
  -> Need alternative embeddings or representation learning

If disturbance clusters cut across sites in PCA:
  -> Universal disturbance signal exists in embedding space
  -> Problem is in feature extraction or classifier, not representation

If adaptive normalisation reduces site separation:
  -> Supports interpretation that it subtracts site-specific background
  -> Motivates NLP/sequence approaches that use local context

OUTPUTS:
  - Console tables for all four questions
  - PCA plots saved as PNG files
  - Variance decomposition table
  - Site distance matrix before/after normalisation
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score

BASE_DIR = r"C:\Users\BitaMassoudi\KasmirWorldProjects\toucanet-option-b"
SEED     = 42

RECORDERS = [
    ('AM4','am4_full_feature_matrix.csv','am4_full_emb.npy','am4_full_labels.npy'),
    ('AM2','am2_feature_matrix.csv','am2_time_controlled_emb.npy','am2_time_controlled_labels.npy'),
    ('AM5','am5_feature_matrix.csv','am5_time_controlled_emb.npy','am5_time_controlled_labels.npy'),
    ('AM6','am6_feature_matrix.csv','am6_time_controlled_emb.npy','am6_time_controlled_labels.npy'),
    ('AM1','am1_feature_matrix.csv','am1_full_emb.npy','am1_full_labels.npy'),
]

SPECTRAL_KEYS = (
    [f"mfcc_mean_{i}" for i in range(13)] +
    [f"mfcc_std_{i}"  for i in range(13)] +
    ["centroid_mean","centroid_std","rolloff_mean","rolloff_std",
     "bandwidth_mean","bandwidth_std","zcr_mean","zcr_std",
     "rms_mean","rms_std","silence_fraction",
     "spec_entropy_mean","spec_entropy_std",
     "temporal_entropy","onset_count"]
)

COLORS_SITE = {'AM4':'#E74C3C','AM2':'#3498DB','AM5':'#2ECC71',
               'AM6':'#F39C12','AM1':'#9B59B6'}
COLORS_DIST = {0:'#95A5A6', 1:'#E74C3C'}


def build_clip_df(feat_df, emb):
    rows = []
    for cn in feat_df['clip_name'].unique():
        mask = feat_df['clip_name'].values == cn
        cl   = feat_df['clip_label'].values[mask][0]
        hour = feat_df['clip_hour'].values[mask][0]
        ce   = emb[mask]
        row  = {'clip_name':cn,'clip_label':cl,'clip_hour':hour}
        for k in SPECTRAL_KEYS:
            if k in feat_df.columns:
                row[k] = feat_df.loc[mask,k].values[0]
        for i,v in enumerate(ce.mean(0)): row[f'em_{i}']=v
        for i,v in enumerate(ce.std(0)):  row[f'es_{i}']=v
        for i,v in enumerate(ce.max(0)):  row[f'ex_{i}']=v
        rows.append(row)
    return pd.DataFrame(rows)


def compute_adaptive_zscores(clip_df, n_clips=40):
    """Adaptive z-scores using unlabelled rolling window."""
    spec_vals = clip_df[SPECTRAL_KEYS].values
    z_rows = []
    for i in range(len(clip_df)):
        ws = max(0, i-n_clips)
        window = spec_vals[ws:i]
        if len(window) < 2:
            z = np.zeros(len(SPECTRAL_KEYS))
        else:
            rm = window.mean(0); rs = window.std(0)
            rs[rs<1e-6] = 1e-6
            z = (spec_vals[i] - rm) / rs
        z_rows.append(z)
    return np.array(z_rows)


if __name__ == '__main__':

    print("Representation Diagnostic Analysis")
    print("Answering Sammy's four questions")
    print("="*60)

    # ── Load all data ─────────────────────────────────────────────────────────
    print("\nLoading data...")
    all_clips = {}
    for name,ff,ef,lf in RECORDERS:
        fp=os.path.join(BASE_DIR,ff); ep=os.path.join(BASE_DIR,ef)
        if not os.path.exists(fp) or not os.path.exists(ep): continue
        feat_df=pd.read_csv(fp); emb=np.load(ep)
        n=min(len(feat_df),len(emb))
        feat_df=feat_df.iloc[:n].reset_index(drop=True); emb=emb[:n]
        clip_df=build_clip_df(feat_df,emb)
        clip_df['recorder']=name
        all_clips[name]=clip_df
        print(f"  {name}: {len(clip_df)} clips ({int(clip_df['clip_label'].sum())} sim)")

    available = list(all_clips.keys())
    all_df = pd.concat(all_clips.values(), ignore_index=True)

    ec_cols = [c for c in all_df.columns if c.startswith('em_')]
    emb_all = all_df[ec_cols].values.astype(np.float32)
    spec_all = all_df[SPECTRAL_KEYS].values.astype(np.float32)
    y_site = all_df['recorder'].map({n:i for i,n in enumerate(available)}).values
    y_dist = all_df['clip_label'].values.astype(int)

    print(f"\nTotal clips: {len(all_df)}")
    print(f"Embedding dim: {emb_all.shape[1]}")

    # ── QUESTION 1: Site predictability from embeddings ────────────────────────
    print("\n" + "="*60)
    print("QUESTION 1: How much site info is in the embeddings?")
    print("="*60)
    print("Train classifier to predict recorder from embeddings.")
    print("100% = embeddings dominated by site identity")
    print("20%  = random (5 classes = 20% chance)")

    sc = StandardScaler()
    pca_50 = PCA(n_components=50, random_state=SEED)
    emb_pca = pca_50.fit_transform(sc.fit_transform(emb_all))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # Logistic regression — linear probe
    lr_site = LogisticRegression(max_iter=500, random_state=SEED, C=1.0)
    site_scores_lr = cross_val_score(lr_site, emb_pca, y_site, cv=cv, scoring='accuracy')

    # Random forest — non-linear probe
    rf_site = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    site_scores_rf = cross_val_score(rf_site, emb_pca, y_site, cv=cv, scoring='accuracy')

    print(f"\n  Linear probe (LR):        {site_scores_lr.mean():.3f} +/- {site_scores_lr.std():.3f}")
    print(f"  Non-linear probe (RF):    {site_scores_rf.mean():.3f} +/- {site_scores_rf.std():.3f}")
    print(f"  Random baseline:          0.200")

    # Also test disturbance predictability
    lr_dist = LogisticRegression(max_iter=500, random_state=SEED, C=1.0)
    dist_scores_lr = cross_val_score(lr_dist, emb_pca, y_dist, cv=cv, scoring='f1')
    rf_dist = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    dist_scores_rf = cross_val_score(rf_dist, emb_pca, y_dist, cv=cv, scoring='f1')

    print(f"\n  Disturbance F1 (LR):      {dist_scores_lr.mean():.3f} +/- {dist_scores_lr.std():.3f}")
    print(f"  Disturbance F1 (RF):      {dist_scores_rf.mean():.3f} +/- {dist_scores_rf.std():.3f}")

    site_dom = site_scores_rf.mean()
    dist_sep = dist_scores_rf.mean()
    ratio = site_dom / max(dist_sep, 0.01)
    print(f"\n  Site/Disturbance ratio:   {ratio:.2f}")
    print(f"  Interpretation: ", end="")
    if ratio > 3:
        print("SITE DOMINATES — embeddings are primarily site fingerprints")
    elif ratio > 1.5:
        print("SITE STRONGER — some disturbance signal but site dominates")
    else:
        print("BALANCED — site and disturbance both present")

    # ── QUESTION 2: PCA visualisation ─────────────────────────────────────────
    print("\n" + "="*60)
    print("QUESTION 2: Do embeddings cluster by site or disturbance?")
    print("="*60)

    pca_2 = PCA(n_components=2, random_state=SEED)
    emb_2d = pca_2.fit_transform(sc.transform(emb_all))
    var_exp = pca_2.explained_variance_ratio_

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Perch Embedding Space — PCA Projection', fontsize=14, fontweight='bold')

    # Plot 1: coloured by site
    ax = axes[0]
    ax.set_title(f'Coloured by Recorder Site\n(PC1={var_exp[0]:.1%} PC2={var_exp[1]:.1%})')
    for name in available:
        mask = all_df['recorder']==name
        ax.scatter(emb_2d[mask,0], emb_2d[mask,1],
                   c=COLORS_SITE[name], alpha=0.4, s=8, label=name)
    ax.legend(markerscale=3, fontsize=9)
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')

    # Plot 2: coloured by disturbance
    ax = axes[1]
    ax.set_title('Coloured by Disturbance Label')
    for label,lname in [(0,'Baseline'),(1,'Disturbance')]:
        mask = y_dist==label
        ax.scatter(emb_2d[mask,0], emb_2d[mask,1],
                   c=COLORS_DIST[label], alpha=0.4, s=8, label=lname)
    ax.legend(markerscale=3, fontsize=9)
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')

    plt.tight_layout()
    plot_path = os.path.join(BASE_DIR, 'embedding_pca.png')
    plt.savefig(plot_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  PCA plot saved -> {plot_path}")
    print(f"  Variance explained: PC1={var_exp[0]:.1%}  PC2={var_exp[1]:.1%}")

    # Quantify clustering: within-site vs between-site variance
    within_var  = np.mean([emb_pca[y_site==i].var(0).mean()
                           for i in range(len(available))])
    between_var = np.var([emb_pca[y_site==i].mean(0)
                          for i in range(len(available))], axis=0).mean()
    print(f"  Within-site variance:   {within_var:.4f}")
    print(f"  Between-site variance:  {between_var:.4f}")
    print(f"  Between/Within ratio:   {between_var/within_var:.3f}")
    print(f"  (>1 = sites well separated, <1 = sites overlap)")

    # ── QUESTION 3: Which features shift most across sites? ───────────────────
    print("\n" + "="*60)
    print("QUESTION 3: Which features shift most across sites?")
    print("="*60)
    print("Variance explained by site vs disturbance for each feature.")

    feature_names = SPECTRAL_KEYS
    site_var   = np.zeros(len(feature_names))
    dist_var   = np.zeros(len(feature_names))

    for fi, feat in enumerate(feature_names):
        vals = all_df[feat].values.astype(float)
        # Variance explained by site (between-site variance / total)
        site_means = [vals[y_site==i].mean() for i in range(len(available))]
        site_var[fi] = np.var(site_means) / (np.var(vals) + 1e-10)
        # Variance explained by disturbance
        dist_means = [vals[y_dist==i].mean() for i in range(2)]
        dist_var[fi] = np.var(dist_means) / (np.var(vals) + 1e-10)

    # Top site-specific features
    top_site_idx = np.argsort(site_var)[::-1][:10]
    top_dist_idx = np.argsort(dist_var)[::-1][:10]

    print("\n  Top 10 most SITE-SPECIFIC spectral features:")
    print(f"  {'Feature':<30} {'Site var%':>10} {'Dist var%':>10} {'Ratio':>8}")
    print("  "+"-"*62)
    for i in top_site_idx:
        ratio = site_var[i] / max(dist_var[i], 1e-6)
        print(f"  {feature_names[i]:<30} {site_var[i]:>10.3f} "
              f"{dist_var[i]:>10.3f} {ratio:>8.1f}x")

    print("\n  Top 10 most DISTURBANCE-DISCRIMINATIVE spectral features:")
    print(f"  {'Feature':<30} {'Dist var%':>10} {'Site var%':>10} {'Ratio':>8}")
    print("  "+"-"*62)
    for i in top_dist_idx:
        ratio = dist_var[i] / max(site_var[i], 1e-6)
        print(f"  {feature_names[i]:<30} {dist_var[i]:>10.3f} "
              f"{site_var[i]:>10.3f} {ratio:>8.1f}x")

    # ── QUESTION 4: Does adaptive normalisation reduce site separation? ────────
    print("\n" + "="*60)
    print("QUESTION 4: Does adaptive z-score reduce site separation?")
    print("="*60)

    # Compute pairwise between-site distances before normalisation
    def between_site_distances(features, site_labels, available):
        means = {n: features[site_labels==i].mean(0)
                 for i,n in enumerate(available)}
        dists = {}
        for i,n1 in enumerate(available):
            for j,n2 in enumerate(available):
                if j<=i: continue
                d = np.linalg.norm(means[n1]-means[n2])
                dists[(n1,n2)] = d
        return dists

    # Before: raw spectral features
    sc_spec = StandardScaler()
    spec_norm = sc_spec.fit_transform(spec_all)
    dists_before = between_site_distances(spec_norm, y_site, available)

    # After: adaptive z-scores
    print("  Computing adaptive z-scores for all clips...")
    z_all_rows = []
    for name in available:
        clip_df = all_clips[name].copy().reset_index(drop=True)
        z = compute_adaptive_zscores(clip_df, 40)
        z_all_rows.append(z)
    z_all = np.vstack(z_all_rows).astype(np.float32)
    # Rebuild site labels in same order
    y_site_ordered = np.concatenate([
        np.full(len(all_clips[n]), i)
        for i,n in enumerate(available)
    ])
    sc_z = StandardScaler()
    z_norm = sc_z.fit_transform(z_all)
    dists_after = between_site_distances(z_norm, y_site_ordered, available)

    print(f"\n  {'Pair':<12} {'Before':>8} {'After':>8} {'Change':>8} {'Reduced?':>10}")
    print("  "+"-"*52)
    avg_before = np.mean(list(dists_before.values()))
    avg_after  = np.mean(list(dists_after.values()))
    for pair in sorted(dists_before.keys()):
        db = dists_before[pair]; da = dists_after[pair]
        change = (da-db)/db*100
        reduced = "YES" if da < db else "no"
        print(f"  {pair[0]+'-'+pair[1]:<12} {db:>8.3f} {da:>8.3f} "
              f"{change:>+7.1f}% {reduced:>10}")
    print(f"\n  Average distance before: {avg_before:.3f}")
    print(f"  Average distance after:  {avg_after:.3f}")
    change_pct = (avg_after-avg_before)/avg_before*100
    print(f"  Change: {change_pct:+.1f}%")
    if avg_after < avg_before:
        print("  CONCLUSION: Adaptive z-score REDUCES site separation")
        print("  -> Supports interpretation as background subtraction")
    else:
        print("  CONCLUSION: Adaptive z-score does NOT reduce site separation")
        print("  -> Improvement comes from local sensitivity, not site removal")

    # Also check disturbance separability before/after
    dist_sep_before = np.linalg.norm(
        spec_norm[y_dist==1].mean(0) - spec_norm[y_dist==0].mean(0))
    dist_sep_after = np.linalg.norm(
        z_norm[y_dist[np.concatenate([np.arange(len(all_clips[n]))
               for n in available])]==1].mean(0) -
        z_norm[y_dist[np.concatenate([np.arange(len(all_clips[n]))
               for n in available])]==0].mean(0))
    print(f"\n  Disturbance separability before: {dist_sep_before:.3f}")
    print(f"  Disturbance separability after:  {dist_sep_after:.3f}")
    print(f"  -> Normalisation {'IMPROVES' if dist_sep_after>dist_sep_before else 'reduces'} "
          f"disturbance separability")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n\n" + "="*60)
    print("DIAGNOSTIC SUMMARY — ANSWERS TO SAMMY'S QUESTIONS")
    print("="*60)
    print(f"\n1. Site predictability from embeddings:")
    print(f"   Linear probe accuracy:    {site_scores_lr.mean():.3f}")
    print(f"   Non-linear probe accuracy:{site_scores_rf.mean():.3f}")
    print(f"   (random baseline = 0.200)")

    print(f"\n2. Embedding cluster structure:")
    print(f"   Between/within site variance ratio: {between_var/within_var:.3f}")
    print(f"   -> See embedding_pca.png for visualisation")

    print(f"\n3. Most site-specific features:")
    top3_site=[feature_names[i] for i in top_site_idx[:3]]
    print(f"   {top3_site}")
    print(f"   Most disturbance-discriminative:")
    top3_dist=[feature_names[i] for i in top_dist_idx[:3]]
    print(f"   {top3_dist}")

    print(f"\n4. Adaptive z-score effect on site separation:")
    print(f"   Site distance change: {change_pct:+.1f}%")
    print(f"   Disturbance sep change: "
          f"{'improved' if dist_sep_after>dist_sep_before else 'reduced'}")

    print("\n" + "="*60)
    print("IMPLICATION FOR NEXT STEPS:")
    if site_scores_rf.mean() > 0.80:
        print("  Site accuracy > 0.80: embeddings dominated by site identity.")
        print("  -> Alternative embeddings (CLAP, BEATs, AudioMAE) are")
        print("     the highest-priority next experiment.")
        print("  -> Trying more classifiers will not help.")
    elif site_scores_rf.mean() > 0.50:
        print("  Site accuracy 0.50-0.80: moderate site dominance.")
        print("  -> Some universal signal exists but is overshadowed.")
        print("  -> Feature selection to remove site-specific features")
        print("     may help before trying alternative embeddings.")
    else:
        print("  Site accuracy < 0.50: embeddings are relatively site-invariant.")
        print("  -> Problem is in the classifier or training setup,")
        print("     not the representation itself.")
    print("\nDone.")
