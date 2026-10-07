"""
61_ar_sequence_model.py
------------------------
Linear Autoregressive (AR) Sequence Model for Disturbance Detection

THE NLP ANALOGY:
  GPT predicts the next word from all previous words.
  This model predicts the next clip's acoustic features
  from the last N clips.

  A clip that cannot be predicted from its neighbours
  is acoustically anomalous — potentially a disturbance.

WHY THIS IS DIFFERENT FROM WHAT WE TRIED:
  Script 50 (anomaly detection) asked: "is this clip unusual
  compared to baseline?" — one clip at a time.

  This model asks: "given the acoustic narrative of the last
  5 clips, what should clip 6 sound like? How wrong was that
  prediction?"

  Prediction error captures CHANGE in the acoustic scene,
  not just absolute unusualness. A vehicle approaching
  creates a trajectory of increasing prediction error
  even if individual clips look normal in isolation.

THE MODEL:
  For each clip at position t, use clips t-5 through t-1
  as context (the "acoustic sentence so far").
  Train a Ridge regression to predict clip t's features
  from that context window.
  At inference: prediction error = how surprised the model
  was by this clip given recent history.
  High sustained prediction error = potential disturbance.

DEPLOYMENT ADVANTAGE:
  No disturbance labels needed.
  Train on unlabelled baseline clips only.
  Adapts automatically to each site's acoustic context.
  This is the formal version of what adaptive z-scores do.

EVALUATION:
  Per-site: train AR model on baseline clips, score all clips,
  evaluate whether high prediction error flags disturbance.
  Cross-site LODO: train on 4 sites, test on 1.
  Compare to adaptive z-score baseline.
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (precision_recall_fscore_support,
                              precision_recall_curve, roc_auc_score)

BASE_DIR = r"C:\Users\BitaMassoudi\KasmirWorldProjects\toucanet-option-b"
SEED     = 42
np.random.seed(SEED)

RECORDERS = [
    ('AM4','am4_full_feature_matrix.csv','am4_full_emb.npy'),
    ('AM2','am2_feature_matrix.csv','am2_time_controlled_emb.npy'),
    ('AM5','am5_feature_matrix.csv','am5_time_controlled_emb.npy'),
    ('AM6','am6_feature_matrix.csv','am6_time_controlled_emb.npy'),
    ('AM1','am1_feature_matrix.csv','am1_full_emb.npy'),
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

# AR model hyperparameters
CONTEXT_WINDOW = 5    # how many past clips to use as context
PCA_COMPONENTS = 20   # reduce embedding dim before AR
ALPHA          = 1.0  # Ridge regularisation
# Anomaly scoring
ROLLING_N      = 40   # rolling window for error normalisation
THRESHOLD_PCT  = 0.90 # flag clips above this percentile of recent errors


def parse_timestamp(clip_name):
    parts = clip_name.split('_')
    try:
        date = parts[3]; time = parts[4].replace('.wav','')
        return pd.Timestamp(
            f"{date[:4]}-{date[4:6]}-{date[6:8]} "
            f"{time[:2]}:{time[2:4]}:{time[4:6]}")
    except:
        return pd.Timestamp('2025-01-01')


def build_clip_df(feat_df, emb):
    """Build clip-level dataframe sorted chronologically."""
    rows = []
    for cn in feat_df['clip_name'].unique():
        mask = feat_df['clip_name'].values == cn
        cl   = feat_df['clip_label'].values[mask][0]
        hour = feat_df['clip_hour'].values[mask][0]
        ts   = parse_timestamp(cn)
        ce   = emb[mask]
        row  = {'clip_name':cn, 'clip_label':cl,
                'clip_hour':hour, 'timestamp':ts}
        for k in SPECTRAL_KEYS:
            if k in feat_df.columns:
                row[k] = feat_df.loc[mask, k].values[0]
        for i,v in enumerate(ce.mean(0)): row[f'em_{i}'] = v
        rows.append(row)
    df = pd.DataFrame(rows).sort_values('timestamp').reset_index(drop=True)
    return df


def build_ar_features(embeddings_pca, context_window):
    """
    Build autoregressive feature matrix.
    For each clip t, context = clips t-W through t-1 (flattened).
    Target = clip t's embedding.
    Returns X (context), y (target), valid indices.
    """
    n, d = embeddings_pca.shape
    X, y, idx = [], [], []
    for t in range(context_window, n):
        context = embeddings_pca[t-context_window:t].flatten()
        X.append(context)
        y.append(embeddings_pca[t])
        idx.append(t)
    return np.array(X), np.array(y), np.array(idx)


def rolling_normalise_errors(errors, n=40):
    """
    Convert raw prediction errors to rolling percentile scores.
    Each error is compared to the last N errors.
    This is the NLP analogy of adaptive z-scores:
    'how surprising is this clip relative to recent surprises?'
    """
    normed = np.zeros_like(errors)
    for i in range(len(errors)):
        ws = max(0, i-n)
        window = errors[ws:i]
        if len(window) < 2:
            normed[i] = 0.5
        else:
            normed[i] = np.mean(window < errors[i])
    return normed


def evaluate(y, scores):
    """Find best threshold and report P, R, AUC."""
    if len(np.unique(y)) < 2:
        return 0, 0, 0, False
    auc = roc_auc_score(y, scores)
    pr, rc, th = precision_recall_curve(y, scores)
    valid = np.where((pr[:-1] >= 0.70) & (rc[:-1] >= 0.70))[0]
    if len(valid) > 0:
        bi = valid[np.argmax(pr[valid] + rc[valid])]
        bt = float(th[bi])
    else:
        bi = np.argmax(pr[:-1] + rc[:-1])
        bt = float(th[bi])
    preds = (scores >= bt).astype(int)
    p, r, _, _ = precision_recall_fscore_support(
        y, preds, average='binary', zero_division=0)
    return p, r, auc, (p >= 0.70 and r >= 0.70)


if __name__ == '__main__':

    print("AR Sequence Model — NLP-Inspired Disturbance Detection")
    print("="*65)
    print(f"Context window: {CONTEXT_WINDOW} clips")
    print(f"PCA components: {PCA_COMPONENTS}")
    print(f"Rolling normalisation: last {ROLLING_N} clips")
    print("="*65)

    # Load all data
    all_clips = {}
    for name, ff, ef in RECORDERS:
        fp = os.path.join(BASE_DIR, ff)
        ep = os.path.join(BASE_DIR, ef)
        if not os.path.exists(fp) or not os.path.exists(ep):
            continue
        feat_df = pd.read_csv(fp)
        emb     = np.load(ep)
        n = min(len(feat_df), len(emb))
        feat_df = feat_df.iloc[:n].reset_index(drop=True)
        emb     = emb[:n]
        clip_df = build_clip_df(feat_df, emb)
        all_clips[name] = clip_df
        n_base = int((clip_df['clip_label'] == 0).sum())
        n_sim  = int((clip_df['clip_label'] == 1).sum())
        print(f"  {name}: {len(clip_df)} clips "
              f"({n_sim} sim, {n_base} baseline) — chronological order")

    available = list(all_clips.keys())

    # ── PER-SITE AR MODEL ──────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("PER-SITE AR MODEL")
    print("Train on baseline clips only. Score all clips.")
    print("="*65)

    per_site_results = {}

    for name in available:
        clip_df = all_clips[name].copy().reset_index(drop=True)
        y       = clip_df['clip_label'].values.astype(int)
        n_clips = len(clip_df)

        print(f"\n{name} ({n_clips} clips chronological):")

        # Get embeddings in chronological order
        ec_cols = [c for c in clip_df.columns if c.startswith('em_')]
        emb_all = clip_df[ec_cols].values.astype(np.float32)

        # PCA to reduce embedding dimension
        sc  = StandardScaler()
        emb_s = sc.fit_transform(emb_all)
        pca = PCA(n_components=PCA_COMPONENTS, random_state=SEED)
        emb_pca = pca.fit_transform(emb_s)

        # Build AR training data from BASELINE clips only
        # (deployment-valid: we only use unlabelled baseline for training)
        baseline_mask = y == 0
        baseline_idx  = np.where(baseline_mask)[0]

        # Train AR on baseline sequences
        # Find contiguous baseline runs for context window
        ar_X, ar_y_target = [], []
        for i in range(CONTEXT_WINDOW, len(baseline_idx)):
            # Check if indices are contiguous (consecutive clips)
            idxs = baseline_idx[i-CONTEXT_WINDOW:i+1]
            if np.all(np.diff(idxs) == 1):  # contiguous
                context = emb_pca[idxs[:-1]].flatten()
                target  = emb_pca[idxs[-1]]
                ar_X.append(context)
                ar_y_target.append(target)

        if len(ar_X) < 20:
            # Fall back: use all baseline clips regardless of continuity
            ar_X2, ar_y2, _ = build_ar_features(
                emb_pca[baseline_idx], CONTEXT_WINDOW)
            ar_X = ar_X2.tolist()
            ar_y_target = ar_y2.tolist()

        ar_X = np.array(ar_X)
        ar_y_target = np.array(ar_y_target)
        print(f"  AR training samples: {len(ar_X)} baseline sequences")

        # Train Ridge regression AR model
        ar_model = Ridge(alpha=ALPHA)
        ar_model.fit(ar_X, ar_y_target)

        # Score all clips in chronological order
        # Prediction error = how surprised the AR model is
        raw_errors = np.zeros(n_clips)
        for t in range(CONTEXT_WINDOW, n_clips):
            context = emb_pca[t-CONTEXT_WINDOW:t].flatten()
            pred    = ar_model.predict(context.reshape(1,-1))[0]
            raw_errors[t] = np.mean((emb_pca[t] - pred)**2)

        # Normalise errors using rolling window (adaptive, like z-scores)
        norm_errors = rolling_normalise_errors(raw_errors, ROLLING_N)

        # Evaluate
        valid_mask = np.arange(n_clips) >= CONTEXT_WINDOW
        p, r, auc, beat = evaluate(
            y[valid_mask], norm_errors[valid_mask])

        per_site_results[name] = (p, r, auc, beat)
        print(f"  AR model:  P={p:.3f}  R={r:.3f}  AUC={auc:.3f}  "
              f"{'BEAT ★' if beat else 'miss'}")

        # Compare to adaptive z-score (previous best)
        prev = {'AM4':0.928,'AM2':0.998,'AM5':0.847,'AM6':0.943,'AM1':1.000}
        print(f"  Adaptive z-score: P={prev.get(name,0):.3f}  "
              f"(AR change: {p-prev.get(name,0):+.3f})")

    # ── CROSS-SITE LODO AR ─────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("CROSS-SITE LODO — AR MODEL")
    print("Train AR on 4 sites, test prediction error on 1 held-out site.")
    print("="*65)

    lodo_results = {}

    for test_name in available:
        train_names = [r for r in available if r != test_name]
        test_df     = all_clips[test_name].copy().reset_index(drop=True)
        y_te        = test_df['clip_label'].values.astype(int)
        if len(np.unique(y_te)) < 2:
            continue

        # Build combined training embeddings from all training sites
        train_dfs = [all_clips[r] for r in train_names]
        ec_cols   = [c for c in test_df.columns if c.startswith('em_')]

        # Fit PCA and scaler on training sites
        train_emb = np.vstack([df[ec_cols].values for df in train_dfs])
        sc_cross  = StandardScaler()
        train_emb_s = sc_cross.fit_transform(train_emb)
        pca_cross   = PCA(n_components=PCA_COMPONENTS, random_state=SEED)
        train_emb_pca = pca_cross.fit_transform(train_emb_s)

        # Build AR training data from baseline clips across all training sites
        ar_X_cross, ar_y_cross = [], []
        offset = 0
        for df in train_dfs:
            n_tr = len(df)
            emb_site = train_emb_pca[offset:offset+n_tr]
            y_site   = df['clip_label'].values
            base_idx = np.where(y_site == 0)[0]
            X_s, y_s, _ = build_ar_features(
                emb_site[base_idx], CONTEXT_WINDOW)
            ar_X_cross.extend(X_s.tolist())
            ar_y_cross.extend(y_s.tolist())
            offset += n_tr

        ar_X_cross = np.array(ar_X_cross)
        ar_y_cross = np.array(ar_y_cross)

        # Train AR
        ar_cross = Ridge(alpha=ALPHA)
        ar_cross.fit(ar_X_cross, ar_y_cross)

        # Score test site
        test_emb   = test_df[ec_cols].values.astype(np.float32)
        test_emb_s = sc_cross.transform(test_emb)
        test_pca   = pca_cross.transform(test_emb_s)
        n_te       = len(test_df)

        raw_errors_te = np.zeros(n_te)
        for t in range(CONTEXT_WINDOW, n_te):
            context = test_pca[t-CONTEXT_WINDOW:t].flatten()
            pred    = ar_cross.predict(context.reshape(1,-1))[0]
            raw_errors_te[t] = np.mean((test_pca[t] - pred)**2)

        norm_errors_te = rolling_normalise_errors(raw_errors_te, ROLLING_N)
        valid_mask_te  = np.arange(n_te) >= CONTEXT_WINDOW

        p, r, auc, beat = evaluate(
            y_te[valid_mask_te], norm_errors_te[valid_mask_te])
        lodo_results[test_name] = (p, r, auc, beat)

        prev_cs = 0.364  # best previous cross-site
        print(f"\n  Test={test_name}: P={p:.3f}  R={r:.3f}  "
              f"AUC={auc:.3f}  {'BEAT ★' if beat else 'miss'}")

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    print("\n\n" + "="*65)
    print("AR SEQUENCE MODEL — FINAL SUMMARY")
    print("="*65)

    print("\nPer-site (AR model vs adaptive z-score best):")
    prev_ps = {'AM4':0.928,'AM2':0.998,'AM5':0.847,'AM6':0.943,'AM1':1.000}
    print(f"{'Rec':<6} {'AR P':>7} {'AR R':>7} {'AUC':>7} "
          f"{'Z-score P':>10} {'Change':>8} {'Beat?':>6}")
    print("-"*55)
    for name in available:
        if name not in per_site_results: continue
        p,r,auc,beat = per_site_results[name]
        prev = prev_ps.get(name,0)
        print(f"{name:<6} {p:>7.3f} {r:>7.3f} {auc:>7.3f} "
              f"{prev:>10.3f} {p-prev:>+8.3f} "
              f"{'BEAT' if beat else 'miss':>6}")

    print("\nCross-site LODO (AR model vs best previous = 0.364):")
    print(f"{'Rec':<6} {'AR P':>7} {'AR R':>7} {'AUC':>7} {'Beat?':>6}")
    print("-"*38)
    cs_ps = []
    for name in available:
        if name not in lodo_results: continue
        p,r,auc,beat = lodo_results[name]
        cs_ps.append(p)
        print(f"{name:<6} {p:>7.3f} {r:>7.3f} {auc:>7.3f} "
              f"{'BEAT ★' if beat else 'miss':>6}")
    if cs_ps:
        print(f"\n  AR avg cross-site P: {np.mean(cs_ps):.3f}")
        print(f"  Previous best:       0.364")
        print(f"  Change:              {np.mean(cs_ps)-0.364:+.3f}")

    print("\n" + "="*65)
    print("INTERPRETATION")
    print("="*65)
    print("""
The AR model predicts each clip's acoustic embedding from the
last 5 clips. High prediction error = acoustic change = anomaly.

This is the NLP-inspired formalisation of what adaptive z-scores
do at the feature level — but operating on the full embedding
sequence rather than individual feature statistics.

Per-site: if AR beats z-scores, learned temporal context
  captures something that rolling mean/std misses.

Cross-site: if AR beats 0.364, temporal continuity in the
  embedding space is more transferable than absolute features.
  The model learns 'what acoustic change looks like'
  rather than 'what disturbance sounds like' — potentially
  more universal across sites.
""")
    print("Done.")
