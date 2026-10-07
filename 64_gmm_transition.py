"""
64_gmm_transition.py
---------------------
GMM Acoustic State Modeling + Transition Matrix Analysis
Based on Griffin's Spring 2026 work.

THE IDEA:
  Instead of classifying clips by what they sound like (site-specific),
  model the DYNAMICS of how the soundscape moves through acoustic space.

  Step 1: Learn K discrete acoustic states from baseline clips using GMM.
          Each state = a cluster of similar-sounding acoustic moments.
          States learned globally across all sites.

  Step 2: Assign each clip to its nearest state.
          Now the recording is a sequence of state labels:
          [3, 3, 3, 3, 7, 3, 3, 3, 12, 3, 3, ...]
                                ^           ^
                           unusual state  unusual state

  Step 3: Compute transition matrices.
          Baseline: forest stays in same state (diagonal matrix).
          Disturbance: sudden jumps to unusual states (off-diagonal).

  Step 4: Measure:
          - Transition entropy: how unpredictable are state transitions?
          - Velocity: how far does embedding move between consecutive clips?
          - State persistence: how long does the forest stay in each state?
          - Surprise score: P(current state | previous state) from baseline model

  Step 5: Use these dynamic features as classifier inputs for LODO.

WHY THIS COULD WORK CROSS-SITE:
  The absolute states differ across sites (AM1's state 3 sounds different
  from AM4's state 3). But the DYNAMICS may be universal:
  - Baseline everywhere: high persistence, low entropy, smooth transitions
  - Disturbance everywhere: sudden jumps, high entropy, low persistence

  This encodes ecological behaviour rather than acoustic identity.

GRIFFIN'S KEY FINDINGS (Spring 2026):
  - Baseline transition matrices are strongly diagonal
  - Disturbance reduces temporal persistence
  - Disruption clips show higher embedding dispersion
  - Increased temporal velocity during human presence
"""

import os
import numpy as np
import pandas as pd
import zipfile
import io
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (precision_recall_fscore_support,
                              precision_recall_curve, roc_auc_score)

BASE_DIR = r"C:\Users\BitaMassoudi\KasmirWorldProjects\toucanet-option-b"
ZIP_PATH = r"C:\Users\BitaMassoudi\Downloads\raw_segments_per_clip.zip"
SEED     = 42
np.random.seed(SEED)

N_STATES = 16   # GMM components — number of acoustic states
PCA_DIM  = 32   # reduce embedding dim before GMM

RECORDERS = [
    ('AM4','am4_full_feature_matrix.csv'),
    ('AM2','am2_feature_matrix.csv'),
    ('AM5','am5_feature_matrix.csv'),
    ('AM6','am6_feature_matrix.csv'),
    ('AM1','am1_feature_matrix.csv'),
]


def parse_timestamp(clip_name):
    parts = clip_name.split('_')
    try:
        date = parts[3]; time = parts[4].replace('.wav','')
        return pd.Timestamp(
            f"{date[:4]}-{date[4:6]}-{date[6:8]} "
            f"{time[:2]}:{time[2:4]}:{time[4:6]}")
    except:
        return pd.Timestamp('2025-01-01')


def load_birdnet_embeddings(clip_names):
    """Load BirdNET embeddings from zip for given clip names."""
    stems = {cn.replace('.wav',''): cn for cn in clip_names}
    emb_map = {}
    with zipfile.ZipFile(ZIP_PATH) as z:
        entries = [n for n in z.namelist()
                   if n.endswith('.npz') and 'Audio_Moth' in n
                   and '__MACOSX' not in n]
        for n in entries:
            stem = n.replace('raw_segments_per_clip/','').replace('.npz','')
            if stem not in stems: continue
            try:
                with z.open(n) as f:
                    data = np.load(io.BytesIO(f.read()))
                    emb = np.nan_to_num(
                        data['segments'].astype(np.float64).mean(axis=0),
                        nan=0.0, posinf=0.0, neginf=0.0)
                    emb_map[stems[stem]] = emb
            except: pass
    return emb_map


def build_recorder_df(feat_df, emb_map):
    """Build chronologically sorted clip dataframe with embeddings."""
    rows = []
    for cn in feat_df['clip_name'].unique():
        if cn not in emb_map: continue
        mask = feat_df['clip_name'].values == cn
        rows.append({
            'clip_name': cn,
            'clip_label': int(feat_df['clip_label'].values[mask][0]),
            'clip_hour': feat_df['clip_hour'].values[mask][0],
            'timestamp': parse_timestamp(cn),
            'emb': emb_map[cn]
        })
    df = pd.DataFrame(rows).sort_values('timestamp').reset_index(drop=True)
    return df


def compute_dynamic_features(emb_seq, state_seq, baseline_trans, n_states,
                              rolling_n=40):
    """
    Compute dynamic/transition features for each clip.

    Features:
    1. velocity: L2 distance from previous clip's embedding
    2. transition_surprise: -log P(current state | prev state) under baseline
    3. state_rarity: -log P(current state) under baseline marginal
    4. rolling_velocity_zscore: velocity normalised to recent baseline
    5. transition_entropy_local: entropy of recent state transitions
    """
    n = len(emb_seq)
    feats = np.zeros((n, 5), dtype=np.float32)

    # Baseline marginal state distribution
    baseline_marginal = baseline_trans.sum(axis=1)
    baseline_marginal = np.maximum(baseline_marginal, 1e-8)
    baseline_marginal /= baseline_marginal.sum()

    raw_velocity = np.zeros(n)
    for t in range(1, n):
        raw_velocity[t] = np.linalg.norm(emb_seq[t] - emb_seq[t-1])

    for t in range(n):
        # 1. Velocity
        feats[t, 0] = raw_velocity[t]

        # 2. Transition surprise
        if t > 0:
            prev_s = state_seq[t-1]; curr_s = state_seq[t]
            p_trans = baseline_trans[prev_s, curr_s]
            feats[t, 1] = -np.log(max(p_trans, 1e-8))
        else:
            feats[t, 1] = 0.0

        # 3. State rarity
        feats[t, 2] = -np.log(baseline_marginal[state_seq[t]])

        # 4. Rolling velocity z-score
        ws = max(0, t - rolling_n)
        window_vel = raw_velocity[ws:t]
        if len(window_vel) > 2:
            mu = window_vel.mean(); sig = max(window_vel.std(), 1e-6)
            feats[t, 3] = (raw_velocity[t] - mu) / sig
        else:
            feats[t, 3] = 0.0

        # 5. Local transition entropy
        ws = max(0, t - 10)
        recent_states = state_seq[ws:t+1]
        if len(recent_states) > 1:
            counts = np.bincount(recent_states, minlength=n_states).astype(float)
            counts = counts / counts.sum()
            entropy = -np.sum(counts * np.log(np.maximum(counts, 1e-8)))
            feats[t, 4] = entropy
        else:
            feats[t, 4] = 0.0

    return feats


def evaluate(y, scores):
    if len(np.unique(y)) < 2: return 0, 0, False
    pr,rc,th = precision_recall_curve(y, scores)
    valid = np.where((pr[:-1]>=0.70) & (rc[:-1]>=0.70))[0]
    if len(valid) > 0:
        bi = valid[np.argmax(pr[valid]+rc[valid])]; bt = float(th[bi])
    else:
        bi = np.argmax(pr[:-1]+rc[:-1]); bt = float(th[bi])
    preds = (scores >= bt).astype(int)
    p,r,_,_ = precision_recall_fscore_support(
        y, preds, average='binary', zero_division=0)
    return p, r, (p>=0.70 and r>=0.70)


if __name__ == '__main__':

    print("GMM Acoustic State Modeling + Transition Analysis")
    print("Based on Griffin's Spring 2026 work")
    print("="*65)
    print(f"N_STATES={N_STATES}  PCA_DIM={PCA_DIM}")

    # ── Load all data ─────────────────────────────────────────────────────────
    print("\nLoading labels and BirdNET embeddings...")
    all_clip_names = []
    recorder_meta  = {}
    for name, ff in RECORDERS:
        fp = os.path.join(BASE_DIR, ff)
        if not os.path.exists(fp): continue
        df = pd.read_csv(fp)
        clips = df['clip_name'].unique().tolist()
        all_clip_names.extend(clips)
        recorder_meta[name] = df
        print(f"  {name}: {len(clips)} clips")

    print("  Loading BirdNET embeddings from zip...")
    emb_map = load_birdnet_embeddings(all_clip_names)
    print(f"  Loaded {len(emb_map)} embeddings")

    # Build per-recorder dataframes
    recorder_dfs = {}
    for name, df in recorder_meta.items():
        rdf = build_recorder_df(df, emb_map)
        if len(rdf) > 0:
            recorder_dfs[name] = rdf
            n_sim = int(rdf['clip_label'].sum())
            print(f"  {name}: {len(rdf)} clips with embeddings ({n_sim} sim)")

    available = list(recorder_dfs.keys())

    # ── Fit global GMM on all baseline clips ──────────────────────────────────
    print(f"\nFitting GMM with {N_STATES} acoustic states...")
    print("Using all baseline clips from all recorders (globally shared states)")

    all_base_embs = []
    for name in available:
        rdf = recorder_dfs[name]
        base_embs = np.array([row['emb'] for _, row in rdf.iterrows()
                              if row['clip_label'] == 0])
        all_base_embs.append(base_embs)

    all_base_matrix = np.vstack(all_base_embs)
    print(f"  {len(all_base_matrix)} baseline clips for GMM training")

    # PCA then GMM
    sc_gmm  = StandardScaler()
    pca_gmm = PCA(n_components=PCA_DIM, random_state=SEED)
    base_s  = pca_gmm.fit_transform(sc_gmm.fit_transform(all_base_matrix))

    gmm = GaussianMixture(n_components=N_STATES, random_state=SEED,
                          covariance_type='diag', max_iter=200)
    gmm.fit(base_s)
    print(f"  GMM fitted. Log-likelihood: {gmm.lower_bound_:.2f}")

    # ── Assign states and compute transition features per recorder ────────────
    print("\nAssigning acoustic states and computing dynamic features...")
    recorder_features = {}

    for name in available:
        rdf = recorder_dfs[name]
        emb_seq = np.array([row['emb'] for _, row in rdf.iterrows()])
        y       = rdf['clip_label'].values.astype(int)

        # Project to GMM space
        emb_pca = pca_gmm.transform(sc_gmm.transform(emb_seq))
        states  = gmm.predict(emb_pca)

        # Build baseline transition matrix from baseline clips only
        # Using chronological baseline clips from THIS recorder
        base_idx = np.where(y == 0)[0]
        trans_matrix = np.ones((N_STATES, N_STATES)) * 1e-3  # Laplace smoothing
        for i in range(len(base_idx)-1):
            if base_idx[i+1] == base_idx[i]+1:  # consecutive
                trans_matrix[states[base_idx[i]], states[base_idx[i+1]]] += 1
        # Normalise rows
        row_sums = trans_matrix.sum(axis=1, keepdims=True)
        trans_matrix = trans_matrix / row_sums

        # Compute diagonality — how diagonal is the baseline transition matrix?
        diagonality = np.trace(trans_matrix) / N_STATES
        print(f"  {name}: baseline transition diagonality = {diagonality:.3f} "
              f"(1.0 = perfectly stable, 0 = random)")

        # Compute dynamic features
        dyn_feats = compute_dynamic_features(
            emb_pca, states, trans_matrix, N_STATES)

        recorder_features[name] = {
            'feats': dyn_feats,
            'y': y,
            'states': states,
            'trans': trans_matrix,
            'diagonality': diagonality
        }

    # ── LODO cross-site using dynamic features ────────────────────────────────
    print(f"\n{'='*65}")
    print("LODO CROSS-SITE — DYNAMIC FEATURES")
    print("Velocity + Transition Surprise + State Rarity + Entropy")
    print("="*65)

    lodo_results = {}

    for test_name in available:
        train_names = [r for r in available if r != test_name]

        X_tr = np.vstack([recorder_features[r]['feats'] for r in train_names])
        y_tr = np.concatenate([recorder_features[r]['y'] for r in train_names])
        X_te = recorder_features[test_name]['feats']
        y_te = recorder_features[test_name]['y']

        X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
        X_te = np.nan_to_num(X_te, nan=0.0, posinf=0.0, neginf=0.0)

        if len(np.unique(y_te)) < 2: continue

        sc   = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)

        n_pos = y_tr.sum(); n_neg = (y_tr==0).sum()
        w = np.where(y_tr==1, n_neg/max(n_pos,1), 1.0)

        gb = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            random_state=SEED, subsample=0.8)
        gb.fit(X_tr_s, y_tr, sample_weight=w)
        probs = gb.predict_proba(X_te_s)[:,1]

        p, r, beat = evaluate(y_te, probs)
        lodo_results[test_name] = (p, r, beat)
        print(f"  {test_name}: P={p:.3f} R={r:.3f} {'BEAT' if beat else 'miss'}")

    # Also test velocity alone as a simple cross-site anomaly score
    print(f"\n  Velocity alone (no classifier):")
    for test_name in available:
        vel = recorder_features[test_name]['feats'][:,3]  # rolling z-score velocity
        y_te = recorder_features[test_name]['y']
        if len(np.unique(y_te)) < 2: continue
        p, r, beat = evaluate(y_te, vel)
        print(f"  {test_name}: P={p:.3f} R={r:.3f} {'BEAT' if beat else 'miss'}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n\n{'='*65}")
    print("GMM + TRANSITION — FINAL SUMMARY")
    print("="*65)
    avg_p = np.mean([v[0] for v in lodo_results.values()])
    n_beat = sum(v[2] for v in lodo_results.values())

    print(f"\nBaseline transition diagonality per recorder:")
    for name in available:
        d = recorder_features[name]['diagonality']
        bar = '█' * int(d * 20)
        print(f"  {name}: {d:.3f}  {bar}")

    print(f"\nCross-site precision:")
    print(f"  GMM+transitions avg P: {avg_p:.3f}")
    print(f"  Previous best:         0.364")
    print(f"  Change:                {avg_p-0.364:+.3f}")
    print(f"  Beat 0.70:             {n_beat}/{len(lodo_results)}")

    if avg_p > 0.364:
        print(f"\n  IMPROVEMENT — dynamic features transfer better than embeddings")
        print(f"  Griffin's approach works: temporal dynamics are more site-invariant")
    else:
        print(f"\n  No improvement over embedding-based approaches")
        print(f"  But check diagonality — if baseline is strongly diagonal,")
        print(f"  the signal exists but the classifier may need tuning")

    print("\nDone.")
