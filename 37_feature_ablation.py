"""
37_feature_ablation.py
-----------------------
Feature ablation study — tests which feature groups generalise
across recorder sites.

BACKGROUND:
  Cross-site training completely failed (script 36):
  0/5 recorders beat 0.70 when trained on other sites.
  Precision collapsed from 0.70-0.85 down to 0.19-0.50.

  This means the model is learning site-specific patterns
  rather than universal disturbance signals.

  We need to find out WHICH features are causing this.

THREE FEATURE SETS TESTED:

  Set A — Z-scores only (41 features)
    These are theoretically the most site-agnostic.
    They measure deviation from what is normal at this
    specific hour at this specific site.
    If anything generalises across sites it should be these.

  Set B — Raw spectral only (41 features)
    These are the absolute acoustic measurements.
    They are most likely to carry site-specific information
    because what is loud at AM4 at night is different from
    what is loud at AM5 in the morning.

  Set C — Perch PCA only (50 features)
    The deep learning representation from Google's model.
    Perch was trained on millions of bird sounds globally.
    Its internal representation might generalise better
    than hand-crafted spectral features.

  Set D — Z-scores + Perch PCA (91 features)
    Combines the two most likely to generalise.
    No raw spectral features.

  Set E — All features (106 features) — baseline comparison

EVALUATION:
  Both per-site CV (honest, as in script 34) and
  cross-site leave-one-out (as in script 36) for each set.
  This directly answers: which features generalise?

Uses saved feature matrices — no re-embedding needed.
Should finish in about 30-40 minutes.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (precision_recall_fscore_support,
                              precision_recall_curve)

BASE_DIR = r"C:\Users\BitaMassoudi\KasmirWorldProjects\toucanet-option-b"

RECORDERS = [
    {'name':'AM4','feat':'am4_full_feature_matrix.csv',
     'emb':'am4_full_emb.npy','lab':'am4_full_labels.npy'},
    {'name':'AM2','feat':'am2_feature_matrix.csv',
     'emb':'am2_time_controlled_emb.npy',
     'lab':'am2_time_controlled_labels.npy'},
    {'name':'AM5','feat':'am5_feature_matrix.csv',
     'emb':'am5_time_controlled_emb.npy',
     'lab':'am5_time_controlled_labels.npy'},
    {'name':'AM6','feat':'am6_feature_matrix.csv',
     'emb':'am6_time_controlled_emb.npy',
     'lab':'am6_time_controlled_labels.npy'},
    {'name':'AM1','feat':'am1_feature_matrix.csv',
     'emb':'am1_full_emb.npy','lab':'am1_full_labels.npy'},
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
Z_KEYS = [f"z_{k}" for k in SPECTRAL_KEYS]

FEATURE_SETS = {
    'A_zscore_only':    {'use_z':True,  'use_raw':False, 'use_emb':False},
    'B_raw_only':       {'use_z':False, 'use_raw':True,  'use_emb':False},
    'C_perch_only':     {'use_z':False, 'use_raw':False, 'use_emb':True},
    'D_zscore_perch':   {'use_z':True,  'use_raw':False, 'use_emb':True},
    'E_all_features':   {'use_z':True,  'use_raw':True,  'use_emb':True},
}

RANDOM_SEED = 42
N_FOLDS     = 5


def compute_rain_score(clip_df):
    rms  = clip_df['rms_mean'].values
    ent  = clip_df['spec_entropy_mean'].values
    roll = clip_df['rolloff_std'].values
    sil  = clip_df['silence_fraction'].values
    rms_n  = (rms-rms.min())/(rms.max()-rms.min()+1e-10)
    ent_n  = (ent-ent.min())/(ent.max()-ent.min()+1e-10)
    roll_n = 1.0-(roll-roll.min())/(roll.max()-roll.min()+1e-10)
    sil_n  = 1.0-(sil-sil.min())/(sil.max()-sil.min()+1e-10)
    return 0.35*rms_n+0.35*ent_n+0.20*roll_n+0.10*sil_n


def build_clip_features(feat_df, embeddings, labels):
    clip_rows = []
    for clip_name in feat_df['clip_name'].unique():
        mask       = feat_df['clip_name'].values == clip_name
        clip_label = feat_df['clip_label'].values[mask][0]
        clip_hour  = feat_df['clip_hour'].values[mask][0]
        clip_embs  = embeddings[mask]
        emb_mean   = clip_embs.mean(axis=0)
        emb_std    = clip_embs.std(axis=0)
        emb_max    = clip_embs.max(axis=0)
        spectral   = feat_df.loc[mask, SPECTRAL_KEYS].values[0]
        row = {'clip_name':clip_name,'clip_label':clip_label,
               'clip_hour':clip_hour}
        for k,v in zip(SPECTRAL_KEYS, spectral): row[k] = v
        for i,v in enumerate(emb_mean): row[f'emb_mean_{i}'] = v
        for i,v in enumerate(emb_std):  row[f'emb_std_{i}']  = v
        for i,v in enumerate(emb_max):  row[f'emb_max_{i}']  = v
        clip_rows.append(row)
    return pd.DataFrame(clip_rows)


def apply_rain_filter(clip_df):
    scores   = compute_rain_score(clip_df)
    clip_df  = clip_df.copy()
    clip_df['rain_score'] = scores
    base_scores = clip_df.loc[clip_df['clip_label']==0,'rain_score']
    threshold   = np.percentile(base_scores, 90)
    clip_df     = clip_df[clip_df['rain_score']<=threshold].copy()
    return clip_df.drop(columns=['rain_score'])


def compute_zscores_train_only(train_df, test_df=None):
    """Compute z-scores from training baseline, apply to train and optionally test."""
    z_tr_rows, z_te_rows = [], []
    for hour in train_df['clip_hour'].unique():
        base_mask = ((train_df['clip_hour']==hour) &
                     (train_df['clip_label']==0))
        base_rows = train_df.loc[base_mask, SPECTRAL_KEYS]
        if len(base_rows) == 0:
            continue
        hour_mean = base_rows.mean()
        hour_std  = base_rows.std().replace(0, 1e-6)

        tr_h = train_df['clip_hour'] == hour
        if tr_h.sum() > 0:
            z = (train_df.loc[tr_h,SPECTRAL_KEYS]-hour_mean)/hour_std
            z.columns = Z_KEYS
            z_tr_rows.append(z)

        if test_df is not None:
            te_h = test_df['clip_hour'] == hour
            if te_h.sum() > 0:
                z = (test_df.loc[te_h,SPECTRAL_KEYS]-hour_mean)/hour_std
                z.columns = Z_KEYS
                z_te_rows.append(z)

    z_tr = (pd.concat(z_tr_rows).sort_index() if z_tr_rows else
            pd.DataFrame(0,index=train_df.index,columns=Z_KEYS))

    if test_df is not None:
        # Handle unseen hours
        seen = set(train_df['clip_hour'].unique())
        unseen_mask = ~test_df['clip_hour'].isin(seen)
        if unseen_mask.sum() > 0:
            z_unseen = pd.DataFrame(
                0, index=test_df.index[unseen_mask], columns=Z_KEYS)
            z_te_rows.append(z_unseen)
        z_te = (pd.concat(z_te_rows).sort_index() if z_te_rows else
                pd.DataFrame(0,index=test_df.index,columns=Z_KEYS))
        return z_tr, z_te

    return z_tr


def get_feature_matrix(clip_df, z_df, pca_result, fs):
    """Assemble feature matrix based on feature set config."""
    parts = []
    if fs['use_raw']:
        parts.append(clip_df[SPECTRAL_KEYS].values)
    if fs['use_z']:
        parts.append(z_df.values)
    if fs['use_emb']:
        parts.append(pca_result)
    if not parts:
        raise ValueError("No features selected")
    return np.hstack(parts).astype(np.float32)


def evaluate_gb(X, y):
    """Run GB with threshold tuning, return (p, r, beat)."""
    n_pos = y.sum()
    n_neg = (y==0).sum()
    w     = np.where(y==1, n_neg/max(n_pos,1), 1.0)
    gb    = GradientBoostingClassifier(
        n_estimators=100, max_depth=3,
        learning_rate=0.1, random_state=RANDOM_SEED, subsample=0.8)
    gb.fit(X, y, sample_weight=w)
    probs = gb.predict_proba(X)[:,1]
    # Note: this is train-set evaluation — only used within CV folds
    return gb, probs


def per_site_cv(clip_df, fs_config, fs_name):
    """Corrected clip-level CV for one recorder."""
    clip_names  = clip_df['clip_name'].values
    clip_labels = clip_df['clip_label'].values
    emb_cols    = [c for c in clip_df.columns if c.startswith('emb_')]

    if len(np.unique(clip_labels)) < 2:
        return None
    if clip_labels.sum() < N_FOLDS:
        return None

    cv    = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                            random_state=RANDOM_SEED)
    probs = np.zeros(len(clip_df), dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(
            cv.split(clip_names, clip_labels)):

        clip_tr = clip_df.iloc[tr_idx].copy().reset_index(drop=True)
        clip_va = clip_df.iloc[va_idx].copy().reset_index(drop=True)
        y_tr    = clip_labels[tr_idx].astype(int)

        # Z-scores
        z_tr, z_va = compute_zscores_train_only(clip_tr, clip_va)

        # PCA on training
        pca       = PCA(n_components=50, random_state=RANDOM_SEED)
        emb_tr_pc = pca.fit_transform(clip_tr[emb_cols].values)
        emb_va_pc = pca.transform(clip_va[emb_cols].values)

        X_tr = get_feature_matrix(clip_tr, z_tr, emb_tr_pc, fs_config)
        X_va = get_feature_matrix(clip_va, z_va, emb_va_pc, fs_config)

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)

        n_pos = y_tr.sum()
        n_neg = (y_tr==0).sum()
        w     = np.where(y_tr==1, n_neg/max(n_pos,1), 1.0)
        gb    = GradientBoostingClassifier(
            n_estimators=100, max_depth=3,
            learning_rate=0.1, random_state=RANDOM_SEED, subsample=0.8)
        gb.fit(X_tr_s, y_tr, sample_weight=w)
        probs[va_idx] = gb.predict_proba(X_va_s)[:,1]

    y_all = clip_labels.astype(int)
    prec, rec, thresh = precision_recall_curve(y_all, probs)
    valid = np.where((prec[:-1]>=0.70)&(rec[:-1]>=0.70))[0]
    if len(valid) > 0:
        bi    = valid[np.argmax(prec[valid]+rec[valid])]
        bt    = float(thresh[bi])
        preds = (probs>=bt).astype(int)
        p,r,_,_ = precision_recall_fscore_support(
            y_all,preds,average="binary",zero_division=0)
        beat = True
    else:
        bi    = np.argmax(prec[:-1]+rec[:-1])
        bt    = float(thresh[bi])
        preds = (probs>=bt).astype(int)
        p,r,_,_ = precision_recall_fscore_support(
            y_all,preds,average="binary",zero_division=0)
        beat = False
    return {'p':p,'r':r,'beat':beat}


def cross_site_eval(all_clips, test_name, fs_config):
    """Train on all other recorders, test on test_name."""
    test_df  = all_clips[test_name].copy().reset_index(drop=True)
    train_df = pd.concat(
        [df for name,df in all_clips.items() if name!=test_name],
        ignore_index=True)

    y_tr = train_df['clip_label'].values.astype(int)
    y_te = test_df['clip_label'].values.astype(int)

    if len(np.unique(y_te)) < 2:
        return None

    emb_cols = [c for c in train_df.columns if c.startswith('emb_')]

    z_tr, z_te = compute_zscores_train_only(train_df, test_df)

    pca       = PCA(n_components=50, random_state=RANDOM_SEED)
    emb_tr_pc = pca.fit_transform(train_df[emb_cols].values)
    emb_te_pc = pca.transform(test_df[emb_cols].values)

    X_tr = get_feature_matrix(train_df, z_tr, emb_tr_pc, fs_config)
    X_te = get_feature_matrix(test_df,  z_te, emb_te_pc, fs_config)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    n_pos = y_tr.sum()
    n_neg = (y_tr==0).sum()
    w     = np.where(y_tr==1, n_neg/max(n_pos,1), 1.0)
    gb    = GradientBoostingClassifier(
        n_estimators=100, max_depth=3,
        learning_rate=0.1, random_state=RANDOM_SEED, subsample=0.8)
    gb.fit(X_tr_s, y_tr, sample_weight=w)

    probs = gb.predict_proba(X_te_s)[:,1]
    prec, rec, thresh = precision_recall_curve(y_te, probs)
    valid = np.where((prec[:-1]>=0.70)&(rec[:-1]>=0.70))[0]
    if len(valid) > 0:
        bi    = valid[np.argmax(prec[valid]+rec[valid])]
        bt    = float(thresh[bi])
        preds = (probs>=bt).astype(int)
        p,r,_,_ = precision_recall_fscore_support(
            y_te,preds,average="binary",zero_division=0)
        beat = True
    else:
        bi    = np.argmax(prec[:-1]+rec[:-1])
        bt    = float(thresh[bi])
        preds = (probs>=bt).astype(int)
        p,r,_,_ = precision_recall_fscore_support(
            y_te,preds,average="binary",zero_division=0)
        beat = False
    return {'p':p,'r':r,'beat':beat}


if __name__ == '__main__':

    OUT_REPORT = os.path.join(BASE_DIR, "feature_ablation_results.txt")

    # ── Load all data ─────────────────────────────────────────────────────────
    print("Loading all recorder data...")
    all_clips = {}

    for rec in RECORDERS:
        name      = rec['name']
        feat_path = os.path.join(BASE_DIR, rec['feat'])
        emb_path  = os.path.join(BASE_DIR, rec['emb'])
        lab_path  = os.path.join(BASE_DIR, rec['lab'])

        if not os.path.exists(feat_path) or not os.path.exists(emb_path):
            print(f"  {name}: files not found — skipping")
            continue

        feat_df    = pd.read_csv(feat_path)
        embeddings = np.load(emb_path)
        labels     = np.load(lab_path)

        if len(feat_df) != len(embeddings):
            n = min(len(feat_df), len(embeddings))
            feat_df    = feat_df.iloc[:n].reset_index(drop=True)
            embeddings = embeddings[:n]
            labels     = labels[:n]

        clip_df = build_clip_features(feat_df, embeddings, labels)
        clip_df = apply_rain_filter(clip_df)
        clip_df['recorder'] = name

        print(f"  {name}: {len(clip_df)} clips  "
              f"({int(clip_df['clip_label'].sum())} sim)")
        all_clips[name] = clip_df

    available = list(all_clips.keys())

    # ── Run ablation ──────────────────────────────────────────────────────────
    results = {fs: {} for fs in FEATURE_SETS}

    for fs_name, fs_config in FEATURE_SETS.items():
        n_feats = (41 if fs_config['use_raw'] else 0) + \
                  (41 if fs_config['use_z']   else 0) + \
                  (50 if fs_config['use_emb'] else 0)
        print(f"\n{'='*60}")
        print(f"Feature set {fs_name} ({n_feats} features)")
        print(f"{'='*60}")

        for rec_name in available:
            print(f"\n  {rec_name}:")

            # Per-site CV
            ps = per_site_cv(all_clips[rec_name], fs_config, fs_name)
            if ps:
                print(f"    Per-site:    P={ps['p']:.3f}  R={ps['r']:.3f}  "
                      f"{'BEAT' if ps['beat'] else 'miss'}")
            else:
                print(f"    Per-site:    insufficient data")
                ps = {'p':0,'r':0,'beat':False}

            # Cross-site
            cs = cross_site_eval(all_clips, rec_name, fs_config)
            if cs:
                print(f"    Cross-site:  P={cs['p']:.3f}  R={cs['r']:.3f}  "
                      f"{'BEAT' if cs['beat'] else 'miss'}")
            else:
                cs = {'p':0,'r':0,'beat':False}

            results[fs_name][rec_name] = {'per_site':ps, 'cross_site':cs}

    # ── Summary tables ────────────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("ABLATION SUMMARY — PER-SITE CV PRECISION")
    print("=" * 70)
    print(f"{'Feature Set':<22} {'AM4':>6} {'AM2':>6} {'AM5':>6} "
          f"{'AM6':>6} {'AM1':>6} {'Pass':>5}")
    print("-" * 70)
    for fs_name in FEATURE_SETS:
        row  = results[fs_name]
        vals = [row.get(r,{}).get('per_site',{}).get('p',0)
                for r in ['AM4','AM2','AM5','AM6','AM1']]
        n_beat = sum(1 for r in ['AM4','AM2','AM5','AM6','AM1']
                     if row.get(r,{}).get('per_site',{}).get('beat',False))
        print(f"{fs_name:<22} "+" ".join(f"{v:>6.3f}" for v in vals)+
              f" {n_beat:>4}/5")

    print("\n\n" + "=" * 70)
    print("ABLATION SUMMARY — CROSS-SITE PRECISION")
    print("(Train on 4 recorders, test on 1 held-out)")
    print("=" * 70)
    print(f"{'Feature Set':<22} {'AM4':>6} {'AM2':>6} {'AM5':>6} "
          f"{'AM6':>6} {'AM1':>6} {'Pass':>5}")
    print("-" * 70)
    for fs_name in FEATURE_SETS:
        row  = results[fs_name]
        vals = [row.get(r,{}).get('cross_site',{}).get('p',0)
                for r in ['AM4','AM2','AM5','AM6','AM1']]
        n_beat = sum(1 for r in ['AM4','AM2','AM5','AM6','AM1']
                     if row.get(r,{}).get('cross_site',{}).get('beat',False))
        print(f"{fs_name:<22} "+" ".join(f"{v:>6.3f}" for v in vals)+
              f" {n_beat:>4}/5")

    print("\n\n" + "=" * 70)
    print("KEY QUESTION: which feature set generalises best cross-site?")
    print("=" * 70)
    best_cs = max(FEATURE_SETS.keys(),
                  key=lambda fs: sum(
                      results[fs].get(r,{}).get('cross_site',{}).get('p',0)
                      for r in available))
    best_ps = max(FEATURE_SETS.keys(),
                  key=lambda fs: sum(
                      results[fs].get(r,{}).get('per_site',{}).get('p',0)
                      for r in available))
    print(f"Best per-site:    {best_ps}")
    print(f"Best cross-site:  {best_cs}")

    # ── Save report ───────────────────────────────────────────────────────────
    with open(OUT_REPORT, "w", encoding="utf-8") as fh:
        fh.write("Feature Ablation Results\n")
        fh.write("=" * 50 + "\n\n")
        fh.write("Tests which feature groups generalise across sites.\n\n")
        for fs_name, fs_config in FEATURE_SETS.items():
            fh.write(f"\n{fs_name}:\n")
            for rec_name in available:
                r  = results[fs_name].get(rec_name, {})
                ps = r.get('per_site',  {'p':0,'r':0,'beat':False})
                cs = r.get('cross_site', {'p':0,'r':0,'beat':False})
                fh.write(f"  {rec_name}: "
                         f"per-site P={ps['p']:.3f} "
                         f"{'BEAT' if ps['beat'] else 'miss'} | "
                         f"cross-site P={cs['p']:.3f} "
                         f"{'BEAT' if cs['beat'] else 'miss'}\n")

    print(f"\nReport saved -> {OUT_REPORT}")
    print("\nDone.")
