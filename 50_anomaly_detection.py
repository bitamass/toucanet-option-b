"""
50_anomaly_detection.py
------------------------
Learn NORMAL instead of DISTURBANCE — anomaly detection approach.

THE KEY INSIGHT (Sammy's suggestion):
  Every rainforest sounds different.
  Normal AM4 ≠ Normal AM2.
  But "abnormal relative to itself" might generalise.

  Instead of:
    Train classifier on (baseline vs simulation)
    Ask: does this clip look like disturbance from other sites?

  This approach:
    Train ONLY on baseline clips
    Ask: does this clip look abnormal relative to this site?

  Advantages:
  1. No simulation data needed at new sites
  2. The model learns site normality not site disturbance
  3. Anomaly relative to self may transfer across sites

THREE ANOMALY DETECTION METHODS:

  Method 1 — Isolation Forest
    Ensemble of random trees that isolates anomalies.
    Short path length in tree = anomaly.
    Fast, no hyperparameter tuning needed.
    Works well on high-dimensional data.

  Method 2 — One-Class SVM
    Learns a boundary around normal data in feature space.
    Anything outside boundary = anomaly.
    More principled but slower and needs kernel tuning.

  Method 3 — Autoencoder reconstruction error
    Train neural network to reconstruct baseline clips.
    At inference: high reconstruction error = anomaly.
    Learns compressed representation of normality.
    Most flexible — can capture complex normal patterns.
    Uses PyTorch (CPU friendly, small network).

EVALUATION:
  Per-site: train on baseline only, test on baseline+simulation
  LODO cross-site: train on other sites' baseline, test on new site
  Compare to GB supervised baseline.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (precision_recall_fscore_support,
                              precision_recall_curve, roc_auc_score)

BASE_DIR = r"C:\Users\BitaMassoudi\KasmirWorldProjects\toucanet-option-b"
SEED     = 42
torch.manual_seed(SEED); np.random.seed(SEED)

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


class Autoencoder(nn.Module):
    """Small autoencoder for baseline normality learning."""
    def __init__(self, input_dim, bottleneck=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, bottleneck), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, input_dim),
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))
    def reconstruction_error(self, x):
        with torch.no_grad():
            recon = self.forward(x)
            return ((recon - x)**2).mean(dim=1).numpy()


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


def get_features(df, scaler=None, pca=None, fit=True):
    """Get standardised + PCA reduced features."""
    ec=[c for c in df.columns if c.startswith(('em_','es_','ex_'))]
    X=np.hstack([df[SPECTRAL_KEYS].values,
                  df[ec].values]).astype(np.float32)
    if fit:
        scaler=StandardScaler(); X=scaler.fit_transform(X)
        pca=PCA(n_components=80,random_state=SEED); X=pca.fit_transform(X)
        return X, scaler, pca
    else:
        X=scaler.transform(X); X=pca.transform(X)
        return X


def evaluate_anomaly_scores(y, scores):
    """scores: higher = more anomalous. Find best threshold."""
    pr,rc,th=precision_recall_curve(y,scores)
    valid=np.where((pr[:-1]>=0.70)&(rc[:-1]>=0.70))[0]
    if len(valid)>0:
        bi=valid[np.argmax(pr[valid]+rc[valid])]; bt=float(th[bi])
    else:
        bi=np.argmax(pr[:-1]+rc[:-1]); bt=float(th[bi])
    preds=(scores>=bt).astype(int)
    p,r,_,_=precision_recall_fscore_support(y,preds,average='binary',zero_division=0)
    try: auc=roc_auc_score(y,scores)
    except: auc=0.0
    return p,r,(p>=0.70 and r>=0.70),auc


def run_isolation_forest(X_train_base, X_all, y_all, contamination=0.15):
    """Train IF on baseline only, get anomaly scores for all."""
    clf=IsolationForest(n_estimators=200,contamination=contamination,
                        random_state=SEED,n_jobs=-1)
    clf.fit(X_train_base)
    # IF returns negative scores — more negative = more anomalous
    scores=-clf.score_samples(X_all)
    return evaluate_anomaly_scores(y_all, scores)


def run_ocsvm(X_train_base, X_all, y_all, nu=0.1):
    """Train One-Class SVM on baseline only."""
    clf=OneClassSVM(kernel='rbf',nu=nu,gamma='scale')
    clf.fit(X_train_base)
    scores=-clf.score_samples(X_all)
    return evaluate_anomaly_scores(y_all, scores)


def run_autoencoder(X_train_base, X_all, y_all, n_epochs=50):
    """Train autoencoder on baseline, use reconstruction error as anomaly score."""
    X_t=torch.tensor(X_train_base,dtype=torch.float32)
    X_a=torch.tensor(X_all,dtype=torch.float32)
    model=Autoencoder(X_train_base.shape[1])
    opt=optim.Adam(model.parameters(),lr=1e-3)
    crit=nn.MSELoss()
    dl=DataLoader(TensorDataset(X_t),batch_size=32,shuffle=True)
    model.train()
    for ep in range(n_epochs):
        for (xb,) in dl:
            opt.zero_grad()
            loss=crit(model(xb),xb)
            loss.backward(); opt.step()
    model.eval()
    scores=model.reconstruction_error(X_a)
    return evaluate_anomaly_scores(y_all, scores)


if __name__ == '__main__':

    print("Anomaly Detection — Learn Normal, Flag Abnormal")
    print("="*65)

    all_clips={}
    for name,ff,ef,lf in RECORDERS:
        fp=os.path.join(BASE_DIR,ff); ep=os.path.join(BASE_DIR,ef)
        if not os.path.exists(fp) or not os.path.exists(ep): continue
        feat_df=pd.read_csv(fp); emb=np.load(ep)
        n=min(len(feat_df),len(emb))
        feat_df=feat_df.iloc[:n].reset_index(drop=True); emb=emb[:n]
        clip_df=build_clip_df(feat_df,emb)
        all_clips[name]=clip_df
        print(f"  {name}: {len(clip_df)} clips ({int(clip_df['clip_label'].sum())} sim)")

    available=list(all_clips.keys())
    methods=['IF','OCSVM','AE']
    per_site={m:{} for m in methods}
    lodo_res={m:{} for m in methods}

    # ── Per-site evaluation ───────────────────────────────────────────────────
    print("\n" + "="*60)
    print("PER-SITE — train on baseline only, test on all clips")
    print("="*60)

    for name in available:
        clip_df=all_clips[name].copy().reset_index(drop=True)
        y=clip_df['clip_label'].values.astype(int)
        base_df=clip_df[clip_df['clip_label']==0].copy()

        print(f"\n{name}: {int(y.sum())} sim, {int((y==0).sum())} base")

        # Use 5-fold CV: train anomaly detector on baseline in train fold
        cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
        scores_if=np.zeros(len(clip_df))
        scores_oc=np.zeros(len(clip_df))
        scores_ae=np.zeros(len(clip_df))

        for tr,va in cv.split(clip_df,y):
            tdf=clip_df.iloc[tr]; vdf=clip_df.iloc[va]
            base_tr=tdf[tdf['clip_label']==0]

            X_base,sc,pca=get_features(base_tr,fit=True)
            X_va=get_features(vdf,sc,pca,fit=False)

            # IF
            clf_if=IsolationForest(n_estimators=200,contamination=0.15,
                                   random_state=SEED,n_jobs=-1)
            clf_if.fit(X_base)
            scores_if[va]=-clf_if.score_samples(X_va)

            # OCSVM
            clf_oc=OneClassSVM(kernel='rbf',nu=0.1,gamma='scale')
            clf_oc.fit(X_base)
            scores_oc[va]=-clf_oc.score_samples(X_va)

            # AE
            X_base_t=torch.tensor(X_base,dtype=torch.float32)
            X_va_t=torch.tensor(X_va,dtype=torch.float32)
            ae=Autoencoder(X_base.shape[1])
            opt=optim.Adam(ae.parameters(),lr=1e-3)
            crit=nn.MSELoss()
            dl=DataLoader(TensorDataset(X_base_t),batch_size=32,shuffle=True)
            ae.train()
            for _ in range(40):
                for (xb,) in dl:
                    opt.zero_grad(); loss=crit(ae(xb),xb); loss.backward(); opt.step()
            ae.eval()
            scores_ae[va]=ae.reconstruction_error(X_va_t)

        p_if,r_if,b_if,auc_if=evaluate_anomaly_scores(y,scores_if)
        p_oc,r_oc,b_oc,auc_oc=evaluate_anomaly_scores(y,scores_oc)
        p_ae,r_ae,b_ae,auc_ae=evaluate_anomaly_scores(y,scores_ae)

        print(f"  Isolation Forest: P={p_if:.3f} R={r_if:.3f} AUC={auc_if:.3f} {'BEAT' if b_if else 'miss'}")
        print(f"  One-Class SVM:    P={p_oc:.3f} R={r_oc:.3f} AUC={auc_oc:.3f} {'BEAT' if b_oc else 'miss'}")
        print(f"  Autoencoder:      P={p_ae:.3f} R={r_ae:.3f} AUC={auc_ae:.3f} {'BEAT' if b_ae else 'miss'}")

        per_site['IF'][name]=(p_if,r_if,b_if)
        per_site['OCSVM'][name]=(p_oc,r_oc,b_oc)
        per_site['AE'][name]=(p_ae,r_ae,b_ae)

    # ── LODO cross-site ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("LODO — train on OTHER SITES' baseline, test on new site")
    print("This is the key cross-site test — no simulation data needed")
    print("="*60)

    for test_name in available:
        train_names=[r for r in available if r!=test_name]
        test_df=all_clips[test_name].copy().reset_index(drop=True)
        # Use ONLY baseline from training sites
        train_base=pd.concat([all_clips[r][all_clips[r]['clip_label']==0]
                               for r in train_names],ignore_index=True)
        y_te=test_df['clip_label'].values.astype(int)
        if len(np.unique(y_te))<2: continue

        print(f"\nTest: {test_name}  "
              f"(train baseline from {train_names}, {len(train_base)} clips)")

        X_base,sc,pca=get_features(train_base,fit=True)
        X_te=get_features(test_df,sc,pca,fit=False)

        p_if,r_if,b_if,auc_if=run_isolation_forest(X_base,X_te,y_te)
        p_oc,r_oc,b_oc,auc_oc=run_ocsvm(X_base,X_te,y_te)
        p_ae,r_ae,b_ae,auc_ae=run_autoencoder(X_base,X_te,y_te)

        print(f"  IF:    P={p_if:.3f} R={r_if:.3f} AUC={auc_if:.3f} {'BEAT' if b_if else 'miss'}")
        print(f"  OCSVM: P={p_oc:.3f} R={r_oc:.3f} AUC={auc_oc:.3f} {'BEAT' if b_oc else 'miss'}")
        print(f"  AE:    P={p_ae:.3f} R={r_ae:.3f} AUC={auc_ae:.3f} {'BEAT' if b_ae else 'miss'}")

        lodo_res['IF'][test_name]=(p_if,r_if,b_if)
        lodo_res['OCSVM'][test_name]=(p_oc,r_oc,b_oc)
        lodo_res['AE'][test_name]=(p_ae,r_ae,b_ae)

    # ── Summary ───────────────────────────────────────────────────────────────
    method_labels={'IF':'Isolation Forest','OCSVM':'One-Class SVM','AE':'Autoencoder'}

    print("\n\n"+"="*70)
    print("ANOMALY DETECTION SUMMARY — PER-SITE")
    print("="*70)
    print(f"{'Method':<18} {'AM4':>7} {'AM2':>7} {'AM5':>7} {'AM6':>7} {'AM1':>7} {'Beat':>5}")
    print("-"*70)
    for m in methods:
        vals=[per_site[m].get(n,(0,0,False))[0] for n in ['AM4','AM2','AM5','AM6','AM1']]
        nb=sum(1 for n in available if per_site[m].get(n,(0,0,False))[2])
        print(f"{method_labels[m]:<18} "+" ".join(f"{v:>7.3f}" for v in vals)+f" {nb:>4}/5")

    print("\n"+"="*70)
    print("ANOMALY DETECTION SUMMARY — LODO CROSS-SITE")
    print("(key question: does abnormal-relative-to-self generalise?)")
    print("="*70)
    print(f"{'Method':<18} {'AM4':>7} {'AM2':>7} {'AM5':>7} {'AM6':>7} {'AM1':>7} {'Beat':>5}")
    print("-"*70)
    for m in methods:
        vals=[lodo_res[m].get(n,(0,0,False))[0] for n in ['AM4','AM2','AM5','AM6','AM1']]
        nb=sum(1 for n in available if lodo_res[m].get(n,(0,0,False))[2])
        avg=np.mean(vals)
        print(f"{method_labels[m]:<18} "+" ".join(f"{v:>7.3f}" for v in vals)+
              f" {nb:>4}/5  avg={avg:.3f}")

    print("\nComparison to supervised GB cross-site baseline: avg P=0.345")
    for m in methods:
        avg=np.mean([lodo_res[m].get(n,(0,0,False))[0] for n in available])
        print(f"  {method_labels[m]}: avg P={avg:.3f}  change={avg-0.345:+.3f}")

    print("\nDone.")
