"""
47_mlp_domain_adversarial.py
-----------------------------
MLP and domain-adversarial MLP for per-site and cross-site evaluation.

THREE MODELS COMPARED:

Model 1 — Standard MLP
  Same features as gradient boosting (script 34/42).
  Replace GB with a 3-layer MLP.
  Baseline to check if MLP improves on GB per-site.

Model 2 — MLP on ordered segment sequence
  Flatten first 6 segments in time order: 6 × 1536 = 9216 dims.
  Compress through MLP: 9216 → 512 → 128 → 2.
  Network learns which temporal positions matter.
  Addresses Sammy's ordered segment suggestion properly.

Model 3 — Domain-adversarial MLP (DANN)
  Architecture:
    Shared encoder: input → 256 → 128 (learns site-invariant features)
    Disturbance head: 128 → 64 → 2 (predict disturbance/baseline)
    Site head: 128 → 64 → n_sites (predict which recorder)
  Training:
    Disturbance head: minimise classification loss (learn to detect)
    Site head: MAXIMISE classification loss via gradient reversal
    (force encoder to learn features that cannot identify site)
  This is Ganin et al. 2016 DANN applied to bioacoustic detection.

EVALUATION:
  Per-site 5-fold CV for all three models.
  LODO cross-site for all three models.
  Key comparison: does DANN improve cross-site precision?
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (precision_recall_fscore_support,
                              precision_recall_curve)

BASE_DIR = r"C:\Users\BitaMassoudi\KasmirWorldProjects\toucanet-option-b"
SEED     = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

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
Z_KEYS = [f"z_{k}" for k in SPECTRAL_KEYS]
K_SEGS = 6   # ordered segments to use
N_EPOCHS_SMALL = 50   # for small datasets
N_EPOCHS_LARGE = 30   # for large datasets
BATCH_SIZE = 32
LAMBDA_DOMAIN = 0.1   # domain adversarial loss weight


# ── MLP Architectures ─────────────────────────────────────────────────────────

class StandardMLP(nn.Module):
    """Standard 3-layer MLP for disturbance classification."""
    def __init__(self, input_dim, hidden1=256, hidden2=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden2, 2)
        )
    def forward(self, x):
        return self.net(x)


class GradientReversal(torch.autograd.Function):
    """Gradient reversal layer for domain-adversarial training."""
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(torch.tensor(lambda_))
        return x.clone()
    @staticmethod
    def backward(ctx, grad_output):
        lambda_, = ctx.saved_tensors
        return -lambda_ * grad_output, None


class DANNModel(nn.Module):
    """
    Domain-Adversarial Neural Network.
    Shared encoder + disturbance head + site head with gradient reversal.
    """
    def __init__(self, input_dim, n_sites, hidden=256, bottleneck=128):
        super().__init__()
        # Shared encoder — learns site-invariant features
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, bottleneck),
            nn.BatchNorm1d(bottleneck),
            nn.ReLU(),
        )
        # Disturbance classification head
        self.dist_head = nn.Sequential(
            nn.Linear(bottleneck, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )
        # Site classification head (trained adversarially)
        self.site_head = nn.Sequential(
            nn.Linear(bottleneck, 64),
            nn.ReLU(),
            nn.Linear(64, n_sites)
        )

    def forward(self, x, lambda_=1.0, return_domain=True):
        features = self.encoder(x)
        dist_out  = self.dist_head(features)
        if return_domain:
            reversed_features = GradientReversal.apply(features, lambda_)
            site_out = self.site_head(reversed_features)
            return dist_out, site_out
        return dist_out


# ── Feature building ──────────────────────────────────────────────────────────

def build_clip_df(feat_df, emb):
    rows = []
    for cn in feat_df['clip_name'].unique():
        mask  = feat_df['clip_name'].values == cn
        cl    = feat_df['clip_label'].values[mask][0]
        hour  = feat_df['clip_hour'].values[mask][0]
        ce    = emb[mask]
        n_seg = len(ce)
        row   = {'clip_name':cn,'clip_label':cl,'clip_hour':hour}
        for k in SPECTRAL_KEYS:
            if k in feat_df.columns:
                row[k] = feat_df.loc[mask,k].values[0]
        # Aggregated
        for i,v in enumerate(ce.mean(0)): row[f'agg_mean_{i}']=v
        for i,v in enumerate(ce.std(0)):  row[f'agg_std_{i}']=v
        for i,v in enumerate(ce.max(0)):  row[f'agg_max_{i}']=v
        # Ordered segments (first K in time order)
        for k in range(K_SEGS):
            seg = ce[k] if k < n_seg else np.zeros(ce.shape[1])
            for i,v in enumerate(seg): row[f'ord_s{k}_{i}']=v
        rows.append(row)
    return pd.DataFrame(rows)


def compute_zscores(train_df, test_df=None):
    z_tr_rows, z_te_rows = [], []
    for hour in train_df['clip_hour'].unique():
        bm = (train_df['clip_hour']==hour)&(train_df['clip_label']==0)
        br = train_df.loc[bm,SPECTRAL_KEYS]
        if len(br)==0: continue
        hm=br.mean(); hs=br.std().replace(0,1e-6)
        th=train_df['clip_hour']==hour
        if th.sum()>0:
            z=(train_df.loc[th,SPECTRAL_KEYS]-hm)/hs
            z.columns=Z_KEYS; z_tr_rows.append(z)
        if test_df is not None:
            te_h=test_df['clip_hour']==hour
            if te_h.sum()>0:
                z=(test_df.loc[te_h,SPECTRAL_KEYS]-hm)/hs
                z.columns=Z_KEYS; z_te_rows.append(z)
    z_tr=pd.concat(z_tr_rows).sort_index() if z_tr_rows else pd.DataFrame(0,index=train_df.index,columns=Z_KEYS)
    if test_df is not None:
        seen=set(train_df['clip_hour'].unique())
        for h in set(test_df['clip_hour'].unique())-seen:
            te_h=test_df['clip_hour']==h
            if te_h.sum()>0:
                z_te_rows.append(pd.DataFrame(0,index=test_df.index[te_h],columns=Z_KEYS))
        z_te=pd.concat(z_te_rows).sort_index() if z_te_rows else pd.DataFrame(0,index=test_df.index,columns=Z_KEYS)
        return z_tr,z_te
    return z_tr


def get_feature_matrices(train_df, test_df=None):
    """Get standard features (agg + z-scores) with PCA."""
    agg_cols=[c for c in train_df.columns if c.startswith('agg_')]
    ord_cols=[c for c in train_df.columns if c.startswith('ord_')]

    z_tr,z_te=(compute_zscores(train_df,test_df) if test_df is not None
               else (compute_zscores(train_df),None))

    pca_agg=PCA(n_components=50,random_state=SEED)
    etr_agg=pca_agg.fit_transform(train_df[agg_cols].values)

    pca_ord=PCA(n_components=60,random_state=SEED)
    etr_ord=pca_ord.fit_transform(train_df[ord_cols].values)

    X_tr_std=np.hstack([train_df[SPECTRAL_KEYS].values,
                         z_tr.values,etr_agg]).astype(np.float32)
    X_tr_ord=np.hstack([etr_ord]).astype(np.float32)

    if test_df is not None:
        ete_agg=pca_agg.transform(test_df[agg_cols].values)
        ete_ord=pca_ord.transform(test_df[ord_cols].values)
        X_te_std=np.hstack([test_df[SPECTRAL_KEYS].values,
                              z_te.values,ete_agg]).astype(np.float32)
        X_te_ord=np.hstack([ete_ord]).astype(np.float32)
        return X_tr_std,X_tr_ord,X_te_std,X_te_ord

    return X_tr_std,X_tr_ord


def train_mlp(model, X_tr, y_tr, X_va, n_epochs, pos_weight):
    """Train standard MLP, return val probabilities."""
    sc=StandardScaler()
    X_tr_s=torch.tensor(sc.fit_transform(X_tr))
    X_va_s=torch.tensor(sc.transform(X_va))
    y_tr_t=torch.tensor(y_tr,dtype=torch.long)
    w=torch.tensor([1.0,float(pos_weight)])
    criterion=nn.CrossEntropyLoss(weight=w)
    opt=optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4)
    ds=TensorDataset(X_tr_s,y_tr_t)
    dl=DataLoader(ds,batch_size=BATCH_SIZE,shuffle=True)
    model.train()
    for epoch in range(n_epochs):
        for xb,yb in dl:
            opt.zero_grad()
            loss=criterion(model(xb),yb)
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        logits=model(X_va_s)
        probs=torch.softmax(logits,dim=1)[:,1].numpy()
    return probs, sc


def train_dann(model, X_tr, y_tr, site_tr, X_va, n_epochs, pos_weight, n_sites):
    """Train DANN model, return val probabilities."""
    sc=StandardScaler()
    X_tr_s=torch.tensor(sc.fit_transform(X_tr))
    X_va_s=torch.tensor(sc.transform(X_va))
    y_tr_t=torch.tensor(y_tr,dtype=torch.long)
    s_tr_t=torch.tensor(site_tr,dtype=torch.long)
    w=torch.tensor([1.0,float(pos_weight)])
    dist_criterion=nn.CrossEntropyLoss(weight=w)
    site_criterion=nn.CrossEntropyLoss()
    opt=optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4)
    ds=TensorDataset(X_tr_s,y_tr_t,s_tr_t)
    dl=DataLoader(ds,batch_size=BATCH_SIZE,shuffle=True)
    model.train()
    for epoch in range(n_epochs):
        # Gradually increase domain loss weight
        p=epoch/n_epochs
        lambda_=2.0/(1+np.exp(-10*p))-1
        for xb,yb,sb in dl:
            opt.zero_grad()
            dist_out,site_out=model(xb,lambda_=lambda_*LAMBDA_DOMAIN)
            dist_loss=dist_criterion(dist_out,yb)
            site_loss=site_criterion(site_out,sb)
            loss=dist_loss+LAMBDA_DOMAIN*site_loss
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        logits,_=model(X_va_s,return_domain=True)
        probs=torch.softmax(logits,dim=1)[:,1].numpy()
    return probs, sc


def best_threshold(y, probs):
    pr,rc,th=precision_recall_curve(y,probs)
    valid=np.where((pr[:-1]>=0.70)&(rc[:-1]>=0.70))[0]
    if len(valid)>0:
        bi=valid[np.argmax(pr[valid]+rc[valid])]; bt=float(th[bi])
    else:
        bi=np.argmax(pr[:-1]+rc[:-1]); bt=float(th[bi])
    preds=(probs>=bt).astype(int)
    p,r,_,_=precision_recall_fscore_support(y,preds,average='binary',zero_division=0)
    return p,r,(p>=0.70 and r>=0.70)


if __name__ == '__main__':

    # Check for torch
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
    except ImportError:
        print("PyTorch not installed. Run: pip install torch --break-system-packages")
        exit()

    print("Loading data...")
    all_clips={}
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

    available=list(all_clips.keys())
    site_map={n:i for i,n in enumerate(available)}
    n_sites=len(available)

    per_site_results={m:{} for m in ['MLP','MLP_ord','DANN']}
    lodo_results={m:{} for m in ['MLP','MLP_ord','DANN']}

    # ── Per-site CV ───────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("PER-SITE 5-FOLD CV")
    print("="*60)

    for name in available:
        clip_df=all_clips[name].copy().reset_index(drop=True)
        y=clip_df['clip_label'].values.astype(int)
        n_epochs=N_EPOCHS_SMALL if len(clip_df)<500 else N_EPOCHS_LARGE
        pos_weight=(y==0).sum()/max(y.sum(),1)

        cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
        probs_mlp=np.zeros(len(clip_df),dtype=np.float32)
        probs_ord=np.zeros(len(clip_df),dtype=np.float32)
        probs_dann=np.zeros(len(clip_df),dtype=np.float32)

        print(f"\n{name}:")
        for fold,(tr,va) in enumerate(cv.split(clip_df,y)):
            tdf=clip_df.iloc[tr].copy().reset_index(drop=True)
            vdf=clip_df.iloc[va].copy().reset_index(drop=True)
            ytr=y[tr]; yva=y[va]
            site_tr=np.zeros(len(tdf),dtype=int)  # all same site per-site CV

            X_tr_std,X_tr_ord,X_va_std,X_va_ord=get_feature_matrices(tdf,vdf)

            # MLP standard
            mlp=StandardMLP(X_tr_std.shape[1])
            probs_mlp[va],_=train_mlp(mlp,X_tr_std,ytr,X_va_std,n_epochs,pos_weight)

            # MLP ordered
            mlp_ord=StandardMLP(X_tr_ord.shape[1],hidden1=128,hidden2=64)
            probs_ord[va],_=train_mlp(mlp_ord,X_tr_ord,ytr,X_va_ord,n_epochs,pos_weight)

            # DANN (per-site: site head is random since all same site)
            dann=DANNModel(X_tr_std.shape[1],n_sites=1)
            probs_dann[va],_=train_dann(dann,X_tr_std,ytr,site_tr,X_va_std,n_epochs,pos_weight,1)

            print(f"  Fold {fold+1} done")

        p_mlp,r_mlp,b_mlp=best_threshold(y,probs_mlp)
        p_ord,r_ord,b_ord=best_threshold(y,probs_ord)
        p_dann,r_dann,b_dann=best_threshold(y,probs_dann)

        print(f"  MLP standard: P={p_mlp:.3f} R={r_mlp:.3f} {'BEAT' if b_mlp else 'miss'}")
        print(f"  MLP ordered:  P={p_ord:.3f} R={r_ord:.3f} {'BEAT' if b_ord else 'miss'}")
        print(f"  DANN:         P={p_dann:.3f} R={r_dann:.3f} {'BEAT' if b_dann else 'miss'}")

        per_site_results['MLP'][name]=(p_mlp,r_mlp,b_mlp)
        per_site_results['MLP_ord'][name]=(p_ord,r_ord,b_ord)
        per_site_results['DANN'][name]=(p_dann,r_dann,b_dann)

    # ── LODO evaluation ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("LODO CROSS-SITE EVALUATION")
    print("="*60)

    for test_name in available:
        train_names=[r for r in available if r!=test_name]
        test_df =all_clips[test_name].copy().reset_index(drop=True)
        train_df=pd.concat([all_clips[r] for r in train_names],ignore_index=True)
        y_tr=train_df['clip_label'].values.astype(int)
        y_te=test_df['clip_label'].values.astype(int)
        site_tr=train_df['recorder'].map(site_map).values.astype(int)
        pos_weight=(y_tr==0).sum()/max(y_tr.sum(),1)

        if len(np.unique(y_te))<2: continue

        print(f"\nTest: {test_name}")
        X_tr_std,X_tr_ord,X_te_std,X_te_ord=get_feature_matrices(train_df,test_df)
        n_epochs=N_EPOCHS_LARGE

        mlp=StandardMLP(X_tr_std.shape[1])
        probs_mlp,_=train_mlp(mlp,X_tr_std,y_tr,X_te_std,n_epochs,pos_weight)

        mlp_ord=StandardMLP(X_tr_ord.shape[1],hidden1=128,hidden2=64)
        probs_ord,_=train_mlp(mlp_ord,X_tr_ord,y_tr,X_te_ord,n_epochs,pos_weight)

        dann=DANNModel(X_tr_std.shape[1],n_sites=n_sites-1)
        probs_dann,_=train_dann(dann,X_tr_std,y_tr,site_tr,X_te_std,n_epochs,pos_weight,n_sites-1)

        p_mlp,r_mlp,b_mlp=best_threshold(y_te,probs_mlp)
        p_ord,r_ord,b_ord=best_threshold(y_te,probs_ord)
        p_dann,r_dann,b_dann=best_threshold(y_te,probs_dann)

        print(f"  MLP standard: P={p_mlp:.3f} R={r_mlp:.3f} {'BEAT' if b_mlp else 'miss'}")
        print(f"  MLP ordered:  P={p_ord:.3f} R={r_ord:.3f} {'BEAT' if b_ord else 'miss'}")
        print(f"  DANN:         P={p_dann:.3f} R={r_dann:.3f} {'BEAT' if b_dann else 'miss'}")

        lodo_results['MLP'][test_name]=(p_mlp,r_mlp,b_mlp)
        lodo_results['MLP_ord'][test_name]=(p_ord,r_ord,b_ord)
        lodo_results['DANN'][test_name]=(p_dann,r_dann,b_dann)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n\n"+"="*70)
    print("SUMMARY — PER-SITE CV PRECISION")
    print("="*70)
    print(f"{'Model':<14} {'AM4':>7} {'AM2':>7} {'AM5':>7} {'AM6':>7} {'AM1':>7} {'Beat':>5}")
    print("-"*70)
    for m in ['MLP','MLP_ord','DANN']:
        vals=[per_site_results[m].get(n,(0,0,False))[0] for n in ['AM4','AM2','AM5','AM6','AM1']]
        nb=sum(1 for n in available if per_site_results[m].get(n,(0,0,False))[2])
        print(f"{m:<14} "+" ".join(f"{v:>7.3f}" for v in vals)+f" {nb:>4}/5")

    print("\n"+"="*70)
    print("SUMMARY — LODO CROSS-SITE PRECISION")
    print("="*70)
    print(f"{'Model':<14} {'AM4':>7} {'AM2':>7} {'AM5':>7} {'AM6':>7} {'AM1':>7} {'Beat':>5}")
    print("-"*70)
    for m in ['MLP','MLP_ord','DANN']:
        vals=[lodo_results[m].get(n,(0,0,False))[0] for n in ['AM4','AM2','AM5','AM6','AM1']]
        nb=sum(1 for n in available if lodo_results[m].get(n,(0,0,False))[2])
        print(f"{m:<14} "+" ".join(f"{v:>7.3f}" for v in vals)+f" {nb:>4}/5")

    print("\n"+"="*70)
    print("KEY QUESTION: does DANN improve cross-site over standard MLP?")
    dann_avg=np.mean([lodo_results['DANN'].get(n,(0,0,False))[0] for n in available])
    mlp_avg =np.mean([lodo_results['MLP'].get(n,(0,0,False))[0] for n in available])
    print(f"DANN avg cross-site P: {dann_avg:.3f}")
    print(f"MLP  avg cross-site P: {mlp_avg:.3f}")
    print(f"DANN improvement: {dann_avg-mlp_avg:+.3f}")
    print("\nDone.")
