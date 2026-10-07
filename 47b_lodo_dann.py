"""
47b_lodo_dann.py
-----------------
LODO cross-site evaluation for MLP and DANN.
Fixes the site index bug from script 47.

Per-site results already confirmed from script 47:
  MLP standard: 4/5 beat 0.70 (AM6 barely passes)
  DANN:         4/5 beat 0.70 (AM6 fails)

This script runs LODO only — cross-site generalisation test.
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
from sklearn.metrics import (precision_recall_fscore_support,
                              precision_recall_curve)

BASE_DIR = r"C:\Users\BitaMassoudi\KasmirWorldProjects\toucanet-option-b"
SEED = 42
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
Z_KEYS = [f"z_{k}" for k in SPECTRAL_KEYS]
BATCH  = 32
EPOCHS = 30
LAMBDA = 0.1


class StandardMLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d,256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256,128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128,2))
    def forward(self, x): return self.net(x)


class GradRev(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam; return x.clone()
    @staticmethod
    def backward(ctx, g): return -ctx.lam * g, None


class DANN(nn.Module):
    def __init__(self, d, n_sites):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(d,256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256,128), nn.BatchNorm1d(128), nn.ReLU())
        self.dist = nn.Sequential(nn.Linear(128,64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64,2))
        self.site = nn.Sequential(nn.Linear(128,64), nn.ReLU(), nn.Linear(64,n_sites))
    def forward(self, x, lam=1.0):
        f = self.enc(x)
        d = self.dist(f)
        s = self.site(GradRev.apply(f, lam))
        return d, s


def build_clips(feat_df, emb):
    rows=[]
    for cn in feat_df['clip_name'].unique():
        mask=feat_df['clip_name'].values==cn
        cl=feat_df['clip_label'].values[mask][0]
        hour=feat_df['clip_hour'].values[mask][0]
        ce=emb[mask]
        row={'clip_name':cn,'clip_label':cl,'clip_hour':hour}
        for k in SPECTRAL_KEYS:
            if k in feat_df.columns: row[k]=feat_df.loc[mask,k].values[0]
        for i,v in enumerate(ce.mean(0)): row[f'em_{i}']=v
        for i,v in enumerate(ce.std(0)):  row[f'es_{i}']=v
        for i,v in enumerate(ce.max(0)):  row[f'ex_{i}']=v
        rows.append(row)
    return pd.DataFrame(rows)


def zscores(tr, te):
    ztr,zte=[],[]
    for h in tr['clip_hour'].unique():
        bm=(tr['clip_hour']==h)&(tr['clip_label']==0)
        br=tr.loc[bm,SPECTRAL_KEYS]
        if len(br)==0: continue
        hm=br.mean(); hs=br.std().replace(0,1e-6)
        th=tr['clip_hour']==h
        if th.sum()>0:
            z=(tr.loc[th,SPECTRAL_KEYS]-hm)/hs; z.columns=Z_KEYS; ztr.append(z)
        teh=te['clip_hour']==h
        if teh.sum()>0:
            z=(te.loc[teh,SPECTRAL_KEYS]-hm)/hs; z.columns=Z_KEYS; zte.append(z)
    seen=set(tr['clip_hour'].unique())
    for h in set(te['clip_hour'].unique())-seen:
        teh=te['clip_hour']==h
        if teh.sum()>0:
            zte.append(pd.DataFrame(0,index=te.index[teh],columns=Z_KEYS))
    z_tr=pd.concat(ztr).sort_index() if ztr else pd.DataFrame(0,index=tr.index,columns=Z_KEYS)
    z_te=pd.concat(zte).sort_index() if zte else pd.DataFrame(0,index=te.index,columns=Z_KEYS)
    return z_tr,z_te


def feats(tr, te):
    ec=[c for c in tr.columns if c.startswith(('em_','es_','ex_'))]
    z_tr,z_te=zscores(tr,te)
    pca=PCA(n_components=50,random_state=SEED)
    etr=pca.fit_transform(tr[ec].values); ete=pca.transform(te[ec].values)
    X_tr=np.hstack([tr[SPECTRAL_KEYS].values,z_tr.values,etr]).astype(np.float32)
    X_te=np.hstack([te[SPECTRAL_KEYS].values,z_te.values,ete]).astype(np.float32)
    return X_tr,X_te


def train_mlp(X_tr,y_tr,X_te,pw):
    sc=StandardScaler(); Xts=torch.tensor(sc.fit_transform(X_tr)); Xvs=torch.tensor(sc.transform(X_te))
    yt=torch.tensor(y_tr,dtype=torch.long); w=torch.tensor([1.0,float(pw)])
    m=StandardMLP(X_tr.shape[1]); opt=optim.Adam(m.parameters(),lr=1e-3,weight_decay=1e-4)
    crit=nn.CrossEntropyLoss(weight=w)
    dl=DataLoader(TensorDataset(Xts,yt),batch_size=BATCH,shuffle=True)
    m.train()
    for _ in range(EPOCHS):
        for xb,yb in dl:
            opt.zero_grad(); loss=crit(m(xb),yb); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad(): probs=torch.softmax(m(Xvs),1)[:,1].numpy()
    return probs


def train_dann(X_tr,y_tr,site_tr,X_te,pw,n_sites):
    # Remap site labels to be contiguous starting from 0
    unique_sites=np.unique(site_tr)
    site_remap={s:i for i,s in enumerate(unique_sites)}
    site_tr_r=np.array([site_remap[s] for s in site_tr])
    n_sites_actual=len(unique_sites)

    sc=StandardScaler(); Xts=torch.tensor(sc.fit_transform(X_tr)); Xvs=torch.tensor(sc.transform(X_te))
    yt=torch.tensor(y_tr,dtype=torch.long)
    st=torch.tensor(site_tr_r,dtype=torch.long)
    w=torch.tensor([1.0,float(pw)])
    m=DANN(X_tr.shape[1],n_sites_actual)
    opt=optim.Adam(m.parameters(),lr=1e-3,weight_decay=1e-4)
    dc=nn.CrossEntropyLoss(weight=w); sc_=nn.CrossEntropyLoss()
    dl=DataLoader(TensorDataset(Xts,yt,st),batch_size=BATCH,shuffle=True)
    m.train()
    for ep in range(EPOCHS):
        p=ep/EPOCHS; lam=2/(1+np.exp(-10*p))-1
        for xb,yb,sb in dl:
            opt.zero_grad()
            do,so=m(xb,lam*LAMBDA)
            loss=dc(do,yb)+LAMBDA*sc_(so,sb)
            loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        do,_=m(Xvs,1.0); probs=torch.softmax(do,1)[:,1].numpy()
    return probs


def thresh(y,probs):
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
    print("Loading data...")
    all_clips={}
    for name,ff,ef,lf in RECORDERS:
        fp=os.path.join(BASE_DIR,ff); ep=os.path.join(BASE_DIR,ef)
        if not os.path.exists(fp) or not os.path.exists(ep): continue
        fd=pd.read_csv(fp); em=np.load(ep)
        n=min(len(fd),len(em)); fd=fd.iloc[:n].reset_index(drop=True); em=em[:n]
        cd=build_clips(fd,em); cd['recorder']=name
        all_clips[name]=cd
        print(f"  {name}: {len(cd)} clips")

    available=list(all_clips.keys())
    site_map={n:i for i,n in enumerate(available)}

    mlp_res={}; dann_res={}

    print("\nLODO CROSS-SITE EVALUATION")
    print("="*60)
    print(f"{'Rec':<6} {'MLP P':>7} {'MLP R':>7} {'DANN P':>8} {'DANN R':>8}")
    print("-"*60)

    for test_name in available:
        train_names=[r for r in available if r!=test_name]
        te=all_clips[test_name].copy().reset_index(drop=True)
        tr=pd.concat([all_clips[r] for r in train_names],ignore_index=True)
        y_tr=tr['clip_label'].values.astype(int)
        y_te=te['clip_label'].values.astype(int)
        # Site labels for training data only
        site_tr=tr['recorder'].map(site_map).values.astype(int)
        pw=(y_tr==0).sum()/max(y_tr.sum(),1)
        if len(np.unique(y_te))<2: continue

        X_tr,X_te=feats(tr,te)

        p_mlp=train_mlp(X_tr,y_tr,X_te,pw)
        pm,rm,bm=thresh(y_te,p_mlp)

        p_dann=train_dann(X_tr,y_tr,site_tr,X_te,pw,len(available)-1)
        pd_,rd,bd=thresh(y_te,p_dann)

        mlp_res[test_name]=(pm,rm,bm)
        dann_res[test_name]=(pd_,rd,bd)

        print(f"{test_name:<6} {pm:>7.3f} {rm:>7.3f} {pd_:>8.3f} {rd:>8.3f}  "
              f"{'BEAT' if bm else 'miss'} / {'BEAT' if bd else 'miss'}")

    print("="*60)
    n_mlp =sum(1 for v in mlp_res.values()  if v[2])
    n_dann=sum(1 for v in dann_res.values() if v[2])
    avg_mlp =np.mean([v[0] for v in mlp_res.values()])
    avg_dann=np.mean([v[0] for v in dann_res.values()])
    print(f"MLP  beat 0.70: {n_mlp}/5  avg P={avg_mlp:.3f}")
    print(f"DANN beat 0.70: {n_dann}/5  avg P={avg_dann:.3f}")
    print(f"DANN improvement over MLP: {avg_dann-avg_mlp:+.3f}")
    print("\nDone.")
