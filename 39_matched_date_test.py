import os,numpy as np,pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve,precision_recall_fscore_support

BASE_DIR=r"C:\Users\BitaMassoudi\KasmirWorldProjects\toucanet-option-b"
SEED=42
RECORDERS=[
    ("AM4","am4_full_feature_matrix.csv","am4_full_emb.npy","am4_full_labels.npy",0.856,0.906),
    ("AM2","am2_feature_matrix.csv","am2_time_controlled_emb.npy","am2_time_controlled_labels.npy",0.779,0.781),
    ("AM5","am5_feature_matrix.csv","am5_time_controlled_emb.npy","am5_time_controlled_labels.npy",0.707,0.932),
    ("AM6","am6_feature_matrix.csv","am6_time_controlled_emb.npy","am6_time_controlled_labels.npy",0.703,0.744),
    ("AM1","am1_feature_matrix.csv","am1_full_emb.npy","am1_full_labels.npy",0.779,0.898),
]
SPEC=[f"mfcc_mean_{i}" for i in range(13)]+[f"mfcc_std_{i}" for i in range(13)]+["centroid_mean","centroid_std","rolloff_mean","rolloff_std","bandwidth_mean","bandwidth_std","zcr_mean","zcr_std","rms_mean","rms_std","silence_fraction","spec_entropy_mean","spec_entropy_std","temporal_entropy","onset_count"]

def build(feat_df,emb,labels):
    rows=[]
    for cn in feat_df["clip_name"].unique():
        mask=feat_df["clip_name"].values==cn
        cl=feat_df["clip_label"].values[mask][0]
        hour=feat_df["clip_hour"].values[mask][0]
        date=cn.split("_")[3]
        ce=emb[mask]
        row={"clip_name":cn,"clip_label":cl,"clip_hour":hour,"date":date}
        for k in SPEC:
            if k in feat_df.columns: row[k]=feat_df.loc[mask,k].values[0]
        for i,v in enumerate(ce.mean(0)): row[f"em_{i}"]=v
        for i,v in enumerate(ce.std(0)):  row[f"es_{i}"]=v
        for i,v in enumerate(ce.max(0)):  row[f"ex_{i}"]=v
        rows.append(row)
    return pd.DataFrame(rows)

def run_cv(X,y):
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
    probs=np.zeros(len(y),dtype=np.float32)
    for tr,va in cv.split(X,y):
        n_pos=y[tr].sum(); n_neg=(y[tr]==0).sum()
        w=np.where(y[tr]==1,n_neg/max(n_pos,1),1.0)
        gb=GradientBoostingClassifier(n_estimators=100,max_depth=3,learning_rate=0.1,random_state=SEED,subsample=0.8)
        gb.fit(X[tr],y[tr],sample_weight=w)
        probs[va]=gb.predict_proba(X[va])[:,1]
    prec,rec,thresh=precision_recall_curve(y,probs)
    valid=np.where((prec[:-1]>=0.70)&(rec[:-1]>=0.70))[0]
    if len(valid)>0:
        bi=valid[np.argmax(prec[valid]+rec[valid])]; bt=float(thresh[bi])
        preds=(probs>=bt).astype(int)
        p,r,_,_=precision_recall_fscore_support(y,preds,average="binary",zero_division=0)
        return p,r,True
    bi=np.argmax(prec[:-1]+rec[:-1]); bt=float(thresh[bi])
    preds=(probs>=bt).astype(int)
    p,r,_,_=precision_recall_fscore_support(y,preds,average="binary",zero_division=0)
    return p,r,False

print("DATE-BALANCED MATCHED SAMPLING TEST")
print("="*60)
summary=[]
for name,feat_f,emb_f,lab_f,best_p,best_r in RECORDERS:
    feat_path=os.path.join(BASE_DIR,feat_f)
    emb_path=os.path.join(BASE_DIR,emb_f)
    if not os.path.exists(feat_path) or not os.path.exists(emb_path):
        print(f"{name}: files not found"); continue
    feat_df=pd.read_csv(feat_path)
    emb=np.load(emb_path); labels=np.load(os.path.join(BASE_DIR,lab_f))
    n=min(len(feat_df),len(emb))
    feat_df=feat_df.iloc[:n].reset_index(drop=True); emb=emb[:n]; labels=labels[:n]
    clip_df=build(feat_df,emb,labels)
    feat_cols=[c for c in clip_df.columns if c not in {"clip_name","clip_label","clip_hour","date"}]
    X_full=clip_df[feat_cols].values.astype(np.float32)
    y_full=clip_df["clip_label"].values.astype(int)
    scaler=StandardScaler(); X_full_s=scaler.fit_transform(X_full)
    p_full,r_full,beat_full=run_cv(X_full_s,y_full)
    print(f"\n{name} unmatched: P={p_full:.3f} R={r_full:.3f} {'BEAT' if beat_full else 'miss'}")
    rng=np.random.default_rng(SEED); keep=[]
    for date,g in clip_df.groupby("date"):
        pos=g.index[g["clip_label"]==1].to_numpy(); neg=g.index[g["clip_label"]==0].to_numpy()
        k=min(len(pos),len(neg))
        if k==0: continue
        keep+=list(rng.choice(pos,k,replace=False))+list(rng.choice(neg,k,replace=False))
    matched_df=clip_df.loc[keep].reset_index(drop=True)
    X_m=matched_df[feat_cols].values.astype(np.float32)
    y_m=matched_df["clip_label"].values.astype(int)
    if len(np.unique(y_m))<2 or y_m.sum()<5:
        print(f"{name} matched: insufficient data"); continue
    scaler_m=StandardScaler(); X_m_s=scaler_m.fit_transform(X_m)
    p_m,r_m,beat_m=run_cv(X_m_s,y_m)
    diff_p=p_m-p_full
    print(f"{name} matched:   P={p_m:.3f} R={r_m:.3f} {'BEAT' if beat_m else 'miss'} change P{diff_p:+.3f}")
    verdict="STABLE" if abs(diff_p)<0.05 else ("DATE CONFOUND LIKELY" if diff_p<-0.10 else "SLIGHT DROP")
    print(f"{name} verdict: {verdict}")
    summary.append({"name":name,"unmatched":p_full,"matched":p_m,"change":diff_p,"verdict":verdict,"beat":beat_m})

print("\n"+"="*60)
print("SUMMARY")
print("="*60)
for r in summary:
    print(f"{r['name']}: unmatched={r['unmatched']:.3f} matched={r['matched']:.3f} change={r['change']:+.3f} {r['verdict']}")
n_stable=sum(1 for r in summary if abs(r["change"])<0.05)
n_drop=sum(1 for r in summary if r["change"]<-0.05)
print(f"\nStable: {n_stable}/{len(summary)}  Drops: {n_drop}/{len(summary)}")
if n_drop==0: print("CONCLUSION: No date confound. Signal appears genuine.")
elif n_drop<=2: print("CONCLUSION: Partial date confound at some recorders.")
else: print("CONCLUSION: Widespread date confound. Results may be inflated.")
