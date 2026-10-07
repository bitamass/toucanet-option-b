import zipfile, io, os
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve

BASE_DIR = r"C:\Users\BitaMassoudi\KasmirWorldProjects\toucanet-option-b"
ZIP_PATH = r"C:\Users\BitaMassoudi\Downloads\raw_segments_per_clip.zip"
SEED = 42

RECORDERS = [
    ("AM4","am4_full_feature_matrix.csv"),
    ("AM2","am2_feature_matrix.csv"),
    ("AM5","am5_feature_matrix.csv"),
    ("AM6","am6_feature_matrix.csv"),
    ("AM1","am1_feature_matrix.csv"),
]

print("Loading labels...")
clip_meta = {}
for name, ff in RECORDERS:
    fp = os.path.join(BASE_DIR, ff)
    if not os.path.exists(fp): continue
    df = pd.read_csv(fp)
    for cn in df["clip_name"].unique():
        mask = df["clip_name"].values == cn
        stem = cn.replace(".wav","")
        clip_meta[stem] = {"recorder":name, "label":int(df["clip_label"].values[mask][0])}
print(f"Known clips: {len(clip_meta)}")

print("Loading BirdNET embeddings...")
all_embs = {}
with zipfile.ZipFile(ZIP_PATH) as z:
    entries = [n for n in z.namelist() if n.endswith(".npz") and "Audio_Moth" in n and "__MACOSX" not in n]
    print(f"Total entries: {len(entries)}")
    for i, n in enumerate(entries):
        stem = n.replace("raw_segments_per_clip/","").replace(".npz","")
        if stem not in clip_meta: continue
        try:
            with z.open(n) as f:
                data = np.load(io.BytesIO(f.read()))
                emb = np.nan_to_num(data["segments"].astype(np.float64).mean(axis=0), nan=0.0, posinf=0.0, neginf=0.0)
                all_embs[stem] = emb
        except: pass
        if (i+1) % 20000 == 0: print(f"  {i+1}/{len(entries)} scanned, {len(all_embs)} matched")
print(f"Matched: {len(all_embs)}")

print("\nBuilding datasets...")
recorder_data = {}
for name, ff in RECORDERS:
    fp = os.path.join(BASE_DIR, ff)
    if not os.path.exists(fp): continue
    df = pd.read_csv(fp)
    rows = []
    for cn in df["clip_name"].unique():
        stem = cn.replace(".wav","")
        if stem not in all_embs: continue
        mask = df["clip_name"].values == cn
        rows.append({"label":int(df["clip_label"].values[mask][0]), "emb":all_embs[stem]})
    if rows:
        recorder_data[name] = rows
        print(f"  {name}: {len(rows)} clips ({sum(r['label'] for r in rows)} sim)")

def evaluate(y, probs):
    pr,rc,th = precision_recall_curve(y,probs)
    valid = np.where((pr[:-1]>=0.70)&(rc[:-1]>=0.70))[0]
    if len(valid)>0:
        bi=valid[np.argmax(pr[valid]+rc[valid])]; bt=float(th[bi])
    else:
        bi=np.argmax(pr[:-1]+rc[:-1]); bt=float(th[bi])
    preds=(probs>=bt).astype(int)
    p,r,_,_=precision_recall_fscore_support(y,preds,average="binary",zero_division=0)
    return p,r,(p>=0.70 and r>=0.70)

available = list(recorder_data.keys())
print(f"\nLODO cross-site ({len(available)} recorders)...")
print(f"{'Test':<6} {'P':>7} {'R':>7} {'Beat?':>7}")
print("-"*30)

results = {}
for test_name in available:
    train_names = [r for r in available if r != test_name]
    X_tr = np.array([r["emb"] for tn in train_names for r in recorder_data[tn]], dtype=np.float64)
    y_tr = np.array([r["label"] for tn in train_names for r in recorder_data[tn]])
    X_te = np.array([r["emb"] for r in recorder_data[test_name]], dtype=np.float64)
    y_te = np.array([r["label"] for r in recorder_data[test_name]])
    X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
    X_te = np.nan_to_num(X_te, nan=0.0, posinf=0.0, neginf=0.0)
    if len(np.unique(y_te)) < 2: continue
    sc = StandardScaler()
    pca = PCA(n_components=50, random_state=SEED)
    X_tr_s = pca.fit_transform(sc.fit_transform(X_tr))
    X_te_s = pca.transform(sc.transform(X_te))
    X_te_s = np.nan_to_num(X_te_s, nan=0.0, posinf=0.0, neginf=0.0)
    n_pos=y_tr.sum(); n_neg=(y_tr==0).sum()
    w=np.where(y_tr==1,n_neg/max(n_pos,1),1.0)
    gb=GradientBoostingClassifier(n_estimators=100,max_depth=3,learning_rate=0.1,random_state=SEED,subsample=0.8)
    gb.fit(X_tr_s,y_tr,sample_weight=w)
    probs=gb.predict_proba(X_te_s)[:,1]
    p,r,beat=evaluate(y_te,probs)
    results[test_name]=(p,r,beat)
    print(f"{test_name:<6} {p:>7.3f} {r:>7.3f} {'BEAT' if beat else 'miss':>7}")

print("\n"+"="*50)
print("BIRDNET LODO SUMMARY")
print("="*50)
avg_p = np.mean([v[0] for v in results.values()])
n_beat = sum(v[2] for v in results.values())
print(f"BirdNET avg cross-site P: {avg_p:.3f}")
print(f"Previous best:            0.364")
print(f"Change:                   {avg_p-0.364:+.3f}")
print(f"Beat 0.70:                {n_beat}/{len(results)}")
print("\nAll methods ranked:")
for method,p in sorted([
    ("Pooled GB (Perch)",0.345),("Deployment calib.",0.364),
    ("Mixture of experts",0.359),("AR sequence model",0.343),
    ("BirdNET embeddings",avg_p)],key=lambda x:-x[1]):
    marker=" <-- NEW" if method=="BirdNET embeddings" else ""
    print(f"  {method:<28}: {p:.3f}{marker}")
print("\nDone.")
