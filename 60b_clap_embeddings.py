import os, sys, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, StratifiedKFold

BASE_DIR = r"C:\Users\BitaMassoudi\KasmirWorldProjects\toucanet-option-b"
SEED = 42
RECORDERS = [("AM4","am4_full_feature_matrix.csv"),("AM2","am2_feature_matrix.csv"),("AM5","am5_feature_matrix.csv"),("AM6","am6_feature_matrix.csv"),("AM1","am1_feature_matrix.csv")]

print("Loading clip names...")
all_clip_names, all_site_labels = [], []
for name,ff in RECORDERS:
    fp = os.path.join(BASE_DIR, ff)
    if not os.path.exists(fp): continue
    clips = pd.read_csv(fp)["clip_name"].unique().tolist()
    all_clip_names.extend(clips)
    all_site_labels.extend([name]*len(clips))
    print(f"  {name}: {len(clips)} clips")

print(f"\nSearching for WAV files in {BASE_DIR}...")
wav_files = {}
for root, dirs, files in os.walk(BASE_DIR):
    for f in files:
        if f.lower().endswith(".wav"):
            base = f.replace(" (1)","").replace(" (2)","").replace(" (3)","")
            wav_files[base] = os.path.join(root, f)

print(f"Found {len(wav_files)} WAV files on disk")
found = [(cn, wav_files[cn]) for cn in all_clip_names if cn in wav_files]
print(f"Matched {len(found)} clips to feature matrix entries")

by_site = {}
for cn, wp in found:
    site = all_site_labels[all_clip_names.index(cn)]
    by_site.setdefault(site, []).append((cn, wp))
for site, clips in sorted(by_site.items()):
    print(f"  {site}: {len(clips)} matched clips")

MAX_PER_SITE = 500
sampled = []
sampled_sites = []
for site, clips in by_site.items():
    chosen = clips[:MAX_PER_SITE]
    sampled.extend(chosen)
    sampled_sites.extend([site]*len(chosen))

print(f"\nUsing {len(sampled)} clips ({MAX_PER_SITE} per site max) for CLAP extraction")
print("Loading CLAP model...")
from msclap import CLAP
model = CLAP(version="2023", use_cuda=False)
print("CLAP loaded")

CACHE = os.path.join(BASE_DIR, "clap_diag_embs.npy")
CACHE_SITES = os.path.join(BASE_DIR, "clap_diag_sites.npy")

if os.path.exists(CACHE):
    print(f"Loading cached embeddings from {CACHE}")
    clap_embs = np.load(CACHE)
    clap_sites = np.load(CACHE_SITES, allow_pickle=True).tolist()
    print(f"  {len(clap_embs)} embeddings loaded")
else:
    print(f"Extracting CLAP embeddings...")
    clap_embs, clap_sites = [], []
    failed = 0
    for i,(cn,wp) in enumerate(sampled):
        try:
            emb = model.get_audio_embeddings([wp])
            clap_embs.append(emb[0])
            clap_sites.append(sampled_sites[i])
        except:
            failed += 1
        if (i+1)%50==0:
            print(f"  {i+1}/{len(sampled)} done ({failed} failed)")
            np.save(CACHE, np.array(clap_embs))
            np.save(CACHE_SITES, np.array(clap_sites, dtype=object))
    clap_embs = np.array(clap_embs)
    np.save(CACHE, clap_embs)
    np.save(CACHE_SITES, np.array(clap_sites, dtype=object))
    print(f"Done: {len(clap_embs)} succeeded, {failed} failed")

X = clap_embs
sites = sorted(set(clap_sites))
y = np.array([sites.index(s) for s in clap_sites])
print(f"\nEmbedding shape: {X.shape}")
print(f"Sites: {sites}")
for s in sites:
    print(f"  {s}: {(y==sites.index(s)).sum()} clips")

sc = StandardScaler(); Xs = sc.fit_transform(X)
Xp = PCA(n_components=min(50, X.shape[1]-1), random_state=SEED).fit_transform(Xs)
min_per_site = min(np.bincount(y))
n_splits = min(5, min_per_site)
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
scores = cross_val_score(LogisticRegression(max_iter=500, random_state=SEED), Xp, y, cv=cv, scoring="accuracy")

print(f"\n{'='*50}")
print(f"CLAP site prediction accuracy:  {scores.mean():.3f} +/- {scores.std():.3f}")
print(f"Perch site prediction accuracy: 0.992")
print(f"Random baseline:                {1/len(sites):.3f}")
print(f"{'='*50}")
if scores.mean() < 0.80:
    print("RESULT: CLAP much less site-specific than Perch")
    print("-> Strong case for switching to CLAP embeddings")
elif scores.mean() < 0.95:
    print(f"RESULT: CLAP moderately less site-specific ({0.992-scores.mean():.3f} improvement)")
    print("-> CLAP worth testing for cross-site LODO")
else:
    print("RESULT: CLAP equally site-specific as Perch")
    print("-> Representation bottleneck not solved by CLAP")
print("Done.")
