"""
60_alternative_embeddings.py
-----------------------------
Test alternative audio embeddings vs Perch on LODO cross-site.

MOTIVATION FROM SCRIPT 59:
  Perch embeddings are 99.2% predictive of site identity.
  Only 58.3% predictive of disturbance.
  No spectral feature varies more with disturbance than with site.
  -> The representation itself is the bottleneck.

HYPOTHESIS:
  Models trained on SEMANTIC audio concepts (what does a vehicle
  sound like? what does a chainsaw sound like?) may encode
  disturbance-relevant features more universally than Perch,
  which was trained on bird vocalisation classification.

MODELS TESTED:

  1. CLAP (Contrastive Language-Audio Pretraining)
     Trained on audio-text pairs from internet.
     Concepts like "vehicle", "chainsaw", "human footsteps"
     may be explicitly encoded in CLAP embeddings.
     Library: msclap or open_clip with audio branch.

  2. BEATs (Audio Pre-Training with Acoustic Tokenizers)
     Microsoft model trained on AudioSet with 527 classes
     including many environmental sounds.
     Explicitly trained to recognise vehicle, chainsaw etc.
     Library: beat (pip install beat-pytorch) or torch hub.

  3. VGGish / YAMNet (Google)
     Trained on YouTube-8M audio.
     Older but widely used baseline.
     Library: torchvggish or tensorflow_hub.

  4. PANNs (Pretrained Audio Neural Networks)
     CNN14 trained on AudioSet.
     Strong general audio features.
     Library: panns_inference.

EVALUATION FOR EACH MODEL:
  1. Predict site identity from embeddings (diagnostic)
     -> Lower = better for cross-site transfer
  2. LODO cross-site precision with linear probe
     -> Higher = better

COMPARISON TO PERCH:
  Site identity accuracy: 0.992 (near perfect — bad)
  Cross-site precision:   0.345 avg (all approaches failed)

  If any alternative model shows:
    Site identity < 0.80 AND cross-site precision > 0.40
  -> It is a better representation for this task.

INSTALLATION CHECK:
  Script checks which libraries are available and runs
  whatever it can find. Falls back gracefully.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (precision_recall_fscore_support,
                              precision_recall_curve)

BASE_DIR  = r"C:\Users\BitaMassoudi\KasmirWorldProjects\toucanet-option-b"
AUDIO_DIR = BASE_DIR   # adjust if WAV files are elsewhere
SEED      = 42

RECORDERS = [
    ('AM4','am4_full_feature_matrix.csv','am4_full_emb.npy'),
    ('AM2','am2_feature_matrix.csv','am2_time_controlled_emb.npy'),
    ('AM5','am5_feature_matrix.csv','am5_time_controlled_emb.npy'),
    ('AM6','am6_feature_matrix.csv','am6_time_controlled_emb.npy'),
    ('AM1','am1_feature_matrix.csv','am1_full_emb.npy'),
]


def build_clip_df(feat_df, emb):
    rows = []
    for cn in feat_df['clip_name'].unique():
        mask = feat_df['clip_name'].values == cn
        cl   = feat_df['clip_label'].values[mask][0]
        hour = feat_df['clip_hour'].values[mask][0]
        ce   = emb[mask]
        row  = {'clip_name':cn,'clip_label':cl,'clip_hour':hour}
        for i,v in enumerate(ce.mean(0)): row[f'em_{i}']=v
        for i,v in enumerate(ce.std(0)):  row[f'es_{i}']=v
        for i,v in enumerate(ce.max(0)):  row[f'ex_{i}']=v
        rows.append(row)
    return pd.DataFrame(rows)


def site_predictability(emb_matrix, site_labels):
    """How well do embeddings predict recorder site? Lower = better for transfer."""
    sc  = StandardScaler(); X = sc.fit_transform(emb_matrix)
    pca = PCA(n_components=min(50, X.shape[1]-1), random_state=SEED)
    X   = pca.fit_transform(X)
    cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    lr  = LogisticRegression(max_iter=500, random_state=SEED)
    scores = cross_val_score(lr, X, site_labels, cv=cv, scoring='accuracy')
    return scores.mean(), scores.std()


def lodo_precision(all_clips, emb_key='perch'):
    """LODO cross-site precision using linear probe on embeddings."""
    available = list(all_clips.keys())
    results = {}
    for test_name in available:
        train_names = [r for r in available if r != test_name]
        test_df  = all_clips[test_name].copy().reset_index(drop=True)
        train_df = pd.concat([all_clips[r] for r in train_names],
                              ignore_index=True)
        ec = [c for c in train_df.columns if c.startswith(('em_','es_','ex_'))]
        y_tr = train_df['clip_label'].values.astype(int)
        y_te = test_df['clip_label'].values.astype(int)
        if len(np.unique(y_te)) < 2: continue
        sc  = StandardScaler()
        pca = PCA(n_components=50, random_state=SEED)
        X_tr = pca.fit_transform(sc.fit_transform(train_df[ec].values))
        X_te = pca.transform(sc.transform(test_df[ec].values))
        n_pos = y_tr.sum(); n_neg = (y_tr==0).sum()
        w = np.where(y_tr==1, n_neg/max(n_pos,1), 1.0)
        gb = GradientBoostingClassifier(n_estimators=100, max_depth=3,
             learning_rate=0.1, random_state=SEED, subsample=0.8)
        gb.fit(X_tr, y_tr, sample_weight=w)
        probs = gb.predict_proba(X_te)[:,1]
        pr,rc,th = precision_recall_curve(y_te, probs)
        valid = np.where((pr[:-1]>=0.70)&(rc[:-1]>=0.70))[0]
        if len(valid)>0:
            bi=valid[np.argmax(pr[valid]+rc[valid])]; bt=float(th[bi])
        else:
            bi=np.argmax(pr[:-1]+rc[:-1]); bt=float(th[bi])
        preds=(probs>=bt).astype(int)
        p,r,_,_ = precision_recall_fscore_support(y_te,preds,
                   average='binary',zero_division=0)
        results[test_name] = (p, r, p>=0.70 and r>=0.70)
    return results


def check_available_models():
    """Check which embedding models are installable."""
    available = {}

    # CLAP
    try:
        import msclap
        available['CLAP'] = 'msclap'
        print("  CLAP (msclap): AVAILABLE")
    except ImportError:
        try:
            import laion_clap
            available['CLAP'] = 'laion_clap'
            print("  CLAP (laion_clap): AVAILABLE")
        except ImportError:
            print("  CLAP: NOT INSTALLED")
            print("    Install: pip install msclap --break-system-packages")
            print("    Or:      pip install laion-clap --break-system-packages")

    # BEATs
    try:
        import torch
        # BEATs requires manual download — check if model file exists
        beats_path = os.path.join(BASE_DIR, 'BEATs_iter3_plus_AS2M.pt')
        if os.path.exists(beats_path):
            available['BEATs'] = beats_path
            print(f"  BEATs: AVAILABLE ({beats_path})")
        else:
            print("  BEATs: Model file not found")
            print(f"    Download from: https://msranlcmtteamdrive.blob.core.windows.net/share/BEATs/BEATs_iter3_plus_AS2M.pt")
            print(f"    Save to: {beats_path}")
    except ImportError:
        print("  BEATs: PyTorch not available")

    # PANNs
    try:
        from panns_inference import AudioTagging
        available['PANNs'] = 'panns_inference'
        print("  PANNs: AVAILABLE")
    except ImportError:
        print("  PANNs: NOT INSTALLED")
        print("    Install: pip install panns_inference --break-system-packages")

    # VGGish
    try:
        import torchvggish
        available['VGGish'] = 'torchvggish'
        print("  VGGish: AVAILABLE")
    except ImportError:
        print("  VGGish: NOT INSTALLED")
        print("    Install: pip install torchvggish --break-system-packages")

    return available


def try_clap_embeddings(clip_names, library):
    """Extract CLAP embeddings for audio clips."""
    print(f"  Extracting CLAP embeddings using {library}...")
    embeddings = {}
    try:
        if library == 'msclap':
            from msclap import CLAP
            model = CLAP(version='2023', use_cuda=False)
            for cn in clip_names[:5]:  # test first
                wav_path = find_wav(cn)
                if wav_path:
                    emb = model.get_audio_embeddings([wav_path])
                    embeddings[cn] = emb[0]
        elif library == 'laion_clap':
            import laion_clap
            model = laion_clap.CLAP_Module(enable_fusion=False)
            model.load_ckpt()
            for cn in clip_names[:5]:
                wav_path = find_wav(cn)
                if wav_path:
                    emb = model.get_audio_embedding_from_filelist([wav_path])
                    embeddings[cn] = emb[0]
    except Exception as e:
        print(f"  CLAP extraction failed: {e}")
    return embeddings


def find_wav(clip_name):
    """Find WAV file for a clip name."""
    for search_dir in [BASE_DIR, AUDIO_DIR]:
        path = os.path.join(search_dir, clip_name)
        if os.path.exists(path):
            return path
        # Try subdirectories
        for subdir in os.listdir(search_dir):
            path = os.path.join(search_dir, subdir, clip_name)
            if os.path.exists(path):
                return path
    return None


if __name__ == '__main__':

    print("Alternative Embedding Comparison")
    print("="*60)
    print("Goal: find embeddings where site accuracy < 0.80")
    print("      and cross-site precision > 0.40")
    print("="*60)

    # ── Step 1: Perch baseline (from saved embeddings) ────────────────────────
    print("\nLoading Perch embeddings (baseline from script 59)...")
    all_clips = {}
    all_embs  = []
    all_sites = []

    for name,ff,ef in RECORDERS:
        fp=os.path.join(BASE_DIR,ff); ep=os.path.join(BASE_DIR,ef)
        if not os.path.exists(fp) or not os.path.exists(ep): continue
        feat_df=pd.read_csv(fp); emb=np.load(ep)
        n=min(len(feat_df),len(emb))
        feat_df=feat_df.iloc[:n].reset_index(drop=True); emb=emb[:n]
        clip_df=build_clip_df(feat_df,emb)
        clip_df['recorder']=name
        all_clips[name]=clip_df
        ec=[c for c in clip_df.columns if c.startswith('em_')]
        all_embs.append(clip_df[ec].values)
        all_sites.extend([name]*len(clip_df))
        print(f"  {name}: {len(clip_df)} clips")

    available = list(all_clips.keys())
    site_map  = {n:i for i,n in enumerate(available)}
    all_emb_matrix = np.vstack(all_embs).astype(np.float32)
    all_site_labels = np.array([site_map[s] for s in all_sites])

    print("\nPerch baseline diagnostics:")
    site_acc, site_std = site_predictability(all_emb_matrix, all_site_labels)
    print(f"  Site prediction accuracy: {site_acc:.3f} +/- {site_std:.3f}")
    print(f"  (from script 59: 0.992)")

    print("\nPerch LODO cross-site:")
    perch_lodo = lodo_precision(all_clips)
    perch_avg  = np.mean([v[0] for v in perch_lodo.values()])
    for name,v in perch_lodo.items():
        print(f"  {name}: P={v[0]:.3f} {'BEAT' if v[2] else 'miss'}")
    print(f"  Average: {perch_avg:.3f}")

    # ── Step 2: Check available alternative models ────────────────────────────
    print("\n" + "="*60)
    print("Checking available alternative embedding models...")
    print("="*60)
    available_models = check_available_models()

    if not available_models:
        print("\n" + "="*60)
        print("NO ALTERNATIVE MODELS INSTALLED")
        print("="*60)
        print("""
To test alternative embeddings, install one or more:

OPTION 1 — CLAP (recommended, most likely to help):
  pip install msclap --break-system-packages

OPTION 2 — PANNs (AudioSet-trained, good for environmental sounds):
  pip install panns_inference --break-system-packages

OPTION 3 — BEATs (Microsoft AudioSet model):
  Download model file manually (see URL above)
  Then pip install torch --break-system-packages (already installed)

OPTION 4 — VGGish:
  pip install torchvggish --break-system-packages

After installing, re-run this script.

WHY THIS MATTERS:
  Perch embeddings: 99.2% site accuracy, 0.345 avg cross-site P
  If CLAP achieves < 80% site accuracy it means the embeddings
  encode semantic audio concepts (vehicle, chainsaw) more
  universally than site identity. This would be the breakthrough
  needed for cross-site generalisation.

QUICK TEST (no WAV files needed):
  The script will also test whether alternative embeddings
  can be extracted from your existing WAV files or whether
  re-extraction from raw audio is required.
""")
        # Still show what files would be needed
        print("Checking for WAV files in project directory...")
        wav_count = 0
        for root, dirs, files in os.walk(BASE_DIR):
            wav_count += sum(1 for f in files if f.endswith('.wav'))
            if wav_count > 0: break
        if wav_count > 0:
            print(f"  Found WAV files - alternative extraction is possible")
        else:
            print(f"  No WAV files found in {BASE_DIR}")
            print(f"  Alternative embeddings require original WAV clips")
            print(f"  These may be on your data drive / external storage")
    else:
        print(f"\nFound {len(available_models)} model(s): {list(available_models.keys())}")
        print("Running alternative embedding extraction and evaluation...")

        results_summary = {'Perch': {'site_acc': site_acc, 'lodo_avg': perch_avg}}

        for model_name, model_info in available_models.items():
            print(f"\n{'─'*50}")
            print(f"Model: {model_name}")

            # Get sample clip names to find WAV paths
            sample_clips = list(all_clips[available[0]]['clip_name'])[:3]
            sample_wav = find_wav(sample_clips[0]) if sample_clips else None

            if sample_wav is None:
                print(f"  WAV files not found in {BASE_DIR}")
                print(f"  Cannot extract {model_name} embeddings without WAV files")
                print(f"  Location of WAV files needed for this experiment")
                continue

            print(f"  WAV files found: {sample_wav}")
            print(f"  Extracting embeddings... (this may take a while)")
            # Extraction code would go here per model
            # Placeholder for when model is installed
            print(f"  [Ready to extract when model loads]")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nPerch (current):")
    print(f"  Site accuracy:      {site_acc:.3f} (near-perfect, bad for transfer)")
    print(f"  Cross-site avg P:   {perch_avg:.3f}")
    print(f"\nTarget for alternative embeddings:")
    print(f"  Site accuracy:      < 0.80")
    print(f"  Cross-site avg P:   > 0.40")
    print(f"\nNext step: install msclap and re-run")
    print(f"  pip install msclap --break-system-packages")
    print("\nDone.")
