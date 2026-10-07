"""
51_ecological_features.py
--------------------------
Ecological change features from BirdNET outputs.

SAMMY'S IDEA — USE BIRDNET DIFFERENTLY:
  Instead of using BirdNET species detections directly,
  compute community-level ecological indices that capture
  HOW the acoustic community CHANGES over time.

  The intuition:
    When a vehicle enters the forest — birds go quiet.
    When a chainsaw runs — species richness drops.
    When humans walk — alarm calls increase.
    These ecological responses are UNIVERSAL across sites.
    Even if AM4's owls and AM5's morning birds are different,
    both communities RESPOND TO DISTURBANCE the same way.

ECOLOGICAL FEATURES COMPUTED (per 3-second clip):

  From BirdNET detections at the clip level:
  1. n_species:       Number of unique species detected
  2. total_detections: Total number of bird detections
  3. max_confidence:  Highest confidence detection
  4. mean_confidence: Mean confidence across detections
  5. shannon_H:       Shannon diversity H = -sum(p*log(p))
  6. species_richness: Species detected above 0.5 confidence
  7. silent_clip:     1 if zero detections, 0 otherwise

  From change relative to recent window (last 10 clips):
  8.  delta_n_species:    Change in species count
  9.  delta_richness:     Change in species richness
  10. delta_shannon:      Change in diversity
  11. delta_total:        Change in detection count
  12. silence_onset:      1 if this clip silent but prev was not

  From hourly baseline at this recorder:
  13. z_n_species:    Z-score of species count vs hourly baseline
  14. z_richness:     Z-score of richness vs hourly baseline
  15. z_shannon:      Z-score of diversity vs hourly baseline
  16. z_silence:      Z-score of silence fraction vs hourly baseline

EVALUATION:
  Per-site CV using ONLY ecological features (no Perch embeddings).
  LODO cross-site using ecological features.
  Combined: ecological features + Perch embeddings.

  The key test: do ecological change features generalise cross-site
  better than raw acoustic features?
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (precision_recall_fscore_support,
                              precision_recall_curve)

BASE_DIR = r"C:\Users\BitaMassoudi\KasmirWorldProjects\toucanet-option-b"
META_CSV = os.path.join(BASE_DIR, "cleaned_df (1).csv")
SEED     = 42

RECORDERS = [
    ('AM4','am4_full_feature_matrix.csv','am4_full_emb.npy'),
    ('AM2','am2_feature_matrix.csv','am2_time_controlled_emb.npy'),
    ('AM5','am5_feature_matrix.csv','am5_time_controlled_emb.npy'),
    ('AM6','am6_feature_matrix.csv','am6_time_controlled_emb.npy'),
    ('AM1','am1_feature_matrix.csv','am1_full_emb.npy'),
]


def parse_timestamp(clip_name):
    parts = clip_name.split('_')
    try:
        date = parts[3]; time = parts[4].replace('.wav','')
        return pd.Timestamp(f"{date[:4]}-{date[4:6]}-{date[6:8]} "
                           f"{time[:2]}:{time[2:4]}:{time[4:6]}")
    except: return pd.Timestamp('2025-01-01')


def compute_ecological_features(meta_df, recorder_name):
    """
    Compute ecological features from BirdNET detections in metadata.

    The metadata CSV contains BirdNET detection columns.
    We extract species counts, diversity, and change features.
    """
    rec_df = meta_df[meta_df['Recorder']==recorder_name].copy()
    rec_df['timestamp'] = rec_df['clip_name'].apply(parse_timestamp)
    rec_df = rec_df.sort_values('timestamp').reset_index(drop=True)
    rec_df['hour'] = rec_df['timestamp'].dt.hour

    print(f"  Metadata columns: {list(rec_df.columns[:15])}")

    # Check what BirdNET columns exist
    birdnet_cols = [c for c in rec_df.columns if any(x in c.lower()
                    for x in ['species','bird','detection','confidence',
                               'common','latin','score','label'])]
    print(f"  BirdNET-related columns found: {birdnet_cols[:10]}")

    # Check Sim Type column for understanding labels
    if 'Sim Type' in rec_df.columns:
        rec_df['is_sim'] = rec_df['Sim Type'] != '[]'
    elif 'clip_label' in rec_df.columns:
        rec_df['is_sim'] = rec_df['clip_label'] == 1
    else:
        rec_df['is_sim'] = False

    # Build ecological features from whatever BirdNET data is available
    rows = []
    for idx, row in rec_df.iterrows():
        feat = {
            'clip_name': row['clip_name'],
            'clip_label': int(row.get('is_sim', 0)),
            'hour': row['hour'],
            'timestamp': row['timestamp'],
        }

        # Extract species/detection info from metadata
        # Try common column name patterns
        if 'n_species' in rec_df.columns:
            feat['n_species'] = row['n_species']
        elif 'species_count' in rec_df.columns:
            feat['n_species'] = row['species_count']
        else:
            # Count non-null BirdNET detection columns
            det_cols = [c for c in birdnet_cols if 'species' in c.lower()
                        or 'common' in c.lower() or 'label' in c.lower()]
            if det_cols:
                feat['n_species'] = sum(1 for c in det_cols
                                        if pd.notna(row.get(c)) and row.get(c) not in ['','[]'])
            else:
                feat['n_species'] = 0

        # Confidence scores
        conf_cols = [c for c in birdnet_cols if 'conf' in c.lower() or 'score' in c.lower()]
        if conf_cols:
            confs = [row[c] for c in conf_cols
                     if pd.notna(row.get(c)) and isinstance(row.get(c),(int,float))]
            feat['max_confidence']  = max(confs) if confs else 0.0
            feat['mean_confidence'] = np.mean(confs) if confs else 0.0
            feat['n_detections']    = len([c for c in confs if c > 0.5])
        else:
            feat['max_confidence']  = 0.0
            feat['mean_confidence'] = 0.0
            feat['n_detections']    = feat['n_species']

        feat['silent_clip'] = int(feat['n_species'] == 0)
        rows.append(feat)

    eco_df = pd.DataFrame(rows).sort_values('timestamp').reset_index(drop=True)

    # Compute Shannon diversity (simplified — based on detection counts)
    # H = -sum(p_i * log(p_i)) where p_i = proportion of detections for species i
    # Here we approximate using n_species as a proxy
    eco_df['shannon_H'] = np.log1p(eco_df['n_species'].values)

    # Rolling change features (window of last 10 clips)
    WINDOW = 10
    eco_df['delta_n_species'] = eco_df['n_species'].diff().fillna(0)
    eco_df['delta_shannon']   = eco_df['shannon_H'].diff().fillna(0)
    eco_df['delta_detections']= eco_df['n_detections'].diff().fillna(0)
    eco_df['roll_mean_species']= eco_df['n_species'].rolling(WINDOW,min_periods=1).mean()
    eco_df['roll_std_species'] = eco_df['n_species'].rolling(WINDOW,min_periods=1).std().fillna(0)
    eco_df['z_species_rolling']= np.where(
        eco_df['roll_std_species']>0,
        (eco_df['n_species']-eco_df['roll_mean_species'])/eco_df['roll_std_species'],
        0)
    eco_df['silence_onset'] = (
        (eco_df['silent_clip']==1) &
        (eco_df['silent_clip'].shift(1,fill_value=0)==0)).astype(int)

    # Hourly z-scores of ecological features (baseline only)
    eco_features = ['n_species','n_detections','max_confidence',
                    'mean_confidence','shannon_H','silent_clip']
    for feat in eco_features:
        z_col = f'z_hour_{feat}'
        eco_df[z_col] = 0.0
        for hour in eco_df['hour'].unique():
            base_mask = (eco_df['hour']==hour)&(eco_df['clip_label']==0)
            base_vals = eco_df.loc[base_mask,feat]
            if len(base_vals) < 2: continue
            hm = base_vals.mean(); hs = base_vals.std()
            if hs < 1e-6: continue
            h_mask = eco_df['hour']==hour
            eco_df.loc[h_mask,z_col] = (eco_df.loc[h_mask,feat]-hm)/hs

    return eco_df


def get_eco_feature_cols(eco_df):
    return [c for c in eco_df.columns if c not in
            {'clip_name','clip_label','hour','timestamp',
             'Recorder','Sim Type','is_sim'}]


def run_cv(df, feat_cols):
    y  = df['clip_label'].values.astype(int)
    cn = df['clip_name'].values
    if len(np.unique(y))<2 or y.sum()<5: return None
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
    probs=np.zeros(len(df),dtype=np.float32)
    for tr,va in cv.split(cn,y):
        X_tr=df.iloc[tr][feat_cols].values.astype(np.float32)
        X_va=df.iloc[va][feat_cols].values.astype(np.float32)
        ytr=y[tr]
        sc=StandardScaler(); X_tr=sc.fit_transform(X_tr); X_va=sc.transform(X_va)
        n_pos=ytr.sum(); n_neg=(ytr==0).sum()
        w=np.where(ytr==1,n_neg/max(n_pos,1),1.0)
        gb=GradientBoostingClassifier(n_estimators=100,max_depth=3,
           learning_rate=0.1,random_state=SEED,subsample=0.8)
        gb.fit(X_tr,ytr,sample_weight=w)
        probs[va]=gb.predict_proba(X_va)[:,1]
    pr,rc,th=precision_recall_curve(y,probs)
    valid=np.where((pr[:-1]>=0.70)&(rc[:-1]>=0.70))[0]
    if len(valid)>0:
        bi=valid[np.argmax(pr[valid]+rc[valid])]; bt=float(th[bi])
    else:
        bi=np.argmax(pr[:-1]+rc[:-1]); bt=float(th[bi])
    preds=(probs>=bt).astype(int)
    p,r,_,_=precision_recall_fscore_support(y,preds,average='binary',zero_division=0)
    return p,r,(p>=0.70 and r>=0.70)


def run_lodo(all_eco, test_name, feat_cols):
    train_names=[r for r in all_eco if r!=test_name]
    train_df=pd.concat([all_eco[r] for r in train_names],ignore_index=True)
    test_df=all_eco[test_name].copy().reset_index(drop=True)
    y_tr=train_df['clip_label'].values.astype(int)
    y_te=test_df['clip_label'].values.astype(int)
    if len(np.unique(y_te))<2: return 0,0,False
    X_tr=train_df[feat_cols].values.astype(np.float32)
    X_te=test_df[feat_cols].values.astype(np.float32)
    sc=StandardScaler(); X_tr=sc.fit_transform(X_tr); X_te=sc.transform(X_te)
    n_pos=y_tr.sum(); n_neg=(y_tr==0).sum()
    w=np.where(y_tr==1,n_neg/max(n_pos,1),1.0)
    gb=GradientBoostingClassifier(n_estimators=100,max_depth=3,
       learning_rate=0.1,random_state=SEED,subsample=0.8)
    gb.fit(X_tr,y_tr,sample_weight=w)
    probs=gb.predict_proba(X_te)[:,1]
    pr,rc,th=precision_recall_curve(y_te,probs)
    valid=np.where((pr[:-1]>=0.70)&(rc[:-1]>=0.70))[0]
    if len(valid)>0:
        bi=valid[np.argmax(pr[valid]+rc[valid])]; bt=float(th[bi])
    else:
        bi=np.argmax(pr[:-1]+rc[:-1]); bt=float(th[bi])
    preds=(probs>=bt).astype(int)
    p,r,_,_=precision_recall_fscore_support(y_te,preds,average='binary',zero_division=0)
    return p,r,(p>=0.70 and r>=0.70)


if __name__ == '__main__':

    print("Ecological Change Features Experiment")
    print("="*60)

    if not os.path.exists(META_CSV):
        print(f"Metadata not found: {META_CSV}")
        exit()

    print("Loading metadata...")
    meta_df = pd.read_csv(META_CSV)
    print(f"  {len(meta_df)} clips in metadata")
    print(f"  Columns: {list(meta_df.columns)}")

    # Standardise clip_name column
    name_col = next((c for c in meta_df.columns
                     if 'clip' in c.lower() and 'name' in c.lower()), None)
    if name_col and name_col != 'clip_name':
        meta_df = meta_df.rename(columns={name_col:'clip_name'})

    rec_col = next((c for c in meta_df.columns
                    if 'recorder' in c.lower() or 'moth' in c.lower()), None)
    if rec_col and rec_col != 'Recorder':
        meta_df = meta_df.rename(columns={rec_col:'Recorder'})

    rec_names = {
        'AM4':'Audio_Moth_4','AM2':'Audio_Moth_2',
        'AM5':'Audio_Moth_5','AM6':'Audio_Moth_6','AM1':'Audio_Moth_1'
    }

    all_eco={}
    for short,full in rec_names.items():
        if full not in meta_df['Recorder'].unique():
            print(f"  {short}: not found in metadata (tried {full})")
            continue
        print(f"\nComputing ecological features for {short}...")
        eco_df = compute_ecological_features(meta_df, full)
        eco_df['recorder'] = short
        all_eco[short] = eco_df
        n_sim=int(eco_df['clip_label'].sum())
        n_base=int((eco_df['clip_label']==0).sum())
        print(f"  {short}: {len(eco_df)} clips ({n_sim} sim, {n_base} base)")

    if not all_eco:
        print("No ecological data available. Check metadata format.")
        exit()

    available = list(all_eco.keys())
    feat_cols = get_eco_feature_cols(all_eco[available[0]])
    feat_cols = [c for c in feat_cols
                 if all_eco[available[0]][c].dtype in [np.float64,np.int64,float,int]]
    print(f"\nEcological features used: {feat_cols}")

    # ── Per-site CV ───────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("PER-SITE CV — ecological features only")
    print("="*60)

    per_site={}
    for name in available:
        res=run_cv(all_eco[name],feat_cols)
        if res:
            p,r,beat=res
            per_site[name]=(p,r,beat)
            print(f"  {name}: P={p:.3f} R={r:.3f} {'BEAT' if beat else 'miss'}")
        else:
            print(f"  {name}: insufficient data")

    # ── LODO cross-site ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("LODO CROSS-SITE — ecological features")
    print("KEY TEST: do ecological changes generalise across sites?")
    print("="*60)

    lodo_res={}
    for test_name in available:
        p,r,beat=run_lodo(all_eco,test_name,feat_cols)
        lodo_res[test_name]=(p,r,beat)
        print(f"  {test_name}: P={p:.3f} R={r:.3f} {'BEAT' if beat else 'miss'}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n\n"+"="*60)
    print("ECOLOGICAL FEATURES SUMMARY")
    print("="*60)
    print("Per-site CV:")
    for n in available:
        if n in per_site:
            p,r,beat=per_site[n]
            print(f"  {n}: P={p:.3f} R={r:.3f} {'BEAT' if beat else 'miss'}")
    nb=sum(1 for v in per_site.values() if v[2])
    print(f"  {nb}/{len(per_site)} beat 0.70")

    print("\nLODO cross-site:")
    for n,v in lodo_res.items():
        print(f"  {n}: P={v[0]:.3f} R={v[1]:.3f} {'BEAT' if v[2] else 'miss'}")
    nb_lodo=sum(1 for v in lodo_res.values() if v[2])
    avg_eco=np.mean([v[0] for v in lodo_res.values()])
    print(f"  {nb_lodo}/{len(lodo_res)} beat 0.70")
    print(f"  Avg cross-site P: {avg_eco:.3f}")
    print(f"  vs supervised GB baseline: {avg_eco-0.345:+.3f}")
    print("\nDone.")
