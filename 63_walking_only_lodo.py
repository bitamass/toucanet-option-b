"""
63_walking_only_lodo.py  (fixed)
---------------------------------
Walking-only cross-site LODO using Sim Type from cleaned_df.

Walking = "Human Presence on Trail" — only disturbance type
shared across all 5 recorders.
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (precision_recall_fscore_support,
                              precision_recall_curve)

BASE_DIR = r"C:\Users\BitaMassoudi\KasmirWorldProjects\toucanet-option-b"
SEED     = 42

RECORDERS = [
    ('AM4','am4_full_feature_matrix.csv','am4_full_emb.npy'),
    ('AM2','am2_feature_matrix.csv','am2_time_controlled_emb.npy'),
    ('AM5','am5_feature_matrix.csv','am5_time_controlled_emb.npy'),
    ('AM6','am6_feature_matrix.csv','am6_time_controlled_emb.npy'),
    ('AM1','am1_feature_matrix.csv','am1_full_emb.npy'),
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


def load_sim_types():
    """Load simulation types from cleaned_df metadata."""
    meta_path = os.path.join(BASE_DIR, "cleaned_df (1).csv")
    if not os.path.exists(meta_path):
        print("  WARNING: cleaned_df (1).csv not found")
        return {}
    df = pd.read_csv(meta_path)
    sim_map = {}
    for _, row in df.iterrows():
        cn   = row['clip_name']
        stype = str(row.get('Sim Type', '[]'))
        sim_map[cn] = stype
    print(f"  Loaded sim types for {len(sim_map)} clips")
    return sim_map


def is_walking_only(sim_type_str):
    """True if clip is baseline or pure walking simulation."""
    s = sim_type_str.strip()
    if s == '[]' or s == '' or s == 'nan':
        return True  # baseline
    if 'Human Presence on Trail' in s and 'Vehicle' not in s \
       and 'Chainsaw' not in s and 'Gunshot' not in s:
        return True  # pure walking
    return False


def build_clip_df(feat_df, emb, sim_map):
    rows = []
    for cn in feat_df['clip_name'].unique():
        mask = feat_df['clip_name'].values == cn
        cl   = feat_df['clip_label'].values[mask][0]
        hour = feat_df['clip_hour'].values[mask][0]
        ce   = emb[mask]
        stype = sim_map.get(cn, '[]')
        row  = {'clip_name':cn, 'clip_label':cl,
                'clip_hour':hour, 'sim_type':stype,
                'is_walking': is_walking_only(stype)}
        for k in SPECTRAL_KEYS:
            if k in feat_df.columns:
                row[k] = feat_df.loc[mask, k].values[0]
        for i,v in enumerate(ce.mean(0)): row[f'em_{i}'] = v
        for i,v in enumerate(ce.std(0)):  row[f'es_{i}'] = v
        for i,v in enumerate(ce.max(0)):  row[f'ex_{i}'] = v
        rows.append(row)
    return pd.DataFrame(rows)


def get_features(train_df, test_df):
    ec = [c for c in train_df.columns if c.startswith(('em_','es_','ex_'))]
    z_tr_rows, z_te_rows = [], []
    for h in train_df['clip_hour'].unique():
        bm = (train_df['clip_hour']==h) & (train_df['clip_label']==0)
        br = train_df.loc[bm, SPECTRAL_KEYS]
        if len(br) == 0: continue
        hm = br.mean(); hs = br.std().replace(0, 1e-6)
        th = train_df['clip_hour'] == h
        if th.sum() > 0:
            z = (train_df.loc[th, SPECTRAL_KEYS]-hm)/hs
            z.columns = Z_KEYS; z_tr_rows.append(z)
        te_h = test_df['clip_hour'] == h
        if te_h.sum() > 0:
            z = (test_df.loc[te_h, SPECTRAL_KEYS]-hm)/hs
            z.columns = Z_KEYS; z_te_rows.append(z)
    seen = set(train_df['clip_hour'].unique())
    for h in set(test_df['clip_hour'].unique()) - seen:
        te_h = test_df['clip_hour'] == h
        if te_h.sum() > 0:
            z_te_rows.append(pd.DataFrame(
                0, index=test_df.index[te_h], columns=Z_KEYS))
    z_tr = (pd.concat(z_tr_rows).sort_index() if z_tr_rows
            else pd.DataFrame(0, index=train_df.index, columns=Z_KEYS))
    z_te = (pd.concat(z_te_rows).sort_index() if z_te_rows
            else pd.DataFrame(0, index=test_df.index, columns=Z_KEYS))
    pca = PCA(n_components=50, random_state=SEED)
    etr = pca.fit_transform(train_df[ec].values)
    ete = pca.transform(test_df[ec].values)
    X_tr = np.hstack([train_df[SPECTRAL_KEYS].values, z_tr.values, etr]).astype(np.float32)
    X_te = np.hstack([test_df[SPECTRAL_KEYS].values, z_te.values, ete]).astype(np.float32)
    return X_tr, X_te


def evaluate(y, probs):
    if len(np.unique(y)) < 2: return 0, 0, False
    pr,rc,th = precision_recall_curve(y, probs)
    valid = np.where((pr[:-1]>=0.70) & (rc[:-1]>=0.70))[0]
    if len(valid) > 0:
        bi = valid[np.argmax(pr[valid]+rc[valid])]; bt = float(th[bi])
    else:
        bi = np.argmax(pr[:-1]+rc[:-1]); bt = float(th[bi])
    preds = (probs >= bt).astype(int)
    p,r,_,_ = precision_recall_fscore_support(
        y, preds, average='binary', zero_division=0)
    return p, r, (p>=0.70 and r>=0.70)


def run_lodo(all_clips, description, filter_fn=None):
    available = list(all_clips.keys())
    results = {}
    print(f"\n{description}")
    print(f"{'Test':<6} {'P':>7} {'R':>7} {'Beat?':>6} {'N_sim':>7} {'N_total':>8}")
    print("-"*45)
    for test_name in available:
        train_names = [r for r in available if r != test_name]
        test_df  = all_clips[test_name].copy().reset_index(drop=True)
        train_df = pd.concat([all_clips[r] for r in train_names],
                              ignore_index=True)
        if filter_fn is not None:
            train_df = train_df[filter_fn(train_df)].reset_index(drop=True)
            test_df  = test_df[filter_fn(test_df)].reset_index(drop=True)
        y_tr = train_df['clip_label'].values.astype(int)
        y_te = test_df['clip_label'].values.astype(int)
        n_sim = int(y_te.sum())
        if len(np.unique(y_te)) < 2 or n_sim == 0:
            print(f"{test_name:<6} {'—':>7} {'—':>7} {'skip':>6} {n_sim:>7} {len(test_df):>8}")
            continue
        X_tr, X_te = get_features(train_df, test_df)
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)
        n_pos = y_tr.sum(); n_neg = (y_tr==0).sum()
        w = np.where(y_tr==1, n_neg/max(n_pos,1), 1.0)
        gb = GradientBoostingClassifier(
            n_estimators=100, max_depth=3,
            learning_rate=0.1, random_state=SEED, subsample=0.8)
        gb.fit(X_tr_s, y_tr, sample_weight=w)
        probs = gb.predict_proba(X_te_s)[:,1]
        p, r, beat = evaluate(y_te, probs)
        results[test_name] = (p, r, beat)
        print(f"{test_name:<6} {p:>7.3f} {r:>7.3f} {'BEAT' if beat else 'miss':>6} "
              f"{n_sim:>7} {len(test_df):>8}")
    avg = np.mean([v[0] for v in results.values()]) if results else 0
    print(f"{'Avg':<6} {avg:>7.3f}")
    return results, avg


if __name__ == '__main__':

    print("Walking-Only Cross-Site LODO")
    print("="*60)

    # Load sim types from metadata
    print("\nLoading simulation types from metadata...")
    sim_map = load_sim_types()

    # Load all data
    print("\nLoading feature matrices and embeddings...")
    all_clips = {}
    for name,ff,ef in RECORDERS:
        fp=os.path.join(BASE_DIR,ff); ep=os.path.join(BASE_DIR,ef)
        if not os.path.exists(fp) or not os.path.exists(ep): continue
        feat_df=pd.read_csv(fp); emb=np.load(ep)
        n=min(len(feat_df),len(emb))
        feat_df=feat_df.iloc[:n].reset_index(drop=True); emb=emb[:n]
        clip_df=build_clip_df(feat_df, emb, sim_map)
        all_clips[name] = clip_df
        n_walk = int(((clip_df['clip_label']==1) & clip_df['is_walking']).sum())
        n_other= int(((clip_df['clip_label']==1) & ~clip_df['is_walking']).sum())
        n_base = int((clip_df['clip_label']==0).sum())
        print(f"  {name}: {n_base} baseline  |  "
              f"{n_walk} walking sim  |  {n_other} other sim")

    # Experiment 1: Full LODO (all types) — sanity check
    full_res, full_avg = run_lodo(
        all_clips, "EXPERIMENT 1: Full LODO (all disturbance types)")

    # Experiment 2: Walking-only
    walk_filter = lambda df: (df['clip_label']==0) | df['is_walking']
    walk_res, walk_avg = run_lodo(
        all_clips, "EXPERIMENT 2: Walking-only LODO",
        filter_fn=walk_filter)

    # Experiment 3: Hour-matched walking
    # Hours shared across multiple recorders: 8,9,15,16,17,18
    shared_hours = {8, 9, 15, 16, 17, 18}
    hour_filter = lambda df: (
        df['clip_hour'].isin(shared_hours) &
        ((df['clip_label']==0) | df['is_walking'])
    )
    # Only include recorders that have walking sims in shared hours
    hour_clips = {}
    for name, cdf in all_clips.items():
        sub = cdf[hour_filter(cdf)]
        if sub['clip_label'].sum() > 0:
            hour_clips[name] = cdf  # pass full df, filter applied in run_lodo
    if len(hour_clips) >= 2:
        hour_res, hour_avg = run_lodo(
            hour_clips,
            "EXPERIMENT 3: Hour-matched walking LODO (hours 8,9,15,16,17,18)",
            filter_fn=hour_filter)
    else:
        print("\nEXPERIMENT 3: Not enough recorders with matching hours")
        hour_res, hour_avg = {}, 0

    # Summary
    print(f"\n\n{'='*65}")
    print("WALKING-ONLY LODO — FINAL SUMMARY")
    print("="*65)
    print(f"\n{'Rec':<6} {'Full P':>8} {'Walk P':>8} {'Hour P':>8} {'Walk vs Full':>13}")
    print("-"*50)
    for name in all_clips.keys():
        fp = full_res.get(name,(0,0,False))[0]
        wp = walk_res.get(name,(0,0,False))[0]
        hp = hour_res.get(name,(0,0,False))[0] if hour_res else 0
        diff = wp - fp
        verdict = f"+{diff:.3f} better" if diff>0.01 else ("same" if abs(diff)<=0.01 else f"{diff:.3f} worse")
        print(f"{name:<6} {fp:>8.3f} {wp:>8.3f} {hp:>8.3f} {verdict:>13}")

    print(f"\nAverage cross-site precision:")
    print(f"  Full LODO (all types):    {full_avg:.3f}")
    print(f"  Walking-only LODO:        {walk_avg:.3f}")
    print(f"  Hour-matched walking:     {hour_avg:.3f}")
    print(f"  Previous best (all):      0.364")
    print(f"  Walking vs full change:   {walk_avg-full_avg:+.3f}")

    print(f"\nINTERPRETATION:")
    if walk_avg > full_avg + 0.05:
        print("  Walking-only substantially improves cross-site precision.")
        print("  -> Disturbance type mismatch was a significant blocker.")
        print("  -> Collecting matched walking data at all sites would help.")
    elif walk_avg > full_avg + 0.01:
        print("  Marginal improvement from walking-only filter.")
        print("  -> Disturbance type mismatch is a minor contributing factor.")
        print("  -> Site acoustics remain the primary blocker.")
    else:
        print("  Walking-only gives similar precision to full LODO.")
        print("  -> Disturbance type mismatch is NOT the primary cause of failure.")
        print("  -> Site acoustics alone block transfer even for walking.")
        print("  -> Hour overlap is the key missing ingredient.")

    out = os.path.join(BASE_DIR, "walking_only_lodo_results.txt")
    with open(out,"w") as fh:
        fh.write("Walking-Only LODO Results\n"+"="*40+"\n\n")
        fh.write(f"Full LODO avg P:      {full_avg:.3f}\n")
        fh.write(f"Walking-only avg P:   {walk_avg:.3f}\n")
        fh.write(f"Hour-matched avg P:   {hour_avg:.3f}\n")
        fh.write(f"Previous best:        0.364\n")
    print(f"\nSaved -> {out}")
    print("\nDone.")
