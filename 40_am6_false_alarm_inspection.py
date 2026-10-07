"""
40_am6_false_alarm_inspection.py
----------------------------------
Inspects AudioMoth 6 false alarms to understand WHY precision
is stuck at 0.597.

A false alarm = a baseline clip the classifier flagged as disturbance.

We want to know:
  - What hour of day are they concentrated in?
  - What date are they from?
  - What do their acoustic features look like?
  - Are they rain-like? Wind-like? Species-specific?
  - How confident is the classifier (probability score)?

This tells us exactly which fix to apply:
  - If clustered at specific hours → hour restriction
  - If high rain scores → tighten rain filter
  - If specific dates → date effect
  - If high confidence false alarms → fundamental feature problem

Uses saved am6 feature matrix and embeddings.
No re-embedding needed. Runs in about 3-5 minutes.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve

BASE_DIR = r"C:\Users\BitaMassoudi\KasmirWorldProjects\toucanet-option-b"

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

RANDOM_SEED = 42
N_FOLDS     = 5


def build_clip_df(feat_df, emb, labels):
    rows = []
    for cn in feat_df['clip_name'].unique():
        mask  = feat_df['clip_name'].values == cn
        cl    = feat_df['clip_label'].values[mask][0]
        hour  = feat_df['clip_hour'].values[mask][0]
        date  = cn.split('_')[3]
        ce    = emb[mask]
        row   = {'clip_name':cn,'clip_label':cl,
                 'clip_hour':hour,'date':date}
        for k in SPECTRAL_KEYS:
            if k in feat_df.columns:
                row[k] = feat_df.loc[mask,k].values[0]
        for i,v in enumerate(ce.mean(0)): row[f'em_{i}'] = v
        for i,v in enumerate(ce.std(0)):  row[f'es_{i}'] = v
        for i,v in enumerate(ce.max(0)):  row[f'ex_{i}'] = v
        rows.append(row)
    return pd.DataFrame(rows)


def compute_rain_score(clip_df):
    rms   = clip_df['rms_mean'].values
    ent   = clip_df['spec_entropy_mean'].values
    roll  = clip_df['rolloff_std'].values
    sil   = clip_df['silence_fraction'].values
    rms_n  = (rms-rms.min())/(rms.max()-rms.min()+1e-10)
    ent_n  = (ent-ent.min())/(ent.max()-ent.min()+1e-10)
    roll_n = 1.0-(roll-roll.min())/(roll.max()-roll.min()+1e-10)
    sil_n  = 1.0-(sil-sil.min())/(sil.max()-sil.min()+1e-10)
    return 0.35*rms_n + 0.35*ent_n + 0.20*roll_n + 0.10*sil_n


def run_cv_get_probs(clip_df):
    """Run clip-level GB CV and return per-clip probability scores."""
    meta    = {'clip_name','clip_label','clip_hour','date'}
    emb_c   = [c for c in clip_df.columns if c.startswith(('em_','es_','ex_'))]
    clip_nm = clip_df['clip_name'].values
    clip_lb = clip_df['clip_label'].values

    cv    = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                            random_state=RANDOM_SEED)
    probs = np.zeros(len(clip_df), dtype=np.float32)

    for fold, (tr, va) in enumerate(cv.split(clip_nm, clip_lb)):
        tr_df = clip_df.iloc[tr].copy().reset_index(drop=True)
        va_df = clip_df.iloc[va].copy().reset_index(drop=True)
        y_tr  = clip_lb[tr].astype(int)

        # Z-scores from training baseline only
        z_tr_rows, z_va_rows = [], []
        for hour in tr_df['clip_hour'].unique():
            bm = (tr_df['clip_hour']==hour) & (tr_df['clip_label']==0)
            br = tr_df.loc[bm, SPECTRAL_KEYS]
            if len(br) == 0: continue
            hm = br.mean(); hs = br.std().replace(0,1e-6)
            th = tr_df['clip_hour']==hour
            if th.sum()>0:
                z = (tr_df.loc[th,SPECTRAL_KEYS]-hm)/hs
                z.columns = Z_KEYS; z_tr_rows.append(z)
            vh = va_df['clip_hour']==hour
            if vh.sum()>0:
                z = (va_df.loc[vh,SPECTRAL_KEYS]-hm)/hs
                z.columns = Z_KEYS; z_va_rows.append(z)

        z_tr = (pd.concat(z_tr_rows).sort_index() if z_tr_rows else
                pd.DataFrame(0,index=tr_df.index,columns=Z_KEYS))
        z_va = (pd.concat(z_va_rows).sort_index() if z_va_rows else
                pd.DataFrame(0,index=va_df.index,columns=Z_KEYS))

        pca      = PCA(n_components=50, random_state=RANDOM_SEED)
        etr      = pca.fit_transform(tr_df[emb_c].values)
        eva      = pca.transform(va_df[emb_c].values)

        X_tr = np.hstack([tr_df[SPECTRAL_KEYS].values,
                          z_tr.values, etr]).astype(np.float32)
        X_va = np.hstack([va_df[SPECTRAL_KEYS].values,
                          z_va.values, eva]).astype(np.float32)

        sc     = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_va_s = sc.transform(X_va)

        n_pos = y_tr.sum(); n_neg = (y_tr==0).sum()
        w     = np.where(y_tr==1, n_neg/max(n_pos,1), 1.0)
        gb    = GradientBoostingClassifier(n_estimators=100,max_depth=3,
                learning_rate=0.1,random_state=RANDOM_SEED,subsample=0.8)
        gb.fit(X_tr_s, y_tr, sample_weight=w)
        probs[va] = gb.predict_proba(X_va_s)[:,1]

    return probs


if __name__ == '__main__':

    print("AM6 False Alarm Inspection")
    print("="*60)

    feat_path = os.path.join(BASE_DIR,'am6_feature_matrix.csv')
    emb_path  = os.path.join(BASE_DIR,'am6_time_controlled_emb.npy')
    lab_path  = os.path.join(BASE_DIR,'am6_time_controlled_labels.npy')

    feat_df = pd.read_csv(feat_path)
    emb     = np.load(emb_path)
    labels  = np.load(lab_path)
    n = min(len(feat_df),len(emb))
    feat_df = feat_df.iloc[:n].reset_index(drop=True)
    emb = emb[:n]; labels = labels[:n]

    print(f"Building clip-level features...")
    clip_df = build_clip_df(feat_df, emb, labels)
    clip_df['rain_score'] = compute_rain_score(clip_df)
    print(f"  {len(clip_df)} clips  "
          f"({int(clip_df['clip_label'].sum())} sim, "
          f"{int((clip_df['clip_label']==0).sum())} base)")

    print(f"\nRunning 5-fold CV to get probability scores...")
    probs = run_cv_get_probs(clip_df)
    clip_df['prob'] = probs

    # Find best threshold
    y  = clip_df['clip_label'].values.astype(int)
    pr, rc, th = precision_recall_curve(y, probs)
    valid = np.where((pr[:-1]>=0.70)&(rc[:-1]>=0.70))[0]
    if len(valid)>0:
        bi = valid[np.argmax(pr[valid]+rc[valid])]
        bt = float(th[bi])
    else:
        bi = np.argmax(pr[:-1]+rc[:-1])
        bt = float(th[bi])

    clip_df['pred'] = (probs >= bt).astype(int)

    # Identify false alarms and missed detections
    false_alarms  = clip_df[(clip_df['clip_label']==0) &
                             (clip_df['pred']==1)].copy()
    true_pos      = clip_df[(clip_df['clip_label']==1) &
                             (clip_df['pred']==1)].copy()
    missed        = clip_df[(clip_df['clip_label']==1) &
                             (clip_df['pred']==0)].copy()
    true_neg      = clip_df[(clip_df['clip_label']==0) &
                             (clip_df['pred']==0)].copy()

    p_achieved = len(true_pos)/(len(true_pos)+len(false_alarms)+1e-10)
    r_achieved = len(true_pos)/(len(true_pos)+len(missed)+1e-10)

    print(f"\nAt threshold t={bt:.3f}:")
    print(f"  True positives:  {len(true_pos)}")
    print(f"  False alarms:    {len(false_alarms)}")
    print(f"  Missed events:   {len(missed)}")
    print(f"  True negatives:  {len(true_neg)}")
    print(f"  Precision: {p_achieved:.3f}  Recall: {r_achieved:.3f}")

    # ── ANALYSIS 1: Hour distribution ─────────────────────────────────────────
    print(f"\n{'─'*55}")
    print("FALSE ALARM DISTRIBUTION BY HOUR")
    print(f"{'─'*55}")
    fa_hours  = false_alarms['clip_hour'].value_counts().sort_index()
    tn_hours  = true_neg['clip_hour'].value_counts().sort_index()
    all_hours = sorted(clip_df['clip_hour'].unique())

    print(f"{'Hour':>5}  {'FA count':>9}  {'TN count':>9}  "
          f"{'FA rate':>8}  {'Alarm rate':>11}")
    for h in all_hours:
        fa_n = fa_hours.get(h,0)
        tn_n = tn_hours.get(h,0)
        total_base_h = fa_n + tn_n
        fa_rate = fa_n/max(total_base_h,1)
        # How many base clips at this hour get flagged
        base_h = clip_df[(clip_df['clip_label']==0) &
                          (clip_df['clip_hour']==h)]
        alarm_rate = (base_h['pred']==1).mean()
        flag = " ← HIGH" if fa_rate > 0.25 else ""
        print(f"  {h:>3}    {fa_n:>9}  {tn_n:>9}  "
              f"{fa_rate:>7.1%}  {alarm_rate:>10.1%}{flag}")

    # ── ANALYSIS 2: Date distribution ─────────────────────────────────────────
    print(f"\n{'─'*55}")
    print("FALSE ALARM DISTRIBUTION BY DATE")
    print(f"{'─'*55}")
    fa_dates = false_alarms['date'].value_counts().sort_index()
    all_dates = sorted(clip_df['date'].unique())
    for d in all_dates:
        fa_n  = fa_dates.get(d,0)
        base_d = clip_df[(clip_df['clip_label']==0)&(clip_df['date']==d)]
        rate  = fa_n/max(len(base_d),1)
        sim_d = clip_df[(clip_df['clip_label']==1)&(clip_df['date']==d)]
        print(f"  {d}: {fa_n} false alarms / {len(base_d)} base clips "
              f"({rate:.1%} alarm rate)  [{len(sim_d)} sim clips]")

    # ── ANALYSIS 3: Rain score of false alarms ────────────────────────────────
    print(f"\n{'─'*55}")
    print("RAIN SCORE ANALYSIS")
    print(f"{'─'*55}")
    fa_rain  = false_alarms['rain_score'].describe()
    tn_rain  = true_neg['rain_score'].describe()
    tp_rain  = true_pos['rain_score'].describe()

    print(f"  {'Group':<20} {'Mean':>7} {'Median':>8} {'p75':>7} {'p90':>7}")
    print(f"  {'False alarms':<20} "
          f"{fa_rain['mean']:>7.3f} "
          f"{false_alarms['rain_score'].median():>8.3f} "
          f"{false_alarms['rain_score'].quantile(0.75):>7.3f} "
          f"{false_alarms['rain_score'].quantile(0.90):>7.3f}")
    print(f"  {'True negatives':<20} "
          f"{tn_rain['mean']:>7.3f} "
          f"{true_neg['rain_score'].median():>8.3f} "
          f"{true_neg['rain_score'].quantile(0.75):>7.3f} "
          f"{true_neg['rain_score'].quantile(0.90):>7.3f}")
    print(f"  {'True positives':<20} "
          f"{tp_rain['mean']:>7.3f} "
          f"{true_pos['rain_score'].median():>8.3f} "
          f"{true_pos['rain_score'].quantile(0.75):>7.3f} "
          f"{true_pos['rain_score'].quantile(0.90):>7.3f}")

    fa_high_rain = (false_alarms['rain_score'] >
                    true_neg['rain_score'].quantile(0.90)).sum()
    print(f"\n  False alarms with high rain score (>base p90): "
          f"{fa_high_rain}/{len(false_alarms)} "
          f"({fa_high_rain/max(len(false_alarms),1):.1%})")

    # ── ANALYSIS 4: Acoustic feature comparison ───────────────────────────────
    print(f"\n{'─'*55}")
    print("TOP DISTINGUISHING FEATURES")
    print("(what makes false alarms look like disturbance)")
    print(f"{'─'*55}")

    key_features = ['rms_mean','spec_entropy_mean','rolloff_mean',
                    'centroid_mean','bandwidth_mean','silence_fraction',
                    'onset_count','temporal_entropy','zcr_mean']

    print(f"  {'Feature':<25} {'FA mean':>9} {'TN mean':>9} "
          f"{'Sim mean':>9} {'FA closer to':>13}")
    for feat in key_features:
        if feat not in clip_df.columns: continue
        fa_m  = false_alarms[feat].mean()
        tn_m  = true_neg[feat].mean()
        tp_m  = true_pos[feat].mean()
        # Is false alarm mean closer to sim or base?
        closer = "SIM" if abs(fa_m-tp_m) < abs(fa_m-tn_m) else "baseline"
        flag   = " ←" if closer=="SIM" else ""
        print(f"  {feat:<25} {fa_m:>9.4f} {tn_m:>9.4f} "
              f"{tp_m:>9.4f} {closer:>13}{flag}")

    # ── ANALYSIS 5: Confidence distribution ───────────────────────────────────
    print(f"\n{'─'*55}")
    print("CLASSIFIER CONFIDENCE OF FALSE ALARMS")
    print(f"{'─'*55}")
    fa_conf = false_alarms['prob'].describe()
    print(f"  Min:    {false_alarms['prob'].min():.3f}")
    print(f"  Mean:   {false_alarms['prob'].mean():.3f}")
    print(f"  Median: {false_alarms['prob'].median():.3f}")
    print(f"  p75:    {false_alarms['prob'].quantile(0.75):.3f}")
    print(f"  Max:    {false_alarms['prob'].max():.3f}")

    high_conf_fa = (false_alarms['prob'] > 0.80).sum()
    print(f"\n  High-confidence false alarms (prob>0.80): "
          f"{high_conf_fa}/{len(false_alarms)} "
          f"({high_conf_fa/max(len(false_alarms),1):.1%})")
    print(f"  These are the hardest to fix — model is very sure about them")

    # ── ANALYSIS 6: Top false alarm clips ────────────────────────────────────
    print(f"\n{'─'*55}")
    print("TOP 15 FALSE ALARMS BY CONFIDENCE")
    print("(clips the model is most wrong about)")
    print(f"{'─'*55}")
    top_fa = false_alarms.nlargest(15,'prob')[
        ['clip_name','clip_hour','date','prob','rain_score',
         'rms_mean','spec_entropy_mean','onset_count']]
    print(top_fa.to_string(index=False))

    # ── ANALYSIS 7: What threshold would fix precision ────────────────────────
    print(f"\n{'─'*55}")
    print("THRESHOLD SENSITIVITY FOR AM6")
    print(f"{'─'*55}")
    print(f"  {'Threshold':>10} {'Precision':>10} {'Recall':>8} "
          f"{'FA count':>9} {'Beat?':>6}")
    for t in np.arange(0.3, 0.95, 0.05):
        preds_t = (probs >= t).astype(int)
        tp_t = ((preds_t==1)&(y==1)).sum()
        fp_t = ((preds_t==1)&(y==0)).sum()
        fn_t = ((preds_t==0)&(y==1)).sum()
        p_t  = tp_t/max(tp_t+fp_t,1)
        r_t  = tp_t/max(tp_t+fn_t,1)
        beat = "YES" if p_t>=0.70 and r_t>=0.70 else "no"
        flag = " ←" if beat=="YES" else ""
        print(f"  {t:>10.2f} {p_t:>10.3f} {r_t:>8.3f} "
              f"{int(fp_t):>9} {beat:>6}{flag}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1,3,figsize=(14,4))
    fig.patch.set_facecolor("#2C5F2D")

    # Hour distribution
    ax = axes[0]; ax.set_facecolor("#2C5F2D")
    fa_by_hour = false_alarms.groupby('clip_hour').size()
    base_by_hour = true_neg.groupby('clip_hour').size()
    hours = sorted(set(fa_by_hour.index)|set(base_by_hour.index))
    x = range(len(hours))
    ax.bar([i-0.2 for i in x],
           [fa_by_hour.get(h,0) for h in hours],
           0.4, label='False alarms', color='#E05C5C')
    ax.bar([i+0.2 for i in x],
           [base_by_hour.get(h,0)//10 for h in hours],
           0.4, label='True neg (/10)', color='#4A90D9', alpha=0.7)
    ax.set_xticks(list(x)); ax.set_xticklabels(hours,color='white',fontsize=9)
    ax.tick_params(colors='white')
    ax.set_title('False alarms by hour',color='white',fontsize=10)
    ax.legend(facecolor='#1F451F',labelcolor='white',fontsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor('#97BC62')

    # Rain score distributions
    ax = axes[1]; ax.set_facecolor("#2C5F2D")
    ax.hist(true_neg['rain_score'],bins=30,color='#4A90D9',
            alpha=0.6,label='True neg',density=True)
    ax.hist(false_alarms['rain_score'],bins=30,color='#E05C5C',
            alpha=0.7,label='False alarms',density=True)
    ax.hist(true_pos['rain_score'],bins=30,color='#97BC62',
            alpha=0.7,label='True pos (sim)',density=True)
    ax.tick_params(colors='white')
    ax.set_title('Rain score distribution',color='white',fontsize=10)
    ax.legend(facecolor='#1F451F',labelcolor='white',fontsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor('#97BC62')

    # Confidence distribution of false alarms
    ax = axes[2]; ax.set_facecolor("#2C5F2D")
    ax.hist(false_alarms['prob'],bins=20,color='#E05C5C',
            alpha=0.8,label='False alarm confidence')
    ax.axvline(bt,color='white',linestyle='--',linewidth=1.5,
               label=f'Threshold={bt:.2f}')
    ax.tick_params(colors='white')
    ax.set_title('FA confidence scores',color='white',fontsize=10)
    ax.legend(facecolor='#1F451F',labelcolor='white',fontsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor('#97BC62')

    plt.tight_layout()
    out = os.path.join(BASE_DIR,'am6_false_alarm_analysis.png')
    plt.savefig(out,dpi=150,bbox_inches='tight',facecolor='#2C5F2D')
    print(f"\nPlot saved -> {out}")
    print("\nDone. Review the hour distribution and rain score analysis")
    print("to determine the best fix for AM6.")