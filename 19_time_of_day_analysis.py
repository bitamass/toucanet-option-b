# ============================================================
# Toucanet Option B — Step 19: Time of day analysis
# Control for time of day confounding
# ============================================================

import pandas as pd

if __name__ == '__main__':

    df = pd.read_csv("cleaned_df (1).csv")

    # ── AudioMoth 1 ──────────────────────────────────────────
    am1 = df[df['Recorder'] == 'Audio_Moth_1'].copy()
    am1['is_sim'] = am1['Sim Type'] != '[]'
    am1['hour'] = am1['clip_name'].str[22:24].astype(int)

    print("=" * 50)
    print("AUDIOMOTH 1")
    print("=" * 50)

    print("\nSimulation clip hours:")
    print(am1[am1['is_sim']]['hour'].value_counts().sort_index().to_string())

    print("\nTop species during simulation hours (hour 18):")
    hour18 = am1[am1['hour'] == 18]
    print(f"  Total clips at hour 18: {len(hour18)}")
    print(f"  Simulation clips:       {hour18['is_sim'].sum()}")
    print(f"  Baseline clips:         {(~hour18['is_sim']).sum()}")
    print(f"\nTop species at hour 18 — baseline only:")
    base_18 = hour18[~hour18['is_sim']]
    print(base_18['species'].value_counts().head(8).to_string())
    print(f"\nTop species at hour 18 — simulation only:")
    sim_18 = hour18[hour18['is_sim']]
    print(sim_18['species'].value_counts().head(8).to_string())

    # ── AudioMoth 4 ──────────────────────────────────────────
    am4 = df[df['Recorder'] == 'Audio_Moth_4'].copy()
    am4['is_sim'] = am4['Sim Type'] != '[]'
    am4['hour'] = am4['clip_name'].str[22:24].astype(int)

    print("\n" + "=" * 50)
    print("AUDIOMOTH 4")
    print("=" * 50)

    print("\nSimulation clip hours:")
    print(am4[am4['is_sim']]['hour'].value_counts().sort_index().to_string())

    print("\nTop species during simulation hours:")
    sim_hours = am4[am4['is_sim']]['hour'].unique()
    for hour in sorted(sim_hours):
        hour_df = am4[am4['hour'] == hour]
        print(f"\nHour {hour}:")
        print(f"  Simulation clips: {hour_df['is_sim'].sum()}")
        print(f"  Baseline clips:   {(~hour_df['is_sim']).sum()}")
        print(f"  Top species baseline:")
        base = hour_df[~hour_df['is_sim']]
        if len(base) > 0:
            print(base['species'].value_counts().head(3).to_string())
        print(f"  Top species simulation:")
        sim = hour_df[hour_df['is_sim']]
        if len(sim) > 0:
            print(sim['species'].value_counts().head(3).to_string())