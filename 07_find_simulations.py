# ============================================================
# Toucanet Option B — Step 7: Find simulation clips
# Identify which AudioMoth 4 clips have simulations running
# ============================================================

import pandas as pd

if __name__ == '__main__':

    df = pd.read_csv("cleaned_df (1).csv")

    # --- Focus on AudioMoth 4 ---
    am4 = df[df['Recorder'] == 'Audio_Moth_4'].copy()
    print(f"Total AudioMoth 4 clips: {len(am4)}")

    # --- Find simulation clips ---
    sim_clips = am4[am4['Sim Type'] != '[]']
    print(f"Simulation clips: {len(sim_clips)}")
    print("\nSimulation types found:")
    print(sim_clips['Sim Type'].value_counts())

    print("\nSimulation clip filenames:")
    print(sim_clips['clip_name'].tolist())

    # --- Find baseline clips (no simulation) ---
    baseline_clips = am4[am4['Sim Type'] == '[]']
    print(f"\nBaseline clips: {len(baseline_clips)}")
    print("\nFirst 10 baseline clip filenames:")
    print(baseline_clips['clip_name'].head(10).tolist())