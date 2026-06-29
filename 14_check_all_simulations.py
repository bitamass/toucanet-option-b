# ============================================================
# Toucanet Option B — Step 14: Check simulation data
# across ALL recorders
# ============================================================

import pandas as pd

if __name__ == '__main__':

    df = pd.read_csv("cleaned_df (1).csv")

    print(f"Total clips: {len(df)}")
    print(f"Total recorders: {df['Recorder'].nunique()}")
    print(f"Recorders: {df['Recorder'].unique()}")

    print(f"\n=== Simulation clips per recorder ===")
    df['is_sim'] = df['Sim Type'] != '[]'

    for recorder in sorted(df['Recorder'].unique()):
        rec_df = df[df['Recorder'] == recorder]
        sim_df = rec_df[rec_df['is_sim']]
        print(f"\n{recorder}:")
        print(f"  Total clips:      {len(rec_df)}")
        print(f"  Simulation clips: {len(sim_df)}")
        if len(sim_df) > 0:
            print(f"  Simulation types:")
            print(sim_df['Sim Type'].value_counts().to_string())