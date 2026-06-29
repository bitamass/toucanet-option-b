import os
import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv("cleaned_df (1).csv")
    am1 = df[df['Recorder'] == 'Audio_Moth_1'].copy()
    sim = am1[am1['Sim Type'] != '[]']

    # What times are simulation clips?
    print("Simulation clip times:")
    print(sim['clip_name'].head(20).tolist())

    # What times do we have in folder?
    all_files = os.listdir("filtered_clips_1")
    march18 = sorted([f for f in all_files if '20250318' in f])
    print(f"\nLatest March 18 files in folder:")
    for f in march18[-10:]:
        print(f"  {f}")