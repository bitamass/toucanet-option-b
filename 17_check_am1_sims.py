import pandas as pd
import os

if __name__ == '__main__':
    df = pd.read_csv("cleaned_df (1).csv")
    am1 = df[df['Recorder'] == 'Audio_Moth_1'].copy()
    sim = am1[am1['Sim Type'] != '[]']

    print(f"Total simulation clips: {len(sim)}")
    print(f"\nSample simulation clips:")
    print(sim[['clip_name', 'Sim Type', 'species', 'confidence']].head(20).to_string())

    # Check how many exist in filtered_clips_1
    found = []
    for clip in sim['clip_name'].tolist():
        path = os.path.join("filtered_clips_1", clip)
        if os.path.exists(path):
            found.append(clip)
    print(f"\nSimulation clips found in filtered_clips_1: {len(found)}")