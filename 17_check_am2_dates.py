import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv("cleaned_df (1).csv")
    am2 = df[df['Recorder'] == 'Audio_Moth_2'].copy()
    sim = am2[am2['Sim Type'] != '[]']

    print(f"Total AM2 simulation clips: {len(sim)}")
    print(f"\nSimulation dates:")
    sim['date'] = sim['clip_name'].str[13:21]
    print(sim['date'].value_counts().to_string())

    print(f"\nSimulation times (first 10):")
    print(sim['clip_name'].head(10).tolist())