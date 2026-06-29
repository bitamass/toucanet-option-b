# ============================================================
# Toucanet Option B — Step 12: Check species diversity
# Make sure Spectacled Owl isn't the only species recorded
# ============================================================

import pandas as pd

if __name__ == '__main__':

    df = pd.read_csv("cleaned_df (1).csv")
    am4 = df[df['Recorder'] == 'Audio_Moth_4'].copy()

    print(f"Total AudioMoth 4 clips: {len(am4)}")
    print(f"\n=== All species detected at AudioMoth 4 ===")
    print(am4['species'].value_counts().to_string())

    print(f"\n=== Total unique species at AudioMoth 4 ===")
    print(f"{am4['species'].nunique()} unique species")

    print(f"\n=== Spectacled Owl % of all clips ===")
    spectacled = am4['species'].str.contains('Pulsatrix', na=False).sum()
    print(f"Spectacled Owl clips: {spectacled}")
    print(f"Total clips: {len(am4)}")
    print(f"Percentage: {100*spectacled/len(am4):.1f}%")