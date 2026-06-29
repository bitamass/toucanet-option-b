# ============================================================
# Toucanet Option B — Step 5: Metadata Exploration
# Investigate differences between AudioMoth recorders
# ============================================================

import pandas as pd

if __name__ == '__main__':

    df = pd.read_csv("cleaned_df (1).csv")
    print(f"Total clips: {len(df)}")
    print(f"Columns: {list(df.columns)}\n")

    # --- Compare recorders ---
    grouped = df.groupby('Recorder')

    print("=== Clips per recorder ===")
    print(grouped.size())

    print("\n=== Average Spectral Centroid per recorder ===")
    print(grouped['Spectral Centroid'].mean())

    print("\n=== Most common species per recorder ===")
    for recorder, group in grouped:
        top = group['species'].explode().value_counts().head(3)
        print(f"\n{recorder}:")
        print(top.to_string())

    print("\n=== Sim Type distribution per recorder ===")
    print(pd.crosstab(df['Recorder'], df['Sim Type']))

    print("\n=== Time of day distribution per recorder ===")
    print(pd.crosstab(df['Recorder'], df['Time Of Day']))