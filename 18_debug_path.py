import os

if __name__ == '__main__':
    folder = "filtered_clips_1"
    all_files = os.listdir(folder)

    # Find ALL files with 181 in the name
    files_181 = sorted([f for f in all_files if '181' in f])
    print(f"Files with '181' in name: {len(files_181)}")
    for f in files_181:
        print(f"  '{f}'")

    # Total files in folder
    print(f"\nTotal files in folder: {len(all_files)}")

    # Check specific file
    target = "Audio_Moth_1_20250318_181418.wav"
    print(f"\nLooking for: '{target}'")
    print(f"Exists: {os.path.exists(os.path.join(folder, target))}")

    # Check if file exists anywhere
    for root, dirs, files in os.walk(folder):
        for f in files:
            if '181418' in f:
                print(f"FOUND at: {os.path.join(root, f)}")