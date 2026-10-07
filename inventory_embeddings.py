import numpy as np, glob, pandas as pd
print("=== NPY / NPZ FILES (candidate Perch embeddings) ===")
for f in sorted(glob.glob("*.npy")) + sorted(glob.glob("*.npz")):
    try:
        if f.endswith(".npz"):
            z = np.load(f, allow_pickle=True)
            for k in z.files:
                print(f"  {f} [{k}] shape={z[k].shape} dtype={z[k].dtype}")
        else:
            a = np.load(f, allow_pickle=True, mmap_mode="r")
            print(f"  {f:44s} shape={getattr(a,'shape','?')} dtype={getattr(a,'dtype','?')}")
    except Exception as e:
        print(f"  {f}: {e}")
print()
print("=== Perch-ish files anywhere under this folder ===")
import os
for root,_,files in os.walk("."):
    for fn in files:
        if any(k in fn.lower() for k in ["perch","embed","emb_"]):
            p=os.path.join(root,fn)
            print(f"  {p}  ({os.path.getsize(p)//1024} KB)")
