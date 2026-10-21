from msclap import CLAP
import os

BASE_DIR = r"C:\Users\BitaMassoudi\KasmirWorldProjects\toucanet-option-b"
wav_dir = os.path.join(BASE_DIR, "filtered_clips_6")
wavs = [f for f in os.listdir(wav_dir) if f.endswith(".wav")][:3]
model = CLAP(version="2023", use_cuda=False)
for w in wavs:
    path = os.path.join(wav_dir, w)
    try:
        emb = model.get_audio_embeddings([path])
        print(f"SUCCESS: {w} -> shape {emb.shape}")
    except Exception as e:
        print(f"FAILED: {e}")
