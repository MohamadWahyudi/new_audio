import os
import numpy as np
import pandas as pd
import librosa

def extract_features(file_path):
    """Ekstraksi fitur MFCC + fitur tambahan"""
    try:
        y, sr = librosa.load(file_path, sr=None, mono=True)
        y = librosa.util.normalize(y)

        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
        spec_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        rms = np.mean(librosa.feature.rms(y=y))
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

        features = np.hstack([
            mfcc,
            chroma,
            spec_contrast,
            [zcr, rms, centroid, bandwidth, rolloff]
        ])

        features = np.nan_to_num(features)
        return {f"mfcc{i}": float(v) for i, v in enumerate(features)}

    except Exception as e:
        print(f"[ERROR] Gagal memproses {file_path}: {e}")
        return None


def create_dataset(data_dir="data", output_csv="data/voice_dataset.csv"):
    rows = []
    supported_ext = (".wav", ".m4a", ".mp3")

    print("[INFO] Mulai membaca dataset...")
    total_files = 0
    good = 0
    bad = 0

    for user in os.listdir(data_dir):
        user_path = os.path.join(data_dir, user)
        if not os.path.isdir(user_path):
            continue

        for status in ["buka", "tutup"]:
            status_path = os.path.join(user_path, status)
            if not os.path.isdir(status_path):
                continue

            for file in os.listdir(status_path):
                if file.lower().endswith(supported_ext):
                    total_files += 1
                    file_path = os.path.join(status_path, file)

                    feats = extract_features(file_path)
                    if feats is not None:
                        good += 1
                        feats["user"] = user
                        feats["status"] = status
                        feats["filename"] = file
                        rows.append(feats)
                    else:
                        bad += 1
                        print(f"[SKIP] File gagal diproses: {file_path}")

    df = pd.DataFrame(rows)

    print(f"\nTotal file audio: {total_files}")
    print(f"Good: {good}")
    print(f"Bad : {bad}\n")

    # ============================
    # AUTO LABEL USER & STATUS
    # ============================
    df["user"] = df["user"].astype(str).str.strip()
    df["status"] = df["status"].astype(str).str.strip()

    # Mapping otomatis
    user_map = {name: idx for idx, name in enumerate(sorted(df["user"].unique()))}
    status_map = {"buka": 0, "tutup": 1}

    df["user"] = df["user"].map(user_map)
    df["status"] = df["status"].map(status_map)

    # Pastikan semua fitur numerik
    for col in df.columns:
        if col not in ["user", "status", "filename"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Simpan CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8")

    print(f"[INFO] Dataset berhasil dibuat: {output_csv} ({len(df)} sampel)")
    print("[INFO] user_map:", user_map)
    print("[INFO] status_map:", status_map)

    return df
