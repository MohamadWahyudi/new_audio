# app.py
import streamlit as st
import numpy as np
import joblib
import tempfile
import pandas as pd
import os
import traceback

from utils.feature_extraction import extract_features
from st_audiorec import st_audiorec
import librosa
import speech_recognition as sr
from pydub import AudioSegment

# DTW
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# ---------------------------
# Konfigurasi dasar
# ---------------------------
st.set_page_config(page_title="Voice Identification System", layout="wide")

st.title("Voice Identification System")
st.sidebar.title("Pengaturan & Informasi")
st.sidebar.info(
    "Sistem mengenali dua pengguna (user1, user2) dan mengidentifikasi kata 'buka' atau 'tutup'.\n"
    "Jika suara tidak mirip dengan data yang ada, akan ditandai sebagai 'anomali'.\n\n"
    "Selain prediksi model, aplikasi ini juga menghitung jarak DTW ke dataset dan menampilkan perhitungan."
)
st.sidebar.markdown("---")

# ---------------------------
# Lokasi file / folder penting
# ---------------------------
MODELS_DIR = "models"
DATA_DIR = "data"           # folder berisi wav asli (struktur: data/user1/buka/*.wav dst)
DATA_MFCC_DIR = "data_mfcc" # folder berisi file .npy MFCC yang digunakan untuk DTW (akan dibuat jika belum ada)

# Developer note: previously uploaded feature_cols.pkl is at /mnt/data/feature_cols.pkl (if needed)
# path_example = "/mnt/data/feature_cols.pkl"

# ---------------------------
# Load model & feature_cols robust
# ---------------------------
def safe_load_feature_cols(path_models):
    p = os.path.join(path_models, "feature_cols.pkl")
    if os.path.exists(p):
        try:
            cols = joblib.load(p)
            # if saved incorrectly as single int, try fallback to CSV
            if isinstance(cols, list) and all(isinstance(c, str) for c in cols):
                return cols
            else:
                st.warning("feature_cols.pkl tidak berformat list string. Mencoba fallback ke CSV dataset...")
        except Exception as e:
            st.warning(f"Gagal load feature_cols.pkl: {e}. Mencoba fallback ke CSV dataset...")
    # fallback: try to read CSV and infer mfcc columns
    csv_path = os.path.join("data", "voice_dataset.csv")
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            cols = [c for c in df.columns if c.startswith("mfcc")]
            if len(cols) > 0:
                st.info(f"Feature cols inferred from CSV ({len(cols)} kolom).")
                # Save corrected feature_cols for future runs
                os.makedirs(path_models, exist_ok=True)
                joblib.dump(cols, p)
                return cols
        except Exception as e:
            st.error(f"Gagal membaca {csv_path}: {e}")
    st.error("Tidak menemukan feature_cols yang valid. Pastikan models/feature_cols.pkl valid atau data/voice_dataset.csv ada.")
    st.stop()

def safe_load_model(path_models, name):
    p = os.path.join(path_models, name)
    if not os.path.exists(p):
        st.error(f"Model tidak ditemukan: {p}")
        st.stop()
    try:
        return joblib.load(p)
    except Exception as e:
        st.error(f"Gagal memuat model {p}: {e}")
        st.stop()

# load models and feature cols
model_user = safe_load_model(MODELS_DIR, "user_model.pkl")
model_status = safe_load_model(MODELS_DIR, "status_model.pkl")
feature_cols = safe_load_feature_cols(MODELS_DIR)

# try load user_map if exists, build inverse map for labels
user_map_path = os.path.join(MODELS_DIR, "user_map.pkl")
if os.path.exists(user_map_path):
    try:
        user_map = joblib.load(user_map_path)
        inv_user_map = {v: k for k, v in user_map.items()}
    except:
        inv_user_map = None
else:
    inv_user_map = None

st.sidebar.write(f"Jumlah fitur model: {len(feature_cols)}")

# ---------------------------
# Util: konversi m4a -> wav
# ---------------------------
def convert_m4a_to_wav(input_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
            audio = AudioSegment.from_file(input_file, format="m4a")
            audio.export(tmp_wav.name, format="wav")
            return tmp_wav.name
    except Exception as e:
        st.error(f"Gagal mengonversi file m4a: {e}")
        return None

# ---------------------------
# Ekstraksi MFCC timeseries untuk DTW
# ---------------------------
def extract_mfcc_timeseries(audio_path, sr_target=16000, n_mfcc=20):
    y, sr = librosa.load(audio_path, sr=sr_target)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc.T  # shape: (time_steps, n_mfcc)

# ---------------------------
# Siapkan folder data_mfcc (simpan .npy jika perlu)
# Jika sudah ada .npy, gunakan itu; jika belum, buat dari wav di data/
# ---------------------------
def prepare_data_mfcc(data_dir=DATA_DIR, out_dir=DATA_MFCC_DIR, n_mfcc=20, sr_target=16000):
    os.makedirs(out_dir, exist_ok=True)
    # jika sudah ada .npy di out_dir, anggap sudah siap
    existing = [f for f in os.listdir(out_dir) if f.endswith(".npy")]
    if len(existing) > 0:
        return
    # scan WAV files under data_dir recursively and create .npy
    for root, _, files in os.walk(data_dir):
        for fname in files:
            if fname.lower().endswith((".wav", ".m4a", ".mp3")):
                wav_path = os.path.join(root, fname)
                try:
                    mf = extract_mfcc_timeseries(wav_path, sr_target=sr_target, n_mfcc=n_mfcc)
                    base = os.path.splitext(fname)[0]
                    # create a safe output name including subfolder structure
                    rel = os.path.relpath(root, data_dir).replace(os.sep, "_")
                    out_name = f"{rel}_{base}.npy" if rel != "." else f"{base}.npy"
                    out_path = os.path.join(out_dir, out_name)
                    np.save(out_path, mf)
                except Exception as e:
                    st.write(f"Gagal memproses {wav_path}: {e}")

# ---------------------------
# Hitung DTW distances antara mfcc_new dan dataset .npy
# ---------------------------
def compute_dtw_distances(mfcc_new, dataset_folder=DATA_MFCC_DIR):
    rows = []
    if not os.path.exists(dataset_folder):
        return pd.DataFrame(rows)
    for fname in os.listdir(dataset_folder):
        if not fname.endswith(".npy"):
            continue
        try:
            mf_old = np.load(os.path.join(dataset_folder, fname))
            dist, _ = fastdtw(mfcc_new, mf_old, dist=euclidean)
            rows.append({"file": fname, "dtw": float(dist)})
        except Exception as e:
            rows.append({"file": fname, "dtw": np.nan, "error": str(e)})
    if len(rows) == 0:
        return pd.DataFrame(rows)
    df = pd.DataFrame(rows)
    df = df.sort_values("dtw", na_position="last").reset_index(drop=True)
    return df

# ---------------------------
# Fungsi utama proses audio
# ---------------------------
def process_audio(audio_path):
    try:
        # 1) ekstrak fitur ringkas untuk model (menggunakan utils.feature_extraction)
        features = extract_features(audio_path)
        if not features:
            st.error("Gagal mengekstraksi fitur suara.")
            return

        # cek duration & rms
        y, sr_audio = librosa.load(audio_path, sr=None)
        rms = float(np.mean(librosa.feature.rms(y=y)))
        dur = len(y) / sr_audio
        if dur < 0.1 or rms < 1e-4:
            st.error("Tidak ada suara yang terdeteksi. Silakan ulangi.")
            return

        feature_df = pd.DataFrame([features])

        # Pastikan semua kolom feature_cols ada di feature_df
        missing = [c for c in feature_cols if c not in feature_df.columns]
        if missing:
            st.warning(f"Beberapa fitur tidak ditemukan dalam ekstraksi: {missing}. Aplikasi akan mencoba melengkapi dengan 0.")
            for c in missing:
                feature_df[c] = 0.0

        X = feature_df[feature_cols].to_numpy().reshape(1, -1)

        # 2) Prediksi user (probabilitas)
        user_proba = model_user.predict_proba(X)[0]
        best_user = int(np.argmax(user_proba))
        conf_user = float(np.max(user_proba))

        # gunakan user_map jika tersedia
        if inv_user_map:
            predicted_user_label = inv_user_map.get(best_user, f"user{best_user}")
        else:
            predicted_user_label = "user1" if best_user == 0 else "user2"

        THRESHOLD = 0.60
        if conf_user < THRESHOLD:
            user_label = "anomali"
        else:
            user_label = predicted_user_label

        # 3) Prediksi status
        status_proba = model_status.predict_proba(X)[0]
        status_pred = int(np.argmax(status_proba))
        status_label = "buka" if status_pred == 0 else "tutup"
        conf_status = float(np.max(status_proba))

        # 4) Speech-to-text (opsional), ubah status jika kata jelas terdeteksi
        recognizer = sr.Recognizer()
        try:
            with sr.AudioFile(audio_path) as source:
                audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data, language="id-ID")
            st.markdown("Hasil Speech-to-Text")
            st.info(text)
            tl = text.lower()
            if "buka" in tl:
                status_label = "buka"
            elif "tutup" in tl:
                status_label = "tutup"
        except Exception:
            # silent fail
            pass

        # Display model results
        st.markdown("Hasil Identifikasi (Model ML)")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Deteksi Pengguna", user_label, f"Confidence {conf_user:.2f}")
        with c2:
            st.metric("Status Suara", status_label, f"Confidence {conf_status:.2f}")

        if user_label == "anomali":
            st.error("Suara tidak dikenali sebagai user1 atau user2.")
        else:
            if status_label == "buka":
                st.success(f"{user_label} terdeteksi sedang membuka.")
            else:
                st.warning(f"{user_label} terdeteksi sedang menutup.")

        # 5) Hitung DTW distances ke dataset MFCC dan tampilkan perhitungan
        st.markdown("Perhitungan DTW ke dataset (perbandingan)")

        # pastikan folder data_mfcc ada (buat jika perlu dari data/)
        prepare_data_mfcc()

        mfcc_new = extract_mfcc_timeseries(audio_path)
        df_dtw = compute_dtw_distances(mfcc_new)

        if df_dtw.empty:
            st.info("Tidak ada data MFCC (.npy) di folder data_mfcc/ untuk dibandingkan.")
        else:
            st.dataframe(df_dtw)
            # tampilkan 5 teratas
            top1 = df_dtw.iloc[0]
            st.info(f"Data paling mirip: {top1['file']} (DTW = {top1['dtw']:.2f})")

        # 6) Tampilkan fitur ringkas (opsional)
        with st.expander("Detail fitur untuk model (ringkasan)"):
            st.dataframe(feature_df.T)

    except Exception as e:
        st.error(f"Terjadi error pada proses audio: {e}")
        st.write(traceback.format_exc())

# ---------------------------
# UI: Rekam atau Upload
# ---------------------------
st.markdown("Pilih Metode Input")
method = st.radio("", ["Rekam suara", "Upload file (wav/m4a)"], horizontal=True)

if method == "Rekam suara":
    audio_bytes = st_audiorec()
    if audio_bytes:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            audio_path = tmp.name
        st.audio(audio_path)
        st.info("Memproses...")
        process_audio(audio_path)
else:
    uploaded = st.file_uploader("Upload file suara", type=["wav", "m4a"])
    if uploaded:
        suffix = os.path.splitext(uploaded.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            temp_path = tmp.name
        if suffix == ".m4a":
            wav_path = convert_m4a_to_wav(temp_path)
            if wav_path:
                st.audio(wav_path)
                process_audio(wav_path)
        else:
            st.audio(temp_path)
            process_audio(temp_path)
