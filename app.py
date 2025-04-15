import streamlit as st
import numpy as np
import os
import datetime
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import GRU  # Import GRU untuk mapping ke CuDNNGRU
import seq2tensor

# Konfigurasi
seq_size = 4000
embedding_dim = 12
embedding_path = "string_vec12.txt"
sequence_file = "output_sequences.tsv"
model_path = "best_model_overall.h5"
log_file = "riwayat_log.txt"

st.title("🧬 Prediksi Interaksi Protein-Protein dengan Metode PIPR")

# Jika memang hanya CPU yang tersedia, tetap pakai env variable ini
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Paksa penggunaan CPU

# Load model dengan custom_objects untuk menggantikan CuDNNGRU ke GRU (CPU-friendly)
if os.path.exists(model_path):
    model = load_model(model_path, custom_objects={'CuDNNGRU': GRU})

# Load embedding
if os.path.exists(embedding_path):
    embedding_model = seq2tensor.s2t(embedding_path)
    embedding_model.dim = embedding_dim

# Load sekuens protein
if os.path.exists(sequence_file):
    df_seq = pd.read_csv(sequence_file, sep="\t", header=None)
    id_to_seq = dict(zip(df_seq[0], df_seq[1]))
else:
    id_to_seq = {}

# Load riwayat dari file (jika ada)
if os.path.exists(log_file):
    with open(log_file, "r", encoding="utf-8") as f:
        history = [line.strip() for line in f.readlines() if line.strip()]
else:
    history = []

# Input ID pasangan protein
st.subheader("🧾 Masukkan ID Pasangan Protein (Pisahkan Dengan Koma)")
user_input = st.text_area(
    "📝 Masukkan ID Pasangan Protein",
    help="Pisahkan dua ID protein dalam satu pasangan dengan koma (,) dan pisahkan setiap pasangan dengan baris baru (enter)."
)


if st.button("🔍 Prediksi"):
    try:
        input_lines = user_input.strip().split("\n")  # Split input by newlines
        for line in input_lines:
            pair = [x.strip() for x in line.split(",")]  # Split pair by koma

            if len(pair) != 2:
                st.warning(f"❗ Format salah: {line}. Harap pisahkan dengan koma.")
                continue

            id1, id2 = pair
            seq1 = id_to_seq.get(id1)
            seq2 = id_to_seq.get(id2)

            if not seq1 or not seq2:
                st.warning(f"❗ Salah satu ID protein {id1} atau {id2} tidak ditemukan.")
                continue

            # Embedding & padding
            t1 = embedding_model.embed(seq1)
            t2 = embedding_model.embed(seq2)

            t1 = np.pad(t1[:, :embedding_dim], ((0, max(0, seq_size - len(t1))), (0, 0)), mode='constant')[:seq_size, :]
            t2 = np.pad(t2[:, :embedding_dim], ((0, max(0, seq_size - len(t2))), (0, 0)), mode='constant')[:seq_size, :]

            t1 = np.expand_dims(t1, axis=0)
            t2 = np.expand_dims(t2, axis=0)

            output = model.predict([t1, t2])[0]
            interaksi_prob = output[0]  # Karena softmax[0] = interaksi
            label = "Interaksi" if interaksi_prob >= 0.5 else "Tidak Berinteraksi"

            timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
            result = f"{timestamp} {id1} - {id2} ➜ {label} (Skor: {interaksi_prob:.4f})"

            # Simpan ke riwayat dan tampilkan
            history.insert(0, result)
            with open(log_file, "w", encoding="utf-8") as f:
                for line in history:
                    f.write(line + "\n")

            st.success(result)

    except Exception as e:
        st.error(f"❌ Error: {e}")

# Tombol hapus riwayat
if st.button("🗑️ Hapus Riwayat"):
    history = []
    if os.path.exists(log_file):
        os.remove(log_file)
    st.success("Riwayat berhasil dihapus.")

# Tampilkan riwayat
if history:
    st.markdown("---")
    st.markdown("### 📜 Riwayat Prediksi")
    for line in history:
        st.markdown(f"- {line}")
