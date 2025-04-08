import streamlit as st
import numpy as np
import os
import datetime
import pandas as pd
from keras.models import model_from_json
from keras.layers import GRU
import seq2tensor

# Konfigurasi
seq_size = 4000
embedding_dim = 12
embedding_path = "string_vec12.txt"
sequence_file = "output_sequences.tsv"
log_file = "riwayat_log.txt"

# ====== Load Model dari JSON dan Weights (CPU Friendly) ======
with open("model.json", "r") as f:
    model = model_from_json(f.read(), custom_objects={"CuDNNGRU": GRU})
model.load_weights("model_weights.h5")

# ====== Load Embedding & Mapping ======
embedding_model = seq2tensor.s2t(embedding_path)
embedding_model.dim = embedding_dim

df_seq = pd.read_csv(sequence_file, sep="\t", header=None)
id_to_seq = dict(zip(df_seq[0], df_seq[1]))

# ====== Riwayat ======
if os.path.exists(log_file):
    with open(log_file, "r", encoding="utf-8") as f:
        history = [line.strip() for line in f.readlines() if line.strip()]
else:
    history = []

# ====== Streamlit App ======
st.title("🧬 Prediksi Interaksi Protein dengan Metode PIPR")

st.subheader("🧾 Masukkan ID Pasangan Protein (pisahkan dengan koma atau baris)")
user_input = st.text_area("Contoh:\n9606.ENSP00000232892, 9606.ENSP00000353720\n9606.ENSP00000298492, 9606.ENSP00000352408")

if st.button("🔍 Prediksi"):
    lines = user_input.strip().splitlines()
    for line in lines:
        try:
            id1, id2 = [x.strip() for x in line.split(",")]
            seq1, seq2 = id_to_seq.get(id1), id_to_seq.get(id2)

            if not seq1 or not seq2:
                st.warning(f"❗ ID tidak ditemukan: {id1} atau {id2}")
                continue

            # Embedding
            t1 = embedding_model.embed(seq1)
            t2 = embedding_model.embed(seq2)

            # Padding
            t1 = np.pad(t1[:, :embedding_dim], ((0, max(0, seq_size - len(t1))), (0, 0)), mode='constant')[:seq_size]
            t2 = np.pad(t2[:, :embedding_dim], ((0, max(0, seq_size - len(t2))), (0, 0)), mode='constant')[:seq_size]

            t1 = np.expand_dims(t1, axis=0)
            t2 = np.expand_dims(t2, axis=0)

            output = model.predict([t1, t2])[0]
            interaksi_prob = output[0]  # softmax[0] = interaksi
            label = "Interaksi" if interaksi_prob >= 0.5 else "Tidak Berinteraksi"

            timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
            result = f"{timestamp} {id1} - {id2} ➜ {label} (Skor: {interaksi_prob:.4f})"

            history.insert(0, result)
            st.success(result)
        except:
            st.error(f"❌ Format tidak valid: {line}")

    # Simpan riwayat
    with open(log_file, "w", encoding="utf-8") as f:
        for line in history:
            f.write(line + "\n")

# ====== Tombol Hapus Riwayat ======
if st.button("🗑️ Hapus Riwayat"):
    history = []
    if os.path.exists(log_file):
        os.remove(log_file)
    st.success("Riwayat berhasil dihapus.")

# ====== Tampilkan Riwayat ======
if history:
    st.markdown("---")
    st.markdown("### 📜 Riwayat Prediksi")
    for line in history:
        st.markdown(f"- {line}")
