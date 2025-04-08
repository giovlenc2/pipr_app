import streamlit as st
import numpy as np
import os
import datetime
import pandas as pd
import h5py
import json
import keras.engine.saving
from keras.models import model_from_json
from keras.optimizers import RMSprop
import seq2tensor

# 🛠️ PATCH 1 – json.loads decode bypass
original_json_loads = json.loads
def safe_loads(s):
    try:
        return original_json_loads(s.decode('utf-8'))
    except AttributeError:
        return original_json_loads(s)
json.loads = safe_loads

# 🛠️ PATCH 2 – load_weights decode bypass
original_loader = keras.engine.saving.load_weights_from_hdf5_group
def safe_load_weights(f, layers, **kwargs):
    if isinstance(f.attrs['keras_version'], bytes):
        f.attrs.modify('keras_version', f.attrs['keras_version'].decode('utf-8'))
    if isinstance(f.attrs['backend'], bytes):
        f.attrs.modify('backend', f.attrs['backend'].decode('utf-8'))
    return original_loader(f, layers, **kwargs)
keras.engine.saving.load_weights_from_hdf5_group = safe_load_weights

# Konfigurasi
seq_size = 4000
embedding_dim = 12
embedding_path = "string_vec12.txt"
sequence_file = "output_sequences.tsv"
model_path = "best_model_overall.h5"
log_file = "riwayat_log.txt"

st.title("🧬 Prediksi Interaksi Protein dengan Metode PIPR")

# Load model dari HDF5 (.h5) secara manual
if os.path.exists(model_path):
    with h5py.File(model_path, 'r') as f:
        model_config = f.attrs.get('model_config')
        if isinstance(model_config, bytes):
            model_config = model_config.decode('utf-8')
        model = model_from_json(model_config)
        model.load_weights(model_path)
        model.compile(optimizer=RMSprop(), loss='categorical_crossentropy', metrics=['accuracy'])

# Load embedding
if os.path.exists(embedding_path):
    embedding_model = seq2tensor.s2t(embedding_path)
    embedding_model.dim = embedding_dim

# Load sekuens protein
if os.path.exists(sequence_file):
    df_seq = pd.read_csv(sequence_file, sep="\t", header=None)
    id_to_seq = dict(zip(df_seq[0], df_seq[1]))

# Load riwayat dari file (jika ada)
if os.path.exists(log_file):
    with open(log_file, "r", encoding="utf-8") as f:
        history = [line.strip() for line in f.readlines() if line.strip()]
else:
    history = []

# Input ID pasangan protein
st.subheader("🧾 Masukkan ID Pasangan Protein (pisahkan dengan koma)")
user_input = st.text_input("Contoh: 9606.ENSP00000232892, 9606.ENSP00000353720")

if st.button("🔍 Prediksi"):
    try:
        id1, id2 = [x.strip() for x in user_input.split(",")]
        seq1 = id_to_seq.get(id1)
        seq2 = id_to_seq.get(id2)

        if not seq1 or not seq2:
            st.warning("❗ Salah satu ID protein tidak ditemukan.")
        else:
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

    except:
        st.error("❌ Format input tidak valid. Harap pisahkan dengan koma.")

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
