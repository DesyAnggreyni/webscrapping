import streamlit as st
import numpy as np


# Judul
st.title("Prediksi Harga Mobil Bekas dengan VAE")
st.write("Model: Variational Autoencoder")

# Input dari user
tahun = st.slider("Pilih Tahun Produksi Mobil", 2003, 2024, 2015)
fitur_tambahan = st.number_input("Masukkan Fitur Tambahan (contoh: nama_mobil)", value=50000)

# Button prediksi
if st.button("Prediksi Harga"):
    # Gabungkan input menjadi array fitur
    input_data = np.array([[tahun, fitur_tambahan]])

    # Lakukan scaling (jika perlu)
    scaled_data = scaler.transform(input_data)

    # Prediksi harga (generate dari decoder)
    z_points = np.random.normal(size=(1, model.input_shape[1]))  # sampling dari latent space
    pred_scaled = model.predict(z_points)

    # Inverse scaling
    harga_prediksi = scaler.inverse_transform(pred_scaled)

    st.success(f"Perkiraan Harga Mobil: Rp {harga_prediksi[0][0]:,.0f}")
