import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Prediksi Harga Mobil Bekas dengan VAE", layout="centered")

st.title("Prediksi Harga Mobil Bekas dengan VAE")
st.write("Model: Variational Autoencoder")

CSV_PATH = "mobil_bekas_pertahun.csv"
LATENT_DIM = 2


# ---------------------------------------------------------------------------
# VAE building blocks
# ---------------------------------------------------------------------------
class Sampling(layers.Layer):
    """Reparameterization trick: z = mean + sigma * epsilon"""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(Model):
    def __init__(self, latent_dim):
        super().__init__()
        self.dense1 = layers.Dense(16, activation="relu")
        self.z_mean = layers.Dense(latent_dim)
        self.z_log_var = layers.Dense(latent_dim)
        self.sampling = Sampling()

    def call(self, x):
        h = self.dense1(x)
        z_mean = self.z_mean(h)
        z_log_var = self.z_log_var(h)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


class Decoder(Model):
    def __init__(self, output_dim):
        super().__init__()
        self.dense1 = layers.Dense(16, activation="relu")
        self.out = layers.Dense(output_dim, activation="sigmoid")

    def call(self, z):
        h = self.dense1(z)
        return self.out(h)


class VAE(Model):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            recon_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(data - reconstruction), axis=1)
            )
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1
                )
            )
            total_loss = recon_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {"loss": total_loss}


# ---------------------------------------------------------------------------
# Load data + train model (cached so it only runs once per deployment)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Melatih model VAE, mohon tunggu sebentar...")
def load_data_and_train():
    df = pd.read_csv(CSV_PATH)

    df["Harga"] = pd.to_numeric(df["Harga"], errors="coerce")
    df["Tahun"] = pd.to_numeric(df["Tahun"], errors="coerce")
    df = df.dropna(subset=["Tahun", "Harga"])

    # Buang data harga yang tidak wajar (hasil scraping yang kotor,
    # misalnya harga "51" atau "216" tanpa digit lengkap)
    df = df[(df["Harga"] >= 5_000_000) & (df["Harga"] <= 3_000_000_000)]
    df = df[(df["Tahun"] >= 2003) & (df["Tahun"] <= 2024)]

    X = df[["Tahun", "Harga"]].values.astype("float32")

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    encoder = Encoder(LATENT_DIM)
    decoder = Decoder(output_dim=X.shape[1])
    vae = VAE(encoder, decoder)
    vae.compile(optimizer="adam")
    vae.fit(X_scaled, epochs=30, batch_size=32, verbose=0)

    avg_per_year = df.groupby("Tahun")["Harga"].mean()

    return vae, scaler, avg_per_year


vae, scaler, avg_per_year = load_data_and_train()

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
tahun = st.slider("Pilih Tahun Produksi Mobil", 2003, 2024, 2015)
fitur_tambahan = st.number_input(
    "Masukkan Perkiraan Harga Mobil (Rp)",
    min_value=0,
    value=100_000_000,
    step=5_000_000,
)

if tahun in avg_per_year.index:
    st.caption(f"Rata-rata harga pasar untuk tahun {tahun}: Rp {avg_per_year[tahun]:,.0f}")

if st.button("Prediksi Harga"):
    input_data = np.array([[tahun, fitur_tambahan]], dtype="float32")
    scaled_input = scaler.transform(input_data)

    # Encode input pengguna ke latent space, lalu decode kembali.
    # Menggunakan z_mean (bukan sampling acak) agar hasil deterministik.
    z_mean, _, _ = vae.encoder(scaled_input)
    reconstruction = vae.decoder(z_mean)

    harga_prediksi = scaler.inverse_transform(reconstruction.numpy())[0][1]

    st.success(f"Perkiraan Harga Mobil: Rp {harga_prediksi:,.0f}")
