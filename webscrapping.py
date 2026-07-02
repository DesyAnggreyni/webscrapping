import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Prediksi Harga Mobil Bekas dengan VAE", layout="centered")

st.title("Prediksi Harga Mobil Bekas dengan VAE")
st.write("Model: Variational Autoencoder (implementasi NumPy, tanpa TensorFlow)")

CSV_PATH = "mobil_bekas_pertahun.csv"


# ---------------------------------------------------------------------------
# VAE kecil, murni NumPy (tidak butuh TensorFlow -> aman untuk semua versi
# Python yang dipakai Streamlit Cloud, termasuk versi terbaru yang belum
# punya wheel TensorFlow).
# ---------------------------------------------------------------------------
class SimpleVAE:
    def __init__(self, input_dim=2, hidden_dim=8, latent_dim=2, seed=42):
        rng = np.random.default_rng(seed)

        def init(fan_in, fan_out):
            limit = np.sqrt(6 / (fan_in + fan_out))
            return rng.uniform(-limit, limit, size=(fan_in, fan_out))

        self.W1, self.b1 = init(input_dim, hidden_dim), np.zeros(hidden_dim)
        self.Wmu, self.bmu = init(hidden_dim, latent_dim), np.zeros(latent_dim)
        self.Wlv, self.blv = init(hidden_dim, latent_dim), np.zeros(latent_dim)
        self.W2, self.b2 = init(latent_dim, hidden_dim), np.zeros(hidden_dim)
        self.W3, self.b3 = init(hidden_dim, input_dim), np.zeros(input_dim)

        self.params = ["W1", "b1", "Wmu", "bmu", "Wlv", "blv", "W2", "b2", "W3", "b3"]
        self.m = {p: np.zeros_like(getattr(self, p)) for p in self.params}
        self.v = {p: np.zeros_like(getattr(self, p)) for p in self.params}
        self.t = 0

    @staticmethod
    def _relu(x):
        return np.maximum(0, x)

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -30, 30)))

    def encode(self, X):
        a1 = X @ self.W1 + self.b1
        h_enc = self._relu(a1)
        mu = h_enc @ self.Wmu + self.bmu
        logvar = np.clip(h_enc @ self.Wlv + self.blv, -10, 10)
        return mu, logvar, h_enc, a1

    def decode(self, z):
        a2 = z @ self.W2 + self.b2
        h_dec = self._relu(a2)
        a3 = h_dec @ self.W3 + self.b3
        out = self._sigmoid(a3)
        return out, h_dec, a2, a3

    def forward(self, X, rng):
        mu, logvar, h_enc, a1 = self.encode(X)
        std = np.exp(0.5 * logvar)
        eps = rng.standard_normal(size=mu.shape)
        z = mu + std * eps
        out, h_dec, a2, a3 = self.decode(z)
        return dict(X=X, mu=mu, logvar=logvar, h_enc=h_enc, a1=a1, std=std,
                    eps=eps, z=z, out=out, h_dec=h_dec, a2=a2, a3=a3)

    def backward(self, cache, beta=0.0005):
        X, out = cache["X"], cache["out"]
        N = X.shape[0]
        mu, logvar, std, eps, z = cache["mu"], cache["logvar"], cache["std"], cache["eps"], cache["z"]
        h_enc, a1 = cache["h_enc"], cache["a1"]
        h_dec, a2 = cache["h_dec"], cache["a2"]

        dloss_dout = 2 * (out - X) / N
        da3 = dloss_dout * out * (1 - out)
        dW3 = h_dec.T @ da3
        db3 = da3.sum(axis=0)
        dh_dec = da3 @ self.W3.T

        da2 = dh_dec * (a2 > 0)
        dW2 = z.T @ da2
        db2 = da2.sum(axis=0)
        dz = da2 @ self.W2.T

        dmu = dz + beta * mu / N
        dlogvar = dz * 0.5 * std * eps + beta * 0.5 * (np.exp(logvar) - 1) / N

        dWmu = h_enc.T @ dmu
        dbmu = dmu.sum(axis=0)
        dWlv = h_enc.T @ dlogvar
        dblv = dlogvar.sum(axis=0)

        dh_enc = dmu @ self.Wmu.T + dlogvar @ self.Wlv.T
        da1 = dh_enc * (a1 > 0)
        dW1 = X.T @ da1
        db1 = da1.sum(axis=0)

        return dict(W1=dW1, b1=db1, Wmu=dWmu, bmu=dbmu, Wlv=dWlv, blv=dblv,
                    W2=dW2, b2=db2, W3=dW3, b3=db3)

    def step(self, grads, lr=0.005, beta1=0.9, beta2=0.999, eps=1e-8, clip=5.0):
        self.t += 1
        for p in self.params:
            g = np.clip(grads[p], -clip, clip)
            self.m[p] = beta1 * self.m[p] + (1 - beta1) * g
            self.v[p] = beta2 * self.v[p] + (1 - beta2) * (g ** 2)
            m_hat = self.m[p] / (1 - beta1 ** self.t)
            v_hat = self.v[p] / (1 - beta2 ** self.t)
            update = lr * m_hat / (np.sqrt(v_hat) + eps)
            setattr(self, p, getattr(self, p) - update)

    def fit(self, X, epochs=500, lr=0.005, beta=0.0005, seed=0):
        rng = np.random.default_rng(seed)
        for _ in range(epochs):
            cache = self.forward(X, rng)
            grads = self.backward(cache, beta=beta)
            self.step(grads, lr=lr)
        return self

    def predict_deterministic(self, X):
        mu, _, _, _ = self.encode(X)
        out, *_ = self.decode(mu)
        return out


# ---------------------------------------------------------------------------
# Load data + train model (cached, hanya sekali per deployment)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Melatih model VAE, mohon tunggu sebentar...")
def load_data_and_train():
    df = pd.read_csv(CSV_PATH)

    df["Harga"] = pd.to_numeric(df["Harga"], errors="coerce")
    df["Tahun"] = pd.to_numeric(df["Tahun"], errors="coerce")
    df = df.dropna(subset=["Tahun", "Harga"])

    # Buang data harga yang tidak wajar (hasil scraping yang kotor)
    df = df[(df["Harga"] >= 5_000_000) & (df["Harga"] <= 3_000_000_000)]
    df = df[(df["Tahun"] >= 2003) & (df["Tahun"] <= 2024)]

    X = df[["Tahun", "Harga"]].values.astype("float64")

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    vae = SimpleVAE(input_dim=2, hidden_dim=8, latent_dim=2, seed=42)
    vae.fit(X_scaled, epochs=500, lr=0.005, beta=0.0005, seed=0)

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
    input_data = np.array([[tahun, fitur_tambahan]], dtype="float64")
    scaled_input = scaler.transform(input_data)

    reconstruction = vae.predict_deterministic(scaled_input)
    harga_prediksi = scaler.inverse_transform(reconstruction)[0][1]

    st.success(f"Perkiraan Harga Mobil: Rp {harga_prediksi:,.0f}")
