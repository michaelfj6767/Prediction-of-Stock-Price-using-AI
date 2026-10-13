"""
app.py — Aplikasi web prediksi ARAH harga saham.

Jalankan lokal:  streamlit run app.py

⚠️ EDUKASI, BUKAN SARAN INVESTASI.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from utils import buat_fitur, buat_target, FITUR

MODEL_PATH = Path("model/model_saham.joblib")
DATA_PATH = Path("data/saham_contoh.csv")

st.set_page_config(page_title="Prediksi Arah Saham", page_icon="📈", layout="centered")


# --- Pemuatan data sudah disediakan (yfinance online, fallback data contoh) ---
@st.cache_data(show_spinner=False)
def ambil_data(ticker: str):
    try:
        import yfinance as yf
        df = yf.download(ticker, start="2020-01-01", auto_adjust=True, progress=False)
        if len(df) > 50:
            df = df.reset_index()[["Date", "Close"]]
            df.columns = ["Date", "Close"]
            return df, "yfinance"
    except Exception:
        pass
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    return df, "contoh"


# ------------------------------------------------------------------ tampilan
st.title("📈 Prediksi Arah Harga Saham")
st.warning("⚠️ **Materi edukasi, bukan saran investasi.**")

# TODO 1 (Pertemuan 4) — Muat model dari MODEL_PATH memakai joblib.load(...)
#   Kalau file model belum ada, tampilkan pesan & st.stop().
model = load_model()
if model is None:
    st.error("Model belum ada. Jalankan dulu: python train_model.py"); st.stop()



ticker = st.text_input("Kode saham (contoh: BBCA.JK, TLKM.JK, AAPL)", value="BBCA.JK")
df, sumber = ambil_data(ticker)
if sumber == "contoh":
    st.info("Memakai data contoh (offline). Di Streamlit Cloud, data nyata otomatis via yfinance.")

tab1, tab2, tab3 = st.tabs(["🔮 Prediksi", "📊 Backtest", "ℹ️ Tentang"])
feat = buat_fitur(df).dropna().reset_index(drop=True)
X_terbaru = feat[FITUR].iloc[[-1]]        # baris paling akhir
proba = model.predict_proba(X_terbaru)[0] # [prob turun, prob naik]
arah = "NAIK" if proba[1] >= 0.5 else "TURUN"
yakin = float(max(proba))
# ---------------- Tab 1: Prediksi ----------------
with tab1:
    st.subheader(f"Prediksi hari berikutnya — {ticker}")

    # TODO 2 (Pertemuan 4) — Hitung fitur data terbaru & prediksi arahnya.
    #   Langkah:
    
    c1, c2 = st.columns(2)
    if arah == "NAIK":
        c1.success(f"### 📈 {arah}")
    elif arah == "TURUN":
        c1.error(f"### 📉 {arah}")
    else:
        c1.write("### (lengkapi TODO 2)")
    c2.metric("Keyakinan model", f"{yakin*100:.1f}%")

    # TODO 3 (Pertemuan 4) — Tampilkan grafik harga 120 hari terakhir.
    st.line_chart(df.set_index("Date")["Close"].tail(120))

# ---------------- Tab 2: Backtest ----------------
with tab2:
    st.subheader("Backtest — strategi model vs beli & tahan")

    # TODO 4 (Pertemuan 5) — Backtest sederhana.
    #   Langkah:
    d = buat_target(buat_fitur(df)).dropna().reset_index(drop=True)
    n = int(len(d) * 0.8)
    proba = model.predict_proba(d[FITUR].iloc[n:])[:, 1]
    signal = (proba >= 0.5).astype(int)
    ret = d["Close"].pct_change().shift(-1).iloc[n:].values
    biaya = 0.001
    trades = np.abs(np.diff(np.concatenate([[0], signal])))
    strat = signal * ret - trades * biaya
    eq = pd.DataFrame({
        "Beli & tahan": np.cumprod(1 + np.nan_to_num(ret)),
        "Strategi model (-biaya)": np.cumprod(1 + np.nan_to_num(strat)),
        })
    st.line_chart(eq)
    st.info("Lengkapi TODO 4 (Pertemuan 5) untuk menampilkan backtest.")

# ---------------- Tab 3: Tentang ----------------
with tab3:
    st.markdown(
        """
        **Cara kerja:** harga → fitur teknikal (return, MA, RSI) → Random Forest
        memprediksi arah (naik/turun), dilatih dengan **split kronologis**.

        Akurasi biasanya hanya ~50–53% — harga saham sangat sulit diprediksi.
        Nilai proyek ada pada **metodologi yang benar**, bukan akurasi tinggi.

        ⚠️ Edukasi, bukan saran investasi.
        """
    )
