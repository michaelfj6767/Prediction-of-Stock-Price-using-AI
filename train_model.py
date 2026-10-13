"""
train_model.py — Melatih model prediksi ARAH harga saham & menyimpannya.

>>> LENGKAPI bagian bertanda TODO. <<<
Dikerjakan di PERTEMUAN 2 (split & model) dan PERTEMUAN 3 (evaluasi).

Jalankan:  python train_model.py
Output  :  model/model_saham.joblib
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from utils import buat_fitur, buat_target, FITUR

DATA_PATH = Path("saham_contoh.csv")
MODEL_PATH = Path("model_saham.joblib")


def main() -> None:
    # 1) Muat data & buat fitur + target (memakai utils.py yang sudah kamu lengkapi)
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    df = buat_target(buat_fitur(df)).dropna().reset_index(drop=True)
    X, y = df[FITUR], df["target"]

    # TODO 1 (Pertemuan 2) — SPLIT KRONOLOGIS.
    #   Ambil 80% data PERTAMA (masa lalu) untuk latih, 20% terakhir untuk uji.
    #   JANGAN pakai train_test_split acak! Gunakan .iloc
    #   Petunjuk:
    #     n = int(len(df) * 0.8)
    #     X_tr, X_te = X.iloc[:n], X.iloc[n:]
    #     y_tr, y_te = y.iloc[:n], y.iloc[n:]
    n = int(len(df) * 0.8)
    X_tr, X_te = X.iloc[:n], X.iloc[n:]
    y_tr, y_te = y.iloc[:n], y.iloc[n:]

    # TODO 2 (Pertemuan 2) — Buat & latih model Random Forest.
    #   Petunjuk: RandomForestClassifier(n_estimators=200, random_state=0).fit(X_tr, y_tr)
    model = RandomForestClassifier(n_estimators=200, random_state=0).fit(X_tr, y_tr)     # <-- GANTI

    # TODO 3 (Pertemuan 2) — Hitung akurasi uji & BASELINE, lalu bandingkan.
    #   Petunjuk:
    #     acc = model.score(X_te, y_te)
    #     baseline = max(y_te.mean(), 1 - y_te.mean())
    acc = model.score(X_te, y_te)       # <-- GANTI
    baseline = max(y_te.mean(), 1-y_te.mean())  # <-- GANTI
    print(f"Akurasi model  : {acc:.3f}")
    print(f"Baseline (naif): {baseline:.3f}")
    print("Model mengalahkan baseline?", acc > baseline)
    print("(Wajar bila nyaris imbang — pasar memang sulit.)")

    # 4) Simpan model (sudah disediakan)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel disimpan ke: {MODEL_PATH}")


if __name__ == "__main__":
    main()
